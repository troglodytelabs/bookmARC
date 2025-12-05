"""
Text preprocessing: chunking, stopwords removal, tokenization.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, explode
from pyspark.sql.types import (
    ArrayType,
    StringType,
    IntegerType,
    StructType,
    StructField,
)
import re
import nltk

# Download required NLTK data (stopwords are downloaded inside UDF if needed)
# Module-level download ensures availability before UDF execution
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

# Download wordnet for lemmatization
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)


def load_books(
    spark: SparkSession,
    books_dir: str,
    metadata_path: str,
    language: str = "en",
    limit: int = None,
):
    """
    Load books from directory and join with metadata.

    Args:
        spark: SparkSession
        books_dir: Directory containing book files
        metadata_path: Path to metadata CSV
        language: Filter by language (default: "en")
        limit: Limit number of books to process (for testing)

    Returns:
        DataFrame with columns: book_id, title, author, text
    """
    # Load metadata
    metadata_df = spark.read.option("header", "true").csv(metadata_path)

    # Filter by language
    if language:
        metadata_df = metadata_df.filter(col("Language") == language)

    # Select relevant columns
    metadata_df = metadata_df.select(
        col("Etext Number").alias("book_id"),
        col("Title").alias("title"),
        col("Authors").alias("author"),
    )

    if limit:
        metadata_df = metadata_df.limit(limit)

    # Load book texts
    def read_book_text(book_id: str) -> str:
        """Read book text from file."""
        try:
            book_path = f"{books_dir}/{book_id}"
            with open(book_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
                # Remove Project Gutenberg headers/footers
                # Remove content between *** START and *** END markers
                text = re.sub(r"\*\*\* START.*?\*\*\*", "", text, flags=re.DOTALL)
                text = re.sub(r"\*\*\* END.*?\*\*\*", "", text, flags=re.DOTALL)
                return text
        except Exception:
            return ""

    # Create RDD of (book_id, text) pairs
    book_ids = [row["book_id"] for row in metadata_df.select("book_id").collect()]
    book_texts = spark.sparkContext.parallelize(book_ids).map(
        lambda bid: (bid, read_book_text(bid))
    )

    # Convert to DataFrame
    from pyspark.sql.types import StructType, StructField, StringType

    schema = StructType(
        [
            StructField("book_id", StringType(), True),
            StructField("text", StringType(), True),
        ]
    )
    books_df = spark.createDataFrame(book_texts, schema)

    # Join with metadata
    result_df = metadata_df.join(books_df, on="book_id", how="inner")

    return result_df


# UDFs are now defined inside create_chunks_df to avoid module serialization issues


def create_chunks_df(
    spark: SparkSession, books_df, num_chunks: int = 20, max_chunk_size: int = 20000
):
    """
    Create chunks from books DataFrame using percentage-based chunking with max size limit.

    Args:
        spark: SparkSession
        books_df: DataFrame with columns: book_id, title, author, text
        num_chunks: Target number of chunks to create (may be adjusted based on max_chunk_size)
        max_chunk_size: Maximum characters per chunk to prevent memory issues (default: 20000)

    Returns:
        DataFrame with columns: book_id, title, author, chunk_index, chunk_text, word
    """

    # Define chunk creation function with num_chunks and max_chunk_size captured in closure
    # This must be self-contained for Spark serialization
    def make_chunk_udf(n_chunks, max_size):
        """Factory function to create chunk UDF with captured parameters."""

        def _create_chunks_wrapper(text):
            """Wrapper function for UDF that creates percentage-based chunks with size limits."""
            try:
                if not text:
                    return []

                text_len = len(text)
                if text_len == 0:
                    return []

                # Calculate ideal chunk size based on percentage
                ideal_chunk_size = (
                    max(1, text_len // n_chunks) if n_chunks > 0 else text_len
                )

                # For very short texts, ensure minimum chunk size to avoid too many tiny chunks
                min_chunk_size = 100
                if text_len < min_chunk_size * n_chunks:
                    # Adjust num_chunks for very short texts
                    actual_num_chunks = (
                        max(1, text_len // min_chunk_size) if min_chunk_size > 0 else 1
                    )
                    chunk_size = (
                        max(1, text_len // actual_num_chunks)
                        if actual_num_chunks > 0
                        else text_len
                    )
                else:
                    # For large texts, limit chunk size to prevent memory issues
                    if ideal_chunk_size > max_size:
                        # Recalculate actual number of chunks based on max_chunk_size
                        actual_num_chunks = (
                            max(n_chunks, (text_len + max_size - 1) // max_size)
                            if max_size > 0
                            else n_chunks
                        )
                        chunk_size = max_size
                    else:
                        actual_num_chunks = n_chunks
                        chunk_size = ideal_chunk_size

                # Ensure we have at least 1 chunk
                actual_num_chunks = max(1, actual_num_chunks)
                chunk_size = max(1, chunk_size)

                chunks = []
                for i in range(actual_num_chunks):
                    start_idx = i * chunk_size
                    # Last chunk gets all remaining text
                    if i == actual_num_chunks - 1:
                        end_idx = text_len
                    else:
                        end_idx = min(start_idx + chunk_size, text_len)

                    # Ensure we don't go out of bounds
                    if start_idx < text_len:
                        chunks.append(
                            {"chunk_index": i, "chunk_text": text[start_idx:end_idx]}
                        )

                return chunks if chunks else [{"chunk_index": 0, "chunk_text": text}]
            except Exception:
                # Fallback: return entire text as single chunk if anything goes wrong
                return [{"chunk_index": 0, "chunk_text": str(text) if text else ""}]

        return udf(
            _create_chunks_wrapper,
            ArrayType(
                StructType(
                    [
                        StructField("chunk_index", IntegerType(), True),
                        StructField("chunk_text", StringType(), True),
                    ]
                )
            ),
        )

    # Create UDF with num_chunks and max_chunk_size
    chunk_udf = make_chunk_udf(num_chunks, max_chunk_size)

    # Define preprocessing function - completely self-contained, defined inside this function
    def _preprocess_wrapper(text):
        """Wrapper function for UDF that handles preprocessing.
        All imports are inside to ensure they're available on worker nodes.
        """
        if not text:
            return []

        # Import inside function to ensure availability on workers
        import re
        import nltk

        # Get stopwords - download if needed
        try:
            from nltk.corpus import stopwords

            try:
                nltk.data.find("corpora/stopwords")
            except LookupError:
                nltk.download("stopwords", quiet=True)
            stop_words = set(stopwords.words("english"))
        except Exception:
            # Fallback if nltk not available
            stop_words = set()

        # Convert to lowercase
        text_lower = text.lower()

        # Remove special characters but keep apostrophes for contractions
        text_clean = re.sub(r"[^a-z0-9\s']", " ", text_lower)

        # Handle contractions to avoid matching partial words
        text_clean = re.sub(r"'s\b", "", text_clean)  # Remove 's
        text_clean = re.sub(r"n't\b", " not", text_clean)  # isn't -> is not
        text_clean = re.sub(r"'re\b", " are", text_clean)
        text_clean = re.sub(r"'ve\b", " have", text_clean)
        text_clean = re.sub(r"'ll\b", " will", text_clean)
        text_clean = re.sub(r"'d\b", " would", text_clean)
        text_clean = re.sub(r"'", "", text_clean)  # Remove remaining apostrophes

        # Tokenize
        words = text_clean.split()

        # Remove stopwords and very short words
        words = [w for w in words if w not in stop_words and len(w) > 2]

        return words

    # Create preprocessing UDF - defined inside function to avoid module serialization
    preprocess_udf = udf(_preprocess_wrapper, ArrayType(StringType()))

    # Create chunks
    chunks_df = books_df.select(
        col("book_id"),
        col("title"),
        col("author"),
        explode(chunk_udf(col("text"))).alias("chunk"),
    )

    # Extract chunk_index and chunk_text
    chunks_df = chunks_df.select(
        col("book_id"),
        col("title"),
        col("author"),
        col("chunk.chunk_index").alias("chunk_index"),
        col("chunk.chunk_text").alias("chunk_text"),
    )

    # Preprocess chunks to get words
    chunks_df = chunks_df.withColumn("words", preprocess_udf(col("chunk_text")))

    # Explode words - one row per word
    chunks_df = chunks_df.select(
        col("book_id"),
        col("title"),
        col("author"),
        col("chunk_index"),
        col("chunk_text"),
        explode(col("words")).alias("word"),
    )

    return chunks_df
