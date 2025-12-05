"""
Recommendation system based on emotion trajectory similarity.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pow, udf, lit
from pyspark.sql.types import DoubleType
import math


def compute_trajectory_similarity(traj1, traj2):
    """
    Compute similarity between two emotion trajectories.
    Uses cosine similarity or Euclidean distance.

    Args:
        traj1: First trajectory (array of emotion scores)
        traj2: Second trajectory (array of emotion scores)

    Returns:
        Similarity score (higher = more similar)
    """
    if not traj1 or not traj2:
        return 0.0

    # Flatten trajectories if needed
    if isinstance(traj1[0], list):
        traj1 = [item for sublist in traj1 for item in sublist]
    if isinstance(traj2[0], list):
        traj2 = [item for sublist in traj2 for item in sublist]

    # Pad to same length
    max_len = max(len(traj1), len(traj2))
    traj1 = traj1 + [0.0] * (max_len - len(traj1))
    traj2 = traj2 + [0.0] * (max_len - len(traj2))

    # Compute cosine similarity
    dot_product = sum(a * b for a, b in zip(traj1, traj2))
    norm1 = math.sqrt(sum(a * a for a in traj1))
    norm2 = math.sqrt(sum(b * b for b in traj2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    similarity = dot_product / (norm1 * norm2)
    return float(similarity)


def compute_feature_similarity(features1, features2):
    """
    Compute similarity based on aggregated features.

    Args:
        features1: Dict with emotion statistics
        features2: Dict with emotion statistics

    Returns:
        Similarity score
    """
    # Extract key features
    feature_names = [
        "avg_anger",
        "avg_joy",
        "avg_fear",
        "avg_sadness",
        "avg_valence",
        "avg_arousal",
        "avg_dominance",
        "valence_std",
        "arousal_std",
    ]

    similarity = 0.0
    count = 0

    for feat in feature_names:
        if feat in features1 and feat in features2:
            val1 = features1[feat] or 0.0
            val2 = features2[feat] or 0.0
            # Use 1 - normalized difference as similarity
            diff = abs(val1 - val2)
            max_val = max(abs(val1), abs(val2), 1.0)
            similarity += 1.0 - (diff / max_val)
            count += 1

    return similarity / count if count > 0 else 0.0


def recommend_books(
    spark: SparkSession, trajectory_df, liked_book_id: str, top_n: int = 10
):
    """
    Recommend books with similar emotion trajectories.

    Args:
        spark: SparkSession
        trajectory_df: DataFrame with book trajectories
        liked_book_id: Book ID that user likes
        top_n: Number of recommendations

    Returns:
        DataFrame with recommended books and similarity scores
    """
    # Get liked book trajectory
    liked_book = trajectory_df.filter(col("book_id") == liked_book_id).collect()

    if not liked_book:
        return spark.createDataFrame([], trajectory_df.schema)

    liked_row = liked_book[0]

    # Compute similarity for all other books
    similarity_udf = udf(
        lambda traj: compute_trajectory_similarity(
            liked_row["emotion_trajectory"], traj
        ),
        DoubleType(),
    )

    # Add similarity scores
    recommendations = (
        trajectory_df.filter(col("book_id") != liked_book_id)
        .withColumn("similarity", similarity_udf(col("emotion_trajectory")))
        .orderBy(col("similarity").desc())
        .limit(top_n)
    )

    return recommendations.select(
        "book_id",
        "title",
        "author",
        "similarity",
        "avg_joy",
        "avg_sadness",
        "avg_fear",
        "avg_anger",
        "avg_valence",
        "avg_arousal",
    )


def recommend_by_features(
    spark: SparkSession, trajectory_df, liked_book_id: str, top_n: int = 10
):
    """
    Recommend books based on aggregated feature similarity.
    This is faster than trajectory similarity.

    Uses normalized Euclidean distance for similarity.
    Features are normalized to 0-1 range to handle different scales.

    Args:
        spark: SparkSession
        trajectory_df: DataFrame with book trajectories
        liked_book_id: Book ID that user likes
        top_n: Number of recommendations

    Returns:
        DataFrame with recommended books
    """
    from pyspark.sql.functions import sqrt, min as spark_min, max as spark_max

    # Get liked book features
    liked_book = trajectory_df.filter(col("book_id") == liked_book_id).collect()

    if not liked_book:
        return spark.createDataFrame([], trajectory_df.schema)

    liked_row = liked_book[0]

    # Extract feature values (handle None)
    def get_val(val):
        return val if val is not None else 0.0

    # Compute min/max for normalization (exclude the liked book to avoid bias)
    other_books = trajectory_df.filter(col("book_id") != liked_book_id)

    # Get ranges for normalization (all 8 Plutchik emotions + VAD)
    stats = other_books.agg(
        # All 8 Plutchik emotions
        spark_min("avg_anger").alias("min_anger"),
        spark_max("avg_anger").alias("max_anger"),
        spark_min("avg_anticipation").alias("min_anticipation"),
        spark_max("avg_anticipation").alias("max_anticipation"),
        spark_min("avg_disgust").alias("min_disgust"),
        spark_max("avg_disgust").alias("max_disgust"),
        spark_min("avg_fear").alias("min_fear"),
        spark_max("avg_fear").alias("max_fear"),
        spark_min("avg_joy").alias("min_joy"),
        spark_max("avg_joy").alias("max_joy"),
        spark_min("avg_sadness").alias("min_sadness"),
        spark_max("avg_sadness").alias("max_sadness"),
        spark_min("avg_surprise").alias("min_surprise"),
        spark_max("avg_surprise").alias("max_surprise"),
        spark_min("avg_trust").alias("min_trust"),
        spark_max("avg_trust").alias("max_trust"),
        # VAD scores
        spark_min("avg_valence").alias("min_valence"),
        spark_max("avg_valence").alias("max_valence"),
        spark_min("avg_arousal").alias("min_arousal"),
        spark_max("avg_arousal").alias("max_arousal"),
        spark_min("avg_dominance").alias("min_dominance"),
        spark_max("avg_dominance").alias("max_dominance"),
    ).first()

    # Normalize liked book features
    def normalize(val, min_val, max_val):
        if min_val is None or max_val is None:
            return 0.5
        if max_val == min_val:
            return 0.5  # If all values are same, use middle
        return (val - min_val) / (max_val - min_val)

    # Get min/max values with defaults (all 8 Plutchik emotions + VAD)
    min_anger = stats["min_anger"] if stats["min_anger"] is not None else 0.0
    max_anger = stats["max_anger"] if stats["max_anger"] is not None else 1.0
    min_anticipation = (
        stats["min_anticipation"] if stats["min_anticipation"] is not None else 0.0
    )
    max_anticipation = (
        stats["max_anticipation"] if stats["max_anticipation"] is not None else 1.0
    )
    min_disgust = stats["min_disgust"] if stats["min_disgust"] is not None else 0.0
    max_disgust = stats["max_disgust"] if stats["max_disgust"] is not None else 1.0
    min_fear = stats["min_fear"] if stats["min_fear"] is not None else 0.0
    max_fear = stats["max_fear"] if stats["max_fear"] is not None else 1.0
    min_joy = stats["min_joy"] if stats["min_joy"] is not None else 0.0
    max_joy = stats["max_joy"] if stats["max_joy"] is not None else 1.0
    min_sadness = stats["min_sadness"] if stats["min_sadness"] is not None else 0.0
    max_sadness = stats["max_sadness"] if stats["max_sadness"] is not None else 1.0
    min_surprise = stats["min_surprise"] if stats["min_surprise"] is not None else 0.0
    max_surprise = stats["max_surprise"] if stats["max_surprise"] is not None else 1.0
    min_trust = stats["min_trust"] if stats["min_trust"] is not None else 0.0
    max_trust = stats["max_trust"] if stats["max_trust"] is not None else 1.0
    min_valence = stats["min_valence"] if stats["min_valence"] is not None else -1.0
    max_valence = stats["max_valence"] if stats["max_valence"] is not None else 1.0
    min_arousal = stats["min_arousal"] if stats["min_arousal"] is not None else -1.0
    max_arousal = stats["max_arousal"] if stats["max_arousal"] is not None else 1.0
    min_dominance = (
        stats["min_dominance"] if stats["min_dominance"] is not None else -1.0
    )
    max_dominance = (
        stats["max_dominance"] if stats["max_dominance"] is not None else 1.0
    )

    # Normalize liked book features (all 8 Plutchik emotions + VAD)
    liked_anger_norm = normalize(get_val(liked_row["avg_anger"]), min_anger, max_anger)
    liked_anticipation_norm = normalize(
        get_val(liked_row["avg_anticipation"]), min_anticipation, max_anticipation
    )
    liked_disgust_norm = normalize(
        get_val(liked_row["avg_disgust"]), min_disgust, max_disgust
    )
    liked_fear_norm = normalize(get_val(liked_row["avg_fear"]), min_fear, max_fear)
    liked_joy_norm = normalize(get_val(liked_row["avg_joy"]), min_joy, max_joy)
    liked_sadness_norm = normalize(
        get_val(liked_row["avg_sadness"]), min_sadness, max_sadness
    )
    liked_surprise_norm = normalize(
        get_val(liked_row["avg_surprise"]), min_surprise, max_surprise
    )
    liked_trust_norm = normalize(get_val(liked_row["avg_trust"]), min_trust, max_trust)
    liked_valence_norm = normalize(
        get_val(liked_row["avg_valence"]), min_valence, max_valence
    )
    liked_arousal_norm = normalize(
        get_val(liked_row["avg_arousal"]), min_arousal, max_arousal
    )
    liked_dominance_norm = normalize(
        get_val(liked_row["avg_dominance"]), min_dominance, max_dominance
    )

    # Normalize all features in the DataFrame and compute similarity
    # Normalize each feature column
    def norm_col(col_name, min_val, max_val):
        if min_val == max_val:
            return lit(0.5)
        return (col(col_name) - min_val) / (max_val - min_val)

    recommendations = (
        other_books.withColumn(
            "anger_norm", norm_col("avg_anger", min_anger, max_anger)
        )
        .withColumn(
            "anticipation_norm",
            norm_col("avg_anticipation", min_anticipation, max_anticipation),
        )
        .withColumn("disgust_norm", norm_col("avg_disgust", min_disgust, max_disgust))
        .withColumn("fear_norm", norm_col("avg_fear", min_fear, max_fear))
        .withColumn("joy_norm", norm_col("avg_joy", min_joy, max_joy))
        .withColumn("sadness_norm", norm_col("avg_sadness", min_sadness, max_sadness))
        .withColumn(
            "surprise_norm", norm_col("avg_surprise", min_surprise, max_surprise)
        )
        .withColumn("trust_norm", norm_col("avg_trust", min_trust, max_trust))
        .withColumn("valence_norm", norm_col("avg_valence", min_valence, max_valence))
        .withColumn("arousal_norm", norm_col("avg_arousal", min_arousal, max_arousal))
        .withColumn(
            "dominance_norm", norm_col("avg_dominance", min_dominance, max_dominance)
        )
        .withColumn(
            "similarity",
            1.0
            / (
                1.0
                + sqrt(
                    pow((col("anger_norm") - liked_anger_norm), 2)
                    + pow((col("anticipation_norm") - liked_anticipation_norm), 2)
                    + pow((col("disgust_norm") - liked_disgust_norm), 2)
                    + pow((col("fear_norm") - liked_fear_norm), 2)
                    + pow((col("joy_norm") - liked_joy_norm), 2)
                    + pow((col("sadness_norm") - liked_sadness_norm), 2)
                    + pow((col("surprise_norm") - liked_surprise_norm), 2)
                    + pow((col("trust_norm") - liked_trust_norm), 2)
                    + pow((col("valence_norm") - liked_valence_norm), 2)
                    + pow((col("arousal_norm") - liked_arousal_norm), 2)
                    + pow((col("dominance_norm") - liked_dominance_norm), 2)
                )
            ),
        )
        .orderBy(col("similarity").desc())
        .limit(top_n)
    )

    return recommendations.select(
        "book_id",
        "title",
        "author",
        "similarity",
        # All 8 Plutchik emotions
        "avg_anger",
        "avg_anticipation",
        "avg_disgust",
        "avg_fear",
        "avg_joy",
        "avg_sadness",
        "avg_surprise",
        "avg_trust",
        # VAD scores
        "avg_valence",
        "avg_arousal",
    )
