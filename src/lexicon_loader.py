"""
Load NRC Emotion and VAD lexicons into Spark DataFrames.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, trim


def load_emotion_lexicon(spark: SparkSession, lexicon_path: str):
    """
    Load NRC Emotion Lexicon (word-level) into Spark DataFrame.

    Format: word\temotion\tvalue (0 or 1)
    Emotions: anger, anticipation, disgust, fear, joy, sadness, surprise, trust (Plutchik's 8 basic emotions)
    Note: The lexicon also includes "negative" and "positive" (sentiment labels), but we use only the 8 emotions

    Returns:
        DataFrame with columns: word, emotion, value
    """
    # Read as text and split by tab
    df = spark.read.text(lexicon_path)
    df = df.select(
        split(col("value"), "\t")[0].alias("word"),
        split(col("value"), "\t")[1].alias("emotion"),
        split(col("value"), "\t")[2].alias("value"),
    )

    # Filter out invalid rows and convert value to int
    df = df.filter(
        (col("word").isNotNull())
        & (col("emotion").isNotNull())
        & (col("value").isNotNull())
    )
    df = df.withColumn("value", col("value").cast("int"))

    # Filter only emotions (exclude positive/negative if needed, or keep them)
    # Keep all emotions including positive/negative
    df = df.filter(col("value") == 1)  # Only words that have the emotion

    return df


def load_vad_lexicon(spark: SparkSession, lexicon_path: str):
    """
    Load NRC VAD Lexicon into Spark DataFrame.

    Format: term\tvalence\tarousal\tdominance

    Returns:
        DataFrame with columns: term, valence, arousal, dominance
    """
    # Read as text, skip header
    df = spark.read.text(lexicon_path)

    # Split by tab and extract columns
    df = df.select(
        split(col("value"), "\t")[0].alias("term"),
        split(col("value"), "\t")[1].alias("valence"),
        split(col("value"), "\t")[2].alias("arousal"),
        split(col("value"), "\t")[3].alias("dominance"),
    )

    # Filter out header row and invalid rows
    df = df.filter(
        (col("term") != "term")  # Skip header
        & (col("term").isNotNull())
        & (col("valence").isNotNull())
    )

    # Convert to proper types
    df = df.withColumn("valence", col("valence").cast("double"))
    df = df.withColumn("arousal", col("arousal").cast("double"))
    df = df.withColumn("dominance", col("dominance").cast("double"))

    # Trim whitespace from term
    df = df.withColumn("term", trim(col("term")))

    return df
