"""
Comprehensive evaluation script for EmoArc pipeline.

Run after main.py to generate metrics for reporting:
- Recommendation quality across diverse test books
- Similarity score distributions
- Emotion/VAD distribution statistics
- Performance benchmarks
"""

import sys
import os
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, stddev, min as spark_min, max as spark_max
from recommender import recommend, WEIGHT_PRESETS


# Comprehensive test set covering diverse genres
TEST_BOOKS = [
    # Children's Literature / Fantasy
    ("11", "Alice in Wonderland", ["children", "fantasy"]),
    ("12", "Through the Looking-Glass", ["children", "fantasy"]),
    ("16", "Peter Pan", ["children", "fantasy"]),
    # Gothic Horror / Science Fiction
    ("84", "Frankenstein", ["gothic", "horror", "science fiction"]),
    ("43", "Dr. Jekyll and Mr. Hyde", ["gothic", "horror"]),
    ("35", "The Time Machine", ["science fiction"]),
    ("36", "War of the Worlds", ["science fiction"]),
    # Adventure
    ("74", "Tom Sawyer", ["adventure", "children"]),
    ("76", "Huckleberry Finn", ["adventure", "satire"]),
    ("1661", "Sherlock Holmes", ["mystery", "detective"]),
    # Classics / Drama
    ("1342", "Pride and Prejudice", ["romance", "classics"]),
    ("98", "A Tale of Two Cities", ["historical", "drama"]),
    ("46", "A Christmas Carol", ["fantasy", "drama"]),
    # Poetry / Non-fiction
    ("1", "Declaration of Independence", ["non-fiction", "historical"]),
    ("100", "Complete Works of Shakespeare", ["drama", "poetry"]),
]


def create_spark_session():
    """Create Spark session for evaluation."""
    return (
        SparkSession.builder.appName("EmoArc-Evaluation")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def evaluate_recommendations(spark, trajectories, metadata_df, test_books):
    """
    Evaluate recommendation quality.

    Returns dict with per-book results and aggregate statistics.
    """
    results = []
    all_similarities = []
    eval_times = []

    # Cache trajectories for faster repeated access
    trajectories.cache()
    trajectories.count()  # Force cache

    # Get available book IDs
    available_ids = set(
        row["book_id"] for row in trajectories.select("book_id").collect()
    )

    print(
        f"\n  Testing {len(test_books)} books ({len(available_ids)} available in corpus)..."
    )

    for book_id, book_name, expected_genres in test_books:
        if book_id not in available_ids:
            print(f"    ⊘ {book_name} (ID: {book_id}) - not in corpus")
            continue

        start_time = time.time()

        # Get recommendations
        recs = recommend(
            spark,
            trajectories,
            book_id,
            top_n=10,
            metadata_df=metadata_df,
            preset="balanced",
        )

        if recs is None:
            print(f"    ✗ {book_name} - no recommendations")
            continue

        recs_list = recs.collect()
        eval_time = time.time() - start_time
        eval_times.append(eval_time)

        if len(recs_list) == 0:
            print(f"    ✗ {book_name} - empty recommendations")
            continue

        # Collect metrics
        similarities = [r["similarity"] for r in recs_list]
        all_similarities.extend(similarities)

        top_rec = recs_list[0]
        avg_sim = sum(similarities[:5]) / min(5, len(similarities))

        print(
            f"    ✓ {book_name[:30]:<30} → {top_rec['title'][:25]:<25} (sim: {top_rec['similarity']:.3f}, time: {eval_time:.2f}s)"
        )

        results.append(
            {
                "book_id": book_id,
                "book_name": book_name,
                "expected_genres": expected_genres,
                "top_recommendation": top_rec["title"],
                "top_similarity": top_rec["similarity"],
                "avg_top5_similarity": avg_sim,
                "eval_time_seconds": eval_time,
                "top_5": [
                    {"title": r["title"], "similarity": r["similarity"]}
                    for r in recs_list[:5]
                ],
            }
        )

    # Aggregate statistics
    stats = {}
    if all_similarities:
        sorted_sims = sorted(all_similarities)
        n = len(sorted_sims)
        stats = {
            "count": n,
            "mean": sum(sorted_sims) / n,
            "median": sorted_sims[n // 2],
            "std": (sum((x - sum(sorted_sims) / n) ** 2 for x in sorted_sims) / n)
            ** 0.5,
            "min": sorted_sims[0],
            "max": sorted_sims[-1],
            "p25": sorted_sims[n // 4],
            "p75": sorted_sims[3 * n // 4],
        }

    perf_stats = {}
    if eval_times:
        perf_stats = {
            "mean_time": sum(eval_times) / len(eval_times),
            "min_time": min(eval_times),
            "max_time": max(eval_times),
            "total_time": sum(eval_times),
        }

    return {
        "per_book_results": results,
        "similarity_stats": stats,
        "performance_stats": perf_stats,
        "books_tested": len(results),
        "books_skipped": len(test_books) - len(results),
    }


def evaluate_preset_comparison(spark, trajectories, metadata_df, book_id="11"):
    """Compare recommendations across different weight presets."""
    print(f"\n  Comparing presets for book ID {book_id}...")

    preset_results = {}

    for preset_name in WEIGHT_PRESETS.keys():
        recs = recommend(
            spark,
            trajectories,
            book_id,
            top_n=5,
            metadata_df=metadata_df,
            preset=preset_name,
        )

        if recs:
            recs_list = recs.collect()
            preset_results[preset_name] = {
                "weights": WEIGHT_PRESETS[preset_name],
                "recommendations": [
                    {"title": r["title"], "similarity": r["similarity"]}
                    for r in recs_list
                ],
            }
            print(
                f"    {preset_name}: {recs_list[0]['title'][:40]} ({recs_list[0]['similarity']:.3f})"
            )

    return preset_results


def evaluate_emotion_distribution(trajectories):
    """Analyze emotion score distributions across the corpus."""
    emotions = [
        "anger",
        "anticipation",
        "disgust",
        "fear",
        "joy",
        "sadness",
        "surprise",
        "trust",
    ]

    emotion_stats = {}
    for emotion in emotions:
        stats = trajectories.agg(
            avg(f"ratio_{emotion}").alias("mean"),
            stddev(f"ratio_{emotion}").alias("std"),
            spark_min(f"ratio_{emotion}").alias("min"),
            spark_max(f"ratio_{emotion}").alias("max"),
        ).collect()[0]

        emotion_stats[emotion] = {
            "mean": stats["mean"],
            "std": stats["std"] or 0,
            "min": stats["min"],
            "max": stats["max"],
        }

    return emotion_stats


def evaluate_vad_distribution(trajectories):
    """Analyze VAD score distributions."""
    vad_stats = {}

    for dim in ["valence", "arousal", "dominance"]:
        stats = trajectories.agg(
            avg(f"avg_{dim}").alias("mean"),
            stddev(f"avg_{dim}").alias("std"),
            spark_min(f"avg_{dim}").alias("min"),
            spark_max(f"avg_{dim}").alias("max"),
        ).collect()[0]

        vad_stats[dim] = {
            "mean": stats["mean"],
            "std": stats["std"] or 0,
            "min": stats["min"],
            "max": stats["max"],
        }

    return vad_stats


def print_summary(eval_data):
    """Print evaluation summary to console."""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    # Recommendation quality
    rec_quality = eval_data["recommendation_quality"]
    sim_stats = rec_quality["similarity_stats"]
    perf_stats = rec_quality["performance_stats"]

    print(f"\n📊 Recommendation Quality ({rec_quality['books_tested']} books tested)")
    print(
        f"   Similarity: mean={sim_stats['mean']:.4f}, median={sim_stats['median']:.4f}, std={sim_stats['std']:.4f}"
    )
    print(f"   Range: [{sim_stats['min']:.4f}, {sim_stats['max']:.4f}]")
    print(
        f"   Performance: {perf_stats['mean_time']:.2f}s avg, {perf_stats['total_time']:.2f}s total"
    )

    # Emotion distribution
    print("\n📈 Emotion Distribution (Plutchik's 8)")
    emotion_stats = eval_data["emotion_distribution"]
    sorted_emotions = sorted(
        emotion_stats.items(), key=lambda x: x[1]["mean"], reverse=True
    )
    for emotion, stats in sorted_emotions:
        print(f"   {emotion:<15} {stats['mean']:.4f} ± {stats['std']:.4f}")

    # VAD
    print("\n🎭 VAD Distribution")
    for dim, stats in eval_data["vad_distribution"].items():
        print(f"   {dim:<12} {stats['mean']:+.4f} ± {stats['std']:.4f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="EmoArc Evaluation Script")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode - fewer test books"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("EmoArc - Evaluation Suite")
    print("=" * 70)

    # Check if trajectories exist
    trajectories_path = os.path.join(args.output, "trajectories")
    if not os.path.exists(trajectories_path):
        print(f"\n❌ Error: Trajectories not found at {trajectories_path}")
        print("   Run main.py first to generate trajectories.")
        sys.exit(1)

    spark = create_spark_session()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        # Load data
        print("\n[1/4] Loading data...")
        trajectories = spark.read.parquet(trajectories_path)
        metadata_df = spark.read.csv(
            "data/gutenberg_metadata.csv", header=True, inferSchema=True
        )

        book_count = trajectories.count()
        print(f"  ✓ Loaded {book_count} book trajectories")

        # Select test books
        test_books = TEST_BOOKS[:6] if args.quick else TEST_BOOKS

        # Run evaluations
        print("\n[2/4] Evaluating recommendations...")
        rec_quality = evaluate_recommendations(
            spark, trajectories, metadata_df, test_books
        )

        print("\n[3/4] Comparing presets...")
        preset_comparison = evaluate_preset_comparison(spark, trajectories, metadata_df)

        print("\n[4/4] Analyzing distributions...")
        emotion_dist = evaluate_emotion_distribution(trajectories)
        vad_dist = evaluate_vad_distribution(trajectories)

        # Compile results
        eval_data = {
            "timestamp": datetime.now().isoformat(),
            "corpus_size": book_count,
            "recommendation_quality": rec_quality,
            "preset_comparison": preset_comparison,
            "emotion_distribution": emotion_dist,
            "vad_distribution": vad_dist,
        }

        # Save results
        eval_file = os.path.join(args.output, "evaluation_metrics.json")
        with open(eval_file, "w") as f:
            json.dump(eval_data, f, indent=2)
        print(f"\n✓ Results saved to {eval_file}")

        # Print summary
        print_summary(eval_data)

        print("\n" + "=" * 70)
        print("Evaluation complete!")
        print("=" * 70)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
