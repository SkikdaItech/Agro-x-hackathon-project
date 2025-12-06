import itertools
import numpy as np
import pandas as pd
from ml.predict import predict_pair_probability
from functools import lru_cache
import concurrent.futures
import time

df = pd.read_csv("genus_data.csv")


# Cache predictions to avoid redundant calculations
@lru_cache(maxsize=10000)
def get_cached_probability(genusA: str, genusB: str) -> float:
    """Cache probability calculations to avoid recomputing same pairs."""
    try:
        return predict_pair_probability(genusA, genusB)
    except:
        return 0.0


def generate_ranking(top_n=10, limit_genera=100):
    """
    Generate ranking with optimized performance.

    Args:
        top_n: Number of top pairs to return
        limit_genera: Limit number of genera to consider for performance
    """
    print(f"Starting ranking generation with top_n={top_n}, limit_genera={limit_genera}")

    # Limit number of genera for performance
    all_genera = df['Genus'].unique()
    if len(all_genera) > limit_genera:
        genera = list(all_genera[:limit_genera])
        print(f"Limited to first {limit_genera} genera out of {len(all_genera)} total")
    else:
        genera = list(all_genera)

    # Generate all unique pairs
    pairs = list(itertools.combinations(genera, 2))
    print(f"Processing {len(pairs)} genus pairs...")

    ranking_results = []

    # Use parallel processing for faster computation
    def process_pair(pair):
        genusA, genusB = pair
        try:
            prob = get_cached_probability(genusA, genusB)
            return {
                "genusA": genusA,
                "genusB": genusB,
                "probability": float(prob)
            }
        except Exception as e:
            print(f"Error processing pair {genusA}-{genusB}: {e}")
            return None

    # Process in batches to avoid memory issues
    batch_size = 100
    start_time = time.time()

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(pairs) // batch_size) + 1}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            batch_results = list(executor.map(process_pair, batch))

        # Add valid results
        for result in batch_results:
            if result is not None:
                ranking_results.append(result)

    # Sort by probability
    ranking_results.sort(key=lambda x: x["probability"], reverse=True)

    elapsed_time = time.time() - start_time
    print(f"Ranking generation completed in {elapsed_time:.2f} seconds")

    return ranking_results[:top_n]