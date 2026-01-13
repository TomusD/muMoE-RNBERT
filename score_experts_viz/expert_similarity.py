'''
This script calculates the cosine similarity between 
expert coefficients for overlapping notes in an interpretation csv file.
'''
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--inter-path', type=str, required=True, help='Path to the input CSV file containing interpretation data')
args = parser.parse_args()

scores_data = []
for chunk in pd.read_csv(args.inter_path, chunksize=50000):
    print(f"Processing chunk with size: {len(chunk)}")
    scores_data.append(chunk)

scores = pd.concat(scores_data)

# Identify columns to check for duplicates
duplicate_check_columns = [
    'score_id', 
    'csv_path',
    'pitch',
    'duration', 
    'note_index',
]

# Check for duplicates based on the specified columns
duplicates = scores[scores.duplicated(subset=duplicate_check_columns, keep=False)]
score_groups = duplicates.groupby(['score_id', 'csv_path','pitch', 'duration','note_index'])
print(f"Calculating cosine similarity for {len(score_groups)} duplicate note groups...")

# Calculate cosine similarity for each group of duplicates
all_means = []
for (score_id, csv_path, pitch, duration, note_index), score_interp_df in tqdm(score_groups):
    coeff_cols = [col for col in score_interp_df.columns if 'coeff_expert_' in col]
    if not coeff_cols:
        print("No 'coeff_expert_' columns found in the DataFrame.")
    expert_data = score_interp_df[coeff_cols]
    mean_cos = cosine_similarity(expert_data)
    np.fill_diagonal(mean_cos, np.nan) 
    all_means.append(np.nanmean(mean_cos))

print(f"Overall mean similarity across all scores: {np.nanmean(all_means):.4f}")
print(f"Overall min similarity across all scores: {np.nanmin(all_means):.4f}")
print(f"Overall max similarity across all scores: {np.nanmax(all_means):.4f}")
print(f"Overall std similarity across all scores: {np.nanstd(all_means):.4f}")
print(f"Overall median similarity across all scores: {np.nanmedian(all_means):.4f}")
print(f"Overall 25th percentile similarity across all scores: {np.nanpercentile(all_means, 25):.4f}")
print(f"Overall 75th percentile similarity across all scores: {np.nanpercentile(all_means, 75):.4f}")
print(f"Overall 90th percentile similarity across all scores: {np.nanpercentile(all_means, 90):.4f}")
print(f"Overall 95th percentile similarity across all scores: {np.nanpercentile(all_means, 95):.4f}")
print(f"Overall 99th percentile similarity across all scores: {np.nanpercentile(all_means, 99):.4f}") 