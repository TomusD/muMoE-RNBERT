'''
This script deduplicates a CSV file containing expert coefficients.
It can also apply crossfading to notes that appear in multiple segments.
'''

import pandas as pd
import os
from functools import reduce
import argparse
from tqdm import tqdm

def drop_duplicates(inter_path, output_dir):

    filename = os.path.basename(inter_path).replace('.csv', '')
    output_path = f'{output_dir}/{filename}_drop.csv'

    # Identify columns to check for duplicates
    duplicate_check_columns = [
        'score_id', 
        'csv_path',
        'pitch',
        'duration', 
        'note_index',
    ]

    # Use reduce to process the file in chunks
    print("Processing CSV in chunks...")
    deduplicated_df = reduce(
        lambda df1, df2: pd.concat([df1, df2]).drop_duplicates(subset=duplicate_check_columns, keep='first'),
        pd.read_csv(inter_path, chunksize=50000)
    )

    original_row_count = sum(1 for _ in open(inter_path)) - 1
    print(f"Original CSV has approximately {original_row_count} rows")
    print(f"After deduplication: {len(deduplicated_df)} rows")
    print(f"Removed approximately {original_row_count - len(deduplicated_df)} rows")
    deduplicated_df.to_csv(output_path, index=False)
    print(f"\nDeduplicated CSV saved to: {output_path}")

def mean_coeff(inter_path, output_dir):

    filename = os.path.basename(inter_path).replace('.csv', '')
    output_path = f'{output_dir}/{filename}_mean.csv'
    score_data = {}

    # Collect all data by score
    for chunk in pd.read_csv(inter_path, chunksize=50000):
        print(f"Processing chunk with {len(chunk)} rows")
        for (score_id, csv_path), group_df in tqdm(chunk.groupby(['score_id', 'csv_path'])):
            if (score_id, csv_path) not in score_data:
                score_data[(score_id, csv_path)] = group_df
            else:
                score_data[(score_id, csv_path)] = pd.concat([score_data[(score_id, csv_path)], group_df])

    print(f"Found {len(score_data)} unique scores in interpretation data\n")
    all_results = []
    
    # Process each score separately
    for (score_id, csv_path), score_interp_df in score_data.items():
        print(f"Processing score: {score_id}")
        
        deduplicated_notes = []
        for _, note_group in tqdm(score_interp_df.groupby('note_index')):

            first_entry = note_group.iloc[0].copy()
            
            # Calculate mean for each expert column if there are duplicates
            if len(note_group) > 1:
                expert_columns = [col for col in note_group.columns if col.startswith('coeff_expert_')]
                for col in expert_columns:
                    mean_value = note_group[col].mean(skipna=True)
                    first_entry[col] = mean_value
            
            # Add the processed entry to results
            deduplicated_notes.append(first_entry)

        # Collect results for all scores
        all_results.extend(deduplicated_notes)
    
    # Combine all scores and save
    scores = pd.DataFrame(all_results)
    print(f"After deduplication: {len(scores)} rows")
    scores.to_csv(output_path, index=False)
    print(f"\nDeduplicated CSV saved to: {output_path}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--inter-path", required=True, help="Path to the csv file containing all expert coefficients.")
    parser.add_argument("--output-dir", required=True, default='.', help="Directory to save output csv (default: current directory).")
    parser.add_argument("--drop", action='store_true', help="Drop duplicate rows based on 'note_index'.")
    parser.add_argument("--mean", action='store_true', help="Calculate mean coefficients for each note.")
    args = parser.parse_args()

    if args.drop:
        print("Dropping duplicates based on note index...")
        drop_duplicates(args.inter_path, args.output_dir)

    if args.mean:
        print("Calculating mean coefficients...")
        mean_coeff(args.inter_path, args.output_dir)
