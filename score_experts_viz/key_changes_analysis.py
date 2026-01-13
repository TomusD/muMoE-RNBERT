'''
Visualize key changes in the scores and expert activity.
'''
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import seaborn as sns

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12
})

# Get all the key changes from each original score
def key_changes(orig_score):
    if 'key' in orig_score.columns:
        keys = orig_score[orig_score['type'] != 'bar'][['onset', 'release', 'key', 'granular_key']]
        key_changes = []
        last_key = None
        last_granular_key = None
        for _, key in keys.iterrows():
            mk = key['key']
            gk = key['granular_key']
            # Check if this is a key change
            if mk != last_key or gk != last_granular_key:
                key_changes.append({
                    'key': mk,
                    'granular_key': gk,
                })
                last_key = mk
                last_granular_key = gk
        key_changes.pop(0)
        key_changes_df = pd.DataFrame(key_changes)
    else:
        print("There is no key column in the original score.")

    return key_changes_df

def analyze_key_changes(inter_path, output_dir, n_experts, threshold):
    """
    Analyze the key changes in the test dataset and which is the most active expert.

    Args:
        inter_path (str): Path to the interpretation data csv file.
        output_dir (str): Directory to save output plots.
        n_experts (int): Number of top experts to display.
        threshold (float): Threshold for filtering expert coefficients.
    """
    filename = os.path.basename(inter_path).replace('.csv', '')
    num_experts = filename.split('_')[3]
    tfmethod = filename.split('_')[0]
    dedouble_method = filename.split('_')[6]
    layer = filename.split('_')[5]

    scores_data = []
    for chunk in pd.read_csv(inter_path, chunksize=50000):
        print(f"Processing chunk with shape: {len(chunk)}")
        scores_data.append(chunk)

    scores = pd.concat(scores_data)
    csv_paths = scores['csv_path'].unique()

    # Get all key changes from the original scores
    all_key_changes = []
    for csv_path in csv_paths:
        orig_score = pd.read_csv(csv_path)
        if 'key' in orig_score.columns:
            key_changes_df = key_changes(orig_score)
            all_key_changes.append(key_changes_df)
        else:
            raise ValueError("The original score does not contain a 'key' column for key changes.")
        
    # Clean and concatenate all key changes into a single DataFrame
    key_changes_df = pd.concat(all_key_changes)
    key_changes_df = key_changes_df[key_changes_df['key'] != 'na']
    key_changes_counts = key_changes_df.groupby(key_changes_df.columns.tolist()).size().reset_index(name='counts')


    # Sort and prepare the top 15 key changes
    keys_ordered = key_changes_counts.sort_values(by='counts', ascending=False).head(15)
    keys_ordered["key/granular_key"] = keys_ordered["key"] + "/" + keys_ordered["granular_key"]
    keys_ordered.drop(columns=['key', 'granular_key'], inplace=True)
    print("Result: Top 15 key changes:")
    print(keys_ordered)

    # Plot the Results
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12
    })
    plt.figure(figsize=(14, 8))
    barplot = sns.barplot(x=keys_ordered['key/granular_key'], y=keys_ordered['counts'], legend=False)
    plt.title(f'Top Most used Key Changes in Test set')
    plt.xlabel('Key/Granular Key')
    plt.ylabel(f'Number of Key Changes Occured')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add values on top of bars
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.0f'), 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'center', 
                            xytext = (0, 9), 
                            textcoords = 'offset points')

    plt.savefig(output_dir + f'/Top_most_used_key_changes_test_set')
    print(f"\nSaved the plot as 'Top_most_used_key_changes_test_set.png' to {output_dir}")
    plt.close()

    # Average Expert Coefficients for all scores
    coeff_cols = [col for col in scores.columns if 'coeff_expert_' in col]
    if not coeff_cols:
        raise ValueError("No 'coeff_expert_' columns found in the DataFrame.")
    expert_data = scores[coeff_cols]
    average_coeffs = expert_data.mean()

    if threshold is None or threshold <= 0:
        raise ValueError("Threshold must be provided and be greater than 0 for composer analysis.")

    # Filter experts based on the threshold
    expert_thresh = []
    for col in coeff_cols:
        thresh_count = (scores[col] >= threshold).sum()
        expert_thresh.append(thresh_count)
    expert_thresh_series = pd.Series(expert_thresh, index=coeff_cols)

    # Find the Top Experts
    top_experts = average_coeffs.sort_values(ascending=False).head(n_experts)
    print(f"Result: Top {n_experts} experts identified for {len(average_coeffs)} experts:")
    print(top_experts)

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12
    })
    plt.figure(figsize=(14, 8))
    barplot = sns.barplot(x=top_experts.index, y=top_experts.values, legend=False)
    plt.title(f'Top {n_experts} Most Active Experts in test dataset from {num_experts} experts with {tfmethod} {dedouble_method} method from layer {layer}')
    plt.xlabel('Expert ID')
    plt.ylabel('Average Coefficient Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add values on top of bars
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.4f'), 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'center', 
                            xytext = (0, 9), 
                            textcoords = 'offset points')

    plt.savefig(output_dir + f'/top_{n_experts}_experts_from_{num_experts}_experts_with_{tfmethod}_{dedouble_method}_method_layer_{layer}', dpi=300)
    print(f"\nSaved the plot as 'top_{n_experts}_experts_from_{num_experts}_experts_with_{tfmethod}_{dedouble_method}_method_layer_{layer}.png' to {output_dir}")
    plt.close()

    top_experts = expert_thresh_series.sort_values(ascending=False).head(n_experts)
    print(f"Result: Top {n_experts} experts identified for {len(expert_thresh_series)} experts:")
    print(top_experts)

    # Plot the Results
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12
    })
    plt.figure(figsize=(14, 8))
    barplot = sns.barplot(x=top_experts.index, y=top_experts.values, legend=False)
    plt.title(f'Top {n_experts} Most Active Experts in test dataset from {num_experts} experts with {tfmethod} {dedouble_method} method from layer {layer}')
    plt.xlabel('Expert ID')
    plt.ylabel(f'Number of Notes Assigned per expert with Coeff ≥ {threshold}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add values on top of bars
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.0f'), 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'center', 
                            xytext = (0, 9), 
                            textcoords = 'offset points')

    plt.savefig(output_dir + f'/top_{n_experts}_experts_from_{num_experts}_experts_with_{tfmethod}_{dedouble_method}_method_and_{threshold}_threshold_layer_{layer}.png', dpi=300)
    print(f"\nSaved the plot as 'top_{n_experts}_experts_from_{num_experts}_experts_with_{tfmethod}_{dedouble_method}_method_and_{threshold}_threshold_layer_{layer}.png' to {output_dir}")
    plt.close()

def midi_to_note(midi_pitch):
    note_names = ['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B']
    
    # Calculate octave and note 
    note_index = midi_pitch % 12
    
    # Return note name with octave
    return f"{note_names[note_index]}"

def plot_expert_activity(inter_path, output_dir, expert_index, threshold):
    """
    Plot the top notes activated by a specific expert and the notes above a threshold.

    Args:
        inter_path (str): Path to the csv file containing expert coefficients.
        output_dir (str): Directory to save output plots.
        expert_index (int): Index of the expert to plot.
        threshold (float): Threshold for expert coefficients.
    """

    filename = os.path.basename(inter_path).replace('.csv', '')
    num_experts = filename.split('_')[3]
    tfmethod = filename.split('_')[0]
    dedouble_method = filename.split('_')[6]
    layer = filename.split('_')[5]

    # Read the CSV files in chunks
    scores_data = []
    for chunk in pd.read_csv(inter_path, chunksize=50000):
        print(f"Processing chunk with shape: {len(chunk)}")
        scores_data.append(chunk)

    if expert_index > (int(num_experts)-1) or expert_index < 0:
        raise ValueError(f"Invalid expert index {expert_index}. Must be between 0 and {int(num_experts) - 1}.")
    if threshold is None or threshold <= 0:
        raise ValueError("Threshold must be provided and be greater than 0 to plot pitch expert activity.")
    
    # Concatenate all chunks into a single DataFrame and group by note name abd expert index
    scores = pd.concat(scores_data)
    scores['note_name'] = [midi_to_note(pitch) for pitch in scores['pitch']]
    note_names = sorted(scores['note_name'].unique())
    print(f"Found {len(note_names)} unique note names")    
    expert_pitches = scores[['note_name',f'coeff_expert_{expert_index}']]

    # Find the Top Pitches based on mean of the coefficients
    meaned_pitches = expert_pitches.groupby('note_name').mean()
    top_pitches = meaned_pitches.sort_values(by=f'coeff_expert_{expert_index}', ascending=False).head(12).reset_index()
    xindex = top_pitches['note_name']
    yindex = top_pitches[f'coeff_expert_{expert_index}'].values
    print(f"Result: Top pitches identified for expert {expert_index}:")
    print(top_pitches)

    # Plot the Results
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12
    })
    plt.figure(figsize=(14, 8))
    barplot = sns.barplot(x=xindex, y=yindex, color='blue', legend=False, order=top_pitches['note_name'])
    plt.title(f'Top notes Activated for expert {expert_index} with mean coeff score from layer {layer}')
    plt.xticks(ticks=range(len(top_pitches)), labels=top_pitches['note_name'], rotation=45, ha='right')
    plt.xlabel('Notes')
    plt.ylabel('Average Coefficient Score')
    plt.tight_layout()
    
    # Add values on top of bars
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.4f'), 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'center', 
                            xytext = (0, 9), 
                            textcoords = 'offset points')

    plt.savefig(output_dir + f'/top_notes_for_expert_{expert_index}_with_{tfmethod}_{dedouble_method}_method_layer_{layer}', dpi=300)
    print(f"\nSaved the plot as 'top_notes_for_expert_{expert_index}_with_{tfmethod}_{dedouble_method}_method_layer_{layer}.png' to {output_dir}")
    plt.close()

    # Filter pitches above the threshold
    threshold_pitches = expert_pitches[expert_pitches[f'coeff_expert_{expert_index}'] >= threshold].groupby('note_name').size()
    top_threshold_pitches = threshold_pitches.sort_values(ascending=False).head(12)
    print(f"Result: Top notes for expert {expert_index}:")
    print(top_threshold_pitches)

    # Plot the Results
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12
    })
    plt.figure(figsize=(14, 8))
    barplot = sns.barplot(x=top_threshold_pitches.index, y=top_threshold_pitches.values, legend=False, order=top_threshold_pitches.index)
    plt.title(f'Top notes for expert {expert_index} with Coeff Threshold ≥ {threshold} from layer {layer}')
    plt.xticks(ticks=range(len(top_threshold_pitches)), labels=top_threshold_pitches.index, rotation=45, ha='right')
    plt.xlabel('Notes')
    plt.ylabel(f'Number of Notes Assigned')
    plt.tight_layout()
    
    # Add values on top of bars
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.0f'), 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'center', 
                            xytext = (0, 9), 
                            textcoords = 'offset points')

    plt.savefig(output_dir + f'/top_notes_for_expert_{expert_index}_with_{tfmethod}_{dedouble_method}_method_and_{threshold}_threshold_layer_{layer}.png', dpi=300)
    print(f"\nSaved the plot as 'top_notes_for_expert_{expert_index}_with_{tfmethod}_{dedouble_method}_method_and_{threshold}_threshold_layer_{layer}.png' to {output_dir}")
    plt.close()

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--inter-path", required=True, help="Path to the csv file containing ALL expert coefficients.")
    parser.add_argument("--output-dir", default='.', help="Directory to save output plots (default: current directory).")
    parser.add_argument("--threshold", type=float, help="Threshold for expert coefficients.")
    parser.add_argument("--n-experts", type=int, default=10, help="Number of top experts to display (default: 10).")
    parser.add_argument("--expert", type=int, help="Index of the expert to analyze (default: 0).")
    args = parser.parse_args()

    if args.expert is not None:
        plot_expert_activity(args.inter_path, args.output_dir, args.expert, args.threshold)
    else:
        analyze_key_changes(args.inter_path, args.output_dir, args.n_experts, args.threshold)