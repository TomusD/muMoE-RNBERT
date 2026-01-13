'''
Visualize diffrent plots for the gating coefficients of the experts.
'''

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import re

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12
})

# Coefficient Analysis
def analyze_experts_coeffs(inter_path, output_dir):
    """
    Performs an analysis of the gating coefficients of experts
    Args:
        inter_path (str): Path to the interpretation data inter_path file.
        output_dir (str): Directory to save output plots.
    """
    # Initialize variables to collect across chunks
    all_coeffs = []
    max_coeffs = []
    total_rows = 0
    filename = os.path.basename(inter_path).replace('.csv', '')
    num_experts = filename.split('_')[3]
    tfmethod = filename.split('_')[0]
    dedouble_method = filename.split('_')[6]
    layer = filename.split('_')[5]
    
    # Process in chunks
    for chunk_idx, chunk in enumerate(pd.read_csv(inter_path, chunksize=50000)):
        print(f"Processing chunk {chunk_idx+1}, rows: {len(chunk)}")
        total_rows += len(chunk)
            
        # Isolate the Expert Coefficient Columns
        coeff_cols = [col for col in chunk.columns if 'coeff_expert_' in col]
        if not coeff_cols:
            raise ValueError("No 'coeff_expert_' columns found in the DataFrame.")
            
        # Extract coefficients from this chunk
        expert_data = chunk[coeff_cols]

        # Add to overall collection
        sample_rate = 0.25  # Sample 25% of coefficients
        chunk_coeffs = expert_data.values.flatten()
        if expert_data.shape[1] > 1000:
            print(f"Sampling {int(100*sample_rate)}% of the data from chunk {chunk_idx+1}")
            all_coeffs.extend(np.random.choice(chunk_coeffs, size=int(len(chunk_coeffs)*sample_rate)))
        else:
            print(f"Using all the data from chunk {chunk_idx+1}")
            all_coeffs.extend(chunk_coeffs)
        
        # Track max coefficient per note
        max_coeffs.extend(expert_data.max(axis=1).tolist())
    
    print(f"Processed {total_rows} total rows with {len(coeff_cols)} experts")

    # Plot the overall coefficient distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(all_coeffs, bins=50, kde=False)
    plt.title(f'Overall Distribution of {num_experts} experts with {tfmethod} {dedouble_method} method from layer {layer}')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Frequency (Log Scale)')
    plt.yscale('log')
    plt.grid(True, which='both', ls="--")
    if expert_data.shape[1] > 1000: 
        plt.savefig(output_dir + f'/Overall_Distribution_of_{num_experts}_experts_with_{tfmethod}_{dedouble_method}_method_from_{int(sample_rate*100)}%_of_data_layer_{layer}.pdf')
        print(f"\nSaved plot as 'Overall_Distribution_of_{num_experts}_experts_with_{tfmethod}_{dedouble_method}_method_from_{int(sample_rate*100)}%_of_data_layer_{layer}.pdf' to {output_dir}")
    else:
        plt.savefig(output_dir + f'/Overall_Distribution_of_{num_experts}_experts_with_{tfmethod}_{dedouble_method}_method_layer_{layer}.pdf')
        print(f"\nSaved plot as 'Overall_Distribution_of_{num_experts}_experts_with_{tfmethod}_{dedouble_method}_method_layer_{layer}.pdf' to {output_dir}")
    plt.close()

    # Plot the distribution of maximum scores
    plt.figure(figsize=(12, 6))
    sns.histplot(max_coeffs, bins=50, kde=True)
    plt.title(f'Distribution of Top Experts Per Note for {num_experts} experts with {tfmethod} {dedouble_method} method from layer {layer}')
    plt.xlabel('Max Coefficient Value per Note')
    plt.ylabel('Frequency')
    plt.grid(True, which='both', ls="--")
    plt.savefig(output_dir + f'/Distribution_of_top_experts_from_{num_experts}_with_{tfmethod}_{dedouble_method}_method_layer_{layer}.pdf')
    print(f"\nSaved plot as 'Distribution_of_top_experts_from_{num_experts}_with_{tfmethod}_{dedouble_method}_method_layer_{layer}.pdf' to {output_dir}")
    plt.close()

def analyze_expert_load_distribution(inter_path, output_dir, threshold):
    """
    Analyze and plot the expert load distribution.

    Args:
        inter_path (str): Path to the interpretation data csv file.
        output_dir (str): Directory to save output plots.
        threshold (float): Threshold for filtering expert coefficients.
    """

    # Initialize variables
    expert_note_counts = []
    expert_thresh = []
    total_rows = 0
    filename = os.path.basename(inter_path).replace('.csv', '')
    num_experts = filename.split('_')[3]
    tfmethod = filename.split('_')[0]
    dedouble_method = filename.split('_')[6]
    layer = filename.split('_')[5]

    # Read the interpretation data
    chunk = pd.read_csv(inter_path)
    total_rows += len(chunk)
    print(f"Processing total rows: {total_rows}")
        
    # Isolate the Expert Coefficient Columns
    coeff_cols = [col for col in chunk.columns if 'coeff_expert_' in col]
    if not coeff_cols:
        raise ValueError("No 'coeff_expert_' columns found in the DataFrame.")

    # Calculate thresholded load (number of notes where expert has coeff ≥ threshold)
    for col in coeff_cols:
        thresh_count = (chunk[col] >= threshold).sum()
        note_count = (chunk[col] > 0).sum()
        expert_thresh.append(thresh_count)
        expert_note_counts.append(note_count)

    print(f"Processed {total_rows} total rows with {len(coeff_cols)} experts")

    # Plot Expert load distribution
    # Define 10 distinct colors that will repeat
    colors = plt.cm.tab10(np.arange(10))

    plt.figure(figsize=(16, 8))
    plt.bar(np.arange(len(expert_note_counts)), expert_note_counts, width=3.0, color=[colors[i % len(colors)] for i in range(len(expert_note_counts))])
    plt.title(f'Expert Load Distribution for {num_experts} Experts with {tfmethod} {dedouble_method} method')
    plt.xlabel('Expert Index')
    plt.ylabel('Number of Notes Assigned in each Expert')
    plt.grid(True, axis='y', ls="--")
    plt.savefig(output_dir + f'/Expert_Load_Distribution_{tfmethod}_{dedouble_method}_{num_experts}_experts_layer_{layer}.pdf')
    print(f"\nSaved plot as 'Expert_Load_Distribution_{tfmethod}_{dedouble_method}_{num_experts}_experts_layer_{layer}.pdf' to {output_dir}")
    plt.close()

    if threshold:
        plt.figure(figsize=(16, 8))
        plt.bar(np.arange(len(expert_note_counts)), expert_thresh, width=3.0, color=[colors[i % len(colors)] for i in range(len(expert_note_counts))])
        plt.title(f'Expert Threshold Load Distribution for {num_experts} Experts with {tfmethod} {dedouble_method} method')
        plt.xlabel('Expert Index')
        plt.ylabel(f'Number of Notes Assigned in Expert with Coeff ≥ {threshold}')
        plt.grid(True, axis='y', ls="--")
        plt.savefig(output_dir + f'/Expert_Load_Distribution_{tfmethod}_{dedouble_method}_{num_experts}_experts_{threshold}_threshold_layer_{layer}.pdf')
        print(f"\nSaved plot as 'Expert_Load_Distribution_{tfmethod}_{dedouble_method}_{num_experts}_experts_{threshold}_threshold_layer_{layer}.pdf' to {output_dir}")
        plt.close()

# Composer Analysis
def analyze_composer_experts(inter_path, output_dir, composer_name, n_experts, threshold):
    """
    Performs an analysis to find and plot the top most
    active experts for a given composer from the test set.

    Args:
        inter_path (str): Path to the interpretation data csv file.
        output_dir (str): Directory to save output plots.
        composer_name (str): The name of the composer to analyze (e.g., 'beethoven').
        n_experts (int): Number of top experts to display.
        threshold (float): Threshold for filtering expert coefficients.
    """
    filename = os.path.basename(inter_path).replace('.csv', '')
    num_experts = filename.split('_')[3]
    tfmethod = filename.split('_')[0]
    dedouble_method = filename.split('_')[6]
    layer = filename.split('_')[5]

    composer_data = []
    for chunk in pd.read_csv(inter_path, chunksize=50000):
        composer_chunk = chunk[chunk['csv_path'].str.contains(composer_name, case=False, na=False)]
        if not composer_chunk.empty:
            composer_data.append(composer_chunk)

    if not composer_data:
        raise ValueError(f"No scores found for composer '{composer_name}'.")
    if threshold is None or threshold <= 0:
        raise ValueError("Threshold must be provided and be greater than 0 for composer analysis.")
    
    composer_scores = pd.concat(composer_data)
    unique_scores = len(composer_scores[['score_id', 'csv_path']].drop_duplicates())
    print(f"\nFound {len(composer_scores)} notes in {unique_scores} scores for {composer_name}")

    # Isolate and Average Expert Coefficients
    coeff_cols = [col for col in composer_scores.columns if 'coeff_expert_' in col]
    if not coeff_cols:
        raise ValueError("No 'coeff_expert_' columns found in the DataFrame.")
            
    expert_data = composer_scores[coeff_cols]
    average_coeffs = expert_data.mean()

    expert_thresh = []
    for col in coeff_cols:
        thresh_count = (composer_scores[col] >= threshold).sum()
        expert_thresh.append(thresh_count)

    expert_thresh_series = pd.Series(expert_thresh, index=coeff_cols)

    # Find the Top Experts
    top_experts = average_coeffs.sort_values(ascending=False).head(n_experts)
    print(f"Result: Top {n_experts} experts identified for {len(average_coeffs)} experts:")
    print(top_experts)

    # Plot the Results
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12
    })
    plt.figure(figsize=(14, 8))
    barplot = sns.barplot(x=top_experts.index, y=top_experts.values, hue=top_experts.index, legend=False)
    plt.title(f'Top {n_experts} Most Active Experts for {composer_name.capitalize()} from {num_experts} experts with {tfmethod} {dedouble_method} method from layer {layer}')
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

    plt.savefig(output_dir + f'/{composer_name}_top_{n_experts}_experts_from_{num_experts}_experts_with_{tfmethod}_{dedouble_method}_method_layer_{layer}', dpi=300)
    print(f"\nSaved the plot as '{composer_name}_top_{n_experts}_experts_from_{num_experts}_experts_with_{tfmethod}_{dedouble_method}_method_layer_{layer}.png' to {output_dir}")
    plt.close()

    # Find the top experts by counting thresholded active experts
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
    barplot = sns.barplot(x=top_experts.index, y=top_experts.values, hue=top_experts.index, legend=False)
    plt.title(f'Top {n_experts} Most Active Experts for {composer_name.capitalize()} from {num_experts} experts with {tfmethod} {dedouble_method} method from layer {layer}')
    plt.xlabel('Expert ID')
    plt.ylabel(f'Number of Notes Assigned in Expert with Coeff ≥ {threshold}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add values on top of bars
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.0f'), 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'center', 
                            xytext = (0, 9), 
                            textcoords = 'offset points')

    plt.savefig(output_dir + f'/{composer_name}_top_{n_experts}_experts_based_on_notes_assigned_from_{num_experts}_experts_with_{tfmethod}_method_and_{threshold}_{dedouble_method}_threshold_layer_{layer}.png', dpi=300)
    print(f"\nSaved the plot as '{composer_name}_top_{n_experts}_experts_based_on_notes_assigned_from_{num_experts}_experts_with_{tfmethod}_method_and_{threshold}_{dedouble_method}_threshold_layer_{layer}.png' to {output_dir}")
    plt.close()

# Score Analysis
def analyze_score_experts(inter_path, output_dir, score_name, n_experts, threshold):
    """
    Performs an analysis to find and plot the top most
    active experts for a given score from the test set.

    Args:
        inter_path (str): Path to the interpretation data csv file.
        output_dir (str): Directory to save output plots.
        score_name (str): The name of the score to analyze
        n_experts (int): Number of top experts to display.
        threshold (float): Threshold for filtering expert coefficients.
    """
    filename = os.path.basename(inter_path).replace('.csv', '')
    num_experts = filename.split('_')[3]
    tfmethod = filename.split('_')[0]
    dedouble_method = filename.split('_')[6]
    layer = filename.split('_')[5]

    score_data = []
    for chunk in pd.read_csv(inter_path, chunksize=50000):
        score_chunk = chunk[chunk['score_id'].str.contains(re.escape(score_name), case=False, na=False)]
        if not score_chunk.empty:
            score_data.append(score_chunk)

    if not score_data:
        raise ValueError(f"No '{score_name}' score found.")
    if threshold is None or threshold <= 0:
        raise ValueError("Threshold must be provided and be greater than 0 for score analysis.")
    
    score = pd.concat(score_data)
    print(f"\nFound {len(score)} notes for {score_name} score")

    # Isolate and Average Expert Coefficients
    coeff_cols = [col for col in score.columns if 'coeff_expert_' in col]
    if not coeff_cols:
        raise ValueError("No 'coeff_expert_' columns found in the DataFrame.")
            
    expert_data = score[coeff_cols]
    average_coeffs = expert_data.mean()

    expert_thresh = []
    for col in coeff_cols:
        thresh_count = (score[col] >= threshold).sum()
        expert_thresh.append(thresh_count)

    expert_thresh_series = pd.Series(expert_thresh, index=coeff_cols)

    # Find the Top Experts
    top_experts = average_coeffs.sort_values(ascending=False).head(n_experts)
    print(f"Result: Top {n_experts} experts identified for {len(average_coeffs)} experts:")
    print(top_experts)

    # Plot the Results
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12
    })
    plt.figure(figsize=(14, 8))
    barplot = sns.barplot(x=top_experts.index, y=top_experts.values, hue=top_experts.index, legend=False)
    plt.title(f'Top {n_experts} Most Active Experts for {score_name.capitalize()} score from {num_experts} experts with {tfmethod} {dedouble_method} method from layer {layer}')
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

    plt.savefig(output_dir + f'/{score_name}_top_{n_experts}_experts_from_{num_experts}_experts_with_{tfmethod}_{dedouble_method}_method_layer_{layer}', dpi=300)
    print(f"\nSaved the plot as '{score_name}_top_{n_experts}_experts_from_{num_experts}_experts_with_{tfmethod}_{dedouble_method}_method_layer_{layer}.png' to {output_dir}")
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
    barplot = sns.barplot(x=top_experts.index, y=top_experts.values, hue=top_experts.index, legend=False)
    plt.title(f'Top {n_experts} Most Active Experts for {score_name.capitalize()} score from {num_experts} experts with {tfmethod} {dedouble_method} method from layer {layer}')
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

    plt.savefig(output_dir + f'/{score_name}_top_{n_experts}_experts_from_{num_experts}_experts_with_{tfmethod}_{dedouble_method}_method_and_{threshold}_threshold_layer_{layer}.png', dpi=300)
    print(f"\nSaved the plot as '{score_name}_top_{n_experts}_experts_from_{num_experts}_experts_with_{tfmethod}_{dedouble_method}_method_and_{threshold}_threshold_layer_{layer}.png' to {output_dir}")
    plt.close()

# Get all the key changes in the original score
def key_changes(orig_score):
    if 'key' in orig_score.columns:
        keys = orig_score[orig_score['type'] != 'bar'][['onset', 'release', 'key', 'granular_key']]
        key_changes = []
        last_key = None
        last_granular_key = None
        for _, key in keys.iterrows():
            mk = key['key']
            gk = key['granular_key']
            ont = key['onset']
            rel = key['release']
            # Check if this is a key change
            if mk != last_key or gk != last_granular_key:
                key_changes.append({
                    'key': mk,
                    'granular_key': gk,
                    'onset': ont,
                    'release': rel
                })
                last_key = mk
                last_granular_key = gk
        key_changes.pop(0)
        key_changes_df = pd.DataFrame(key_changes)
        print(f"Key changes DataFrame:{key_changes_df.shape}")
    else:
        print("There is no key column in the original score.")

    return key_changes_df

# Convert MIDI pitch to note name
def midi_to_note(midi_pitch):
    note_names = ['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B']
    
    # Calculate octave and note
    octave = midi_pitch // 12 - 1 
    note_index = midi_pitch % 12
    
    # Return note name with octave
    return f"{note_names[note_index]}{octave}"
    
# Piano roll analysis
def analyze_expert_notes(inter_path, output_dir, score_name, expert_index, threshold, n_notes):
    """
    Analyzes the number of most active notes per expert and plot the results in a piano roll format.

    Args:
        inter_path (str): Path to the interpretation data csv file.
        output_dir (str): Directory to save output plots.
        score_name (str): The name of the score to analyze.
        expert_index (int): Index of the expert to analyze.
        threshold (float): Threshold for filtering expert coefficients.
        n_notes (int): Number of top notes to analyze per expert.
    """
    filename = os.path.basename(inter_path).replace('.csv', '')
    num_experts = filename.split('_')[3]
    tfmethod = filename.split('_')[0]
    dedouble_method = filename.split('_')[6]
    layer = filename.split('_')[5]

    score_data = []
    for chunk in pd.read_csv(inter_path, chunksize=50000):
        score_chunk = chunk[chunk['score_id'].str.contains(re.escape(score_name), case=False, na=False)]
        if not score_chunk.empty:
            score_data.append(score_chunk)

    # Check if any data was found for the specified score and if the expert index is valid
    if not score_data:
        raise ValueError(f"No '{score_name}' score found.")
    if expert_index > (int(num_experts)-1) or expert_index < 0:
        raise ValueError(f"Invalid expert index {expert_index}. Must be between 0 and {int(num_experts) - 1}.")
    
    # Concatenate all chunks into a single DataFrame and filter notes based on the expert's coefficients and threshold and short them
    score = pd.concat(score_data)
    notes = score[score[f'coeff_expert_{expert_index}'] >= threshold][['pitch', 'duration', f'coeff_expert_{expert_index}']]
    top_notes = notes.sort_values(by=f'coeff_expert_{expert_index}', ascending=False).head(n_notes)
    coeff_df = score[['note_index', f'coeff_expert_{expert_index}']]


    # Original score path, notes and mapping with expert coefficients
    orig_csv_path = score['csv_path'].values[0]
    orig_score = pd.read_csv(orig_csv_path)
    orig_notes = orig_score[orig_score['type'] == 'note']
    coeff_orig_notes = pd.merge(orig_notes, coeff_df, left_on='Unnamed: 0', right_on='note_index', how='outer')

    if coeff_orig_notes.empty:
        raise ValueError("No notes found in the CSV file.")

    # DEBUG: Check the notes from the original and coefficients csvs
    orig_indices = set(orig_notes['Unnamed: 0'])
    coeff_indices = set(coeff_df['note_index'])
    duplicate_indices = orig_indices.intersection(coeff_indices)
    og = list(orig_indices -  duplicate_indices)
    og.sort()
    cg = list(coeff_indices - duplicate_indices)
    cg.sort()
    print(f"Original indices that don't match: {len(og)}", og)
    print(f"Coefficient indices that don't match: {len(cg)}", cg)
    print(f"Original notes: {len(orig_notes)}")
    print(f"Notes with coefficients: {len(coeff_df)}")

    # Find the key changes from the original score
    if 'key' in orig_score.columns:
        key_changes_df = key_changes(orig_score)
    else:
        raise ValueError("The original score does not contain a 'key' column for key changes.")
    
    key_changes_df = key_changes_df[key_changes_df['key'] != 'na']
    print(f"\nNumber of notes for Expert {expert_index} with threshold ≥ {threshold}: {len(notes)}")

    # Calculate y axis limits for plot
    min_pitch = max(0, int(coeff_orig_notes['pitch'].min()) - 2)
    max_pitch = min(127, int(coeff_orig_notes['pitch'].max()) + 2)
    
    # Calculate x axis
    max_time = coeff_orig_notes['release'].max()

    # Create a dynamic figure size based on the max_time
    if max_time < 100:
        print(f"Using large Figure size") 
        xaxis = 50
        yaxis = 20
        fontsize = 15
        x_ticks_max = max_time + 5
        x_ticks_step = 5
    if max_time >= 100:
        print(f"Using small Figure size")
        xaxis = 85
        yaxis = 35
        fontsize = 12
        x_ticks_max = max_time + 10
        x_ticks_step = 10

    fig, ax1 = plt.subplots(1, 1, figsize=(xaxis, yaxis), facecolor='white')
    ax1.set_facecolor('white')
    
    # Get min/max coefficient values for coloring
    min_coef = top_notes[f'coeff_expert_{expert_index}'].min()
    max_coef = top_notes[f'coeff_expert_{expert_index}'].max()
    norm = Normalize(vmin=min_coef, vmax=max_coef)
    cmap = cm.inferno
    
    # Plot each note for original score
    for _, note in coeff_orig_notes.iterrows():
        pitch = note['pitch']
        start = note['onset']
        end = note['release']
        coef_value = note[f'coeff_expert_{expert_index}']
        color = cmap(norm(coef_value))
        rect = plt.Rectangle((start, pitch), end - start, 0.8, color=color, alpha=0.8)
        ax1.add_patch(rect)
    
    # Set y-axis limits and ticks for plot
    ax1.set_ylim(min_pitch, max_pitch)
    y_ticks = np.arange(min_pitch, max_pitch)
    y_tick_labels = [midi_to_note(pitch) for pitch in y_ticks]
    
    # Apply to plot
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_tick_labels)
    
    # Set x-axis limits and ticks for plot
    ax1.set_xlim(0, max_time)
    x_ticks = np.arange(0, x_ticks_max, x_ticks_step)
    ax1.set_xticks(x_ticks)

    # Add key changes annotation in the plot if available
    for i, (_, key_change) in enumerate(key_changes_df.iterrows()):
        # Key change text
        key_text = f"{key_change['key']}/{key_change['granular_key']}"         

        start = key_change['onset']
        ax1.axvspan(start, start + 0.3, ymin=0.0, ymax=1, alpha=0.3, color='lightgreen')

        y_offset = 3 - (i % 3)
        
        # Add text labels
        ax1.text(start, min_pitch + y_offset, key_text, 
                ha='center', va='top', bbox=dict(facecolor='white', 
                                                alpha=0.7, boxstyle='round'),
                                                fontsize=fontsize, rotation=45)

    ax1.grid(True, color='lightgray', linestyle='-', linewidth=0.5)

    ax1.set_xlabel('Time (beats)', fontsize=41)
    ax1.set_ylabel('MIDI Pitch', fontsize=41)
    ax1.set_title(f'Piano Roll of score {score_name} for expert {expert_index} Activity with Coeff Threshold ≥ {threshold} from layer {layer} ', weight='bold', fontsize=42)
    plt.tight_layout()
    plt.savefig(output_dir + f'/Expert_{expert_index}_Notes_Pianoroll_Analysis_for_{score_name}_{tfmethod}_{dedouble_method}_{threshold}_threshold_layer_{layer}.pdf')
    print(f"Saved the plot as 'Expert_{expert_index}_Notes_Pianoroll_Analysis_for_{score_name}_{tfmethod}_{dedouble_method}_{threshold}_threshold_layer_{layer}.pdf' to {output_dir}")
    plt.close()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--inter-path", required=True, help="Path to the csv file containing ALL expert coefficients.")
    parser.add_argument("--output-dir", default='.', help="Directory to save output plots (default: current directory).")
    parser.add_argument("--threshold", type=float, help="Threshold for expert coefficients.")
    parser.add_argument("--experts-analysis", action='store_true', help="Perform analysis for a specific expert.")
    parser.add_argument("--experts-load-analysis", action='store_true', help="Perform expert load distribution analysis.")
    parser.add_argument("--composer", help="Name of the composer to analyze (default: 'beethoven').")
    parser.add_argument("--composer-analysis", action='store_true', help="Perform analysis for a specific composer.")
    parser.add_argument("--n-experts", type=int, default=10, help="Number of top experts to display (default: 10).")
    parser.add_argument("--score", help="Name of the score to analyze.")
    parser.add_argument("--score-analysis", action='store_true', help="Perform analysis for a specific score.")
    parser.add_argument("--notes-analysis", action='store_true', help="Plot the number of notes per expert.")
    parser.add_argument("--expert", type=int, default=0, help="Expert index to analyze (default: 0).")
    parser.add_argument("--n-notes", type=int, default=1000, help="Number of top notes to analyze per expert (default: 1000).")
    args = parser.parse_args()

    if args.experts_analysis:
        print(f"Analyzing experts coefficients")
        analyze_experts_coeffs(args.inter_path, args.output_dir)

    if args.experts_load_analysis:
        print(f"Analyzing expert load distribution with threshold ≥ {args.threshold}")
        analyze_expert_load_distribution(args.inter_path, args.output_dir, args.threshold)

    if args.composer_analysis:
        print(f"Analyzing experts for composer: {args.composer}")
        analyze_composer_experts(args.inter_path, args.output_dir, args.composer.lower(), args.n_experts, args.threshold)

    if args.score_analysis:
        print(f"Analyzing experts for score: {args.score}")
        analyze_score_experts(args.inter_path, args.output_dir, args.score, args.n_experts, args.threshold)

    if args.notes_analysis:
        print(f"Plotting the notes that the expert {args.expert} was most active on, with threshold ≥ {args.threshold}")  
        analyze_expert_notes(args.inter_path, args.output_dir, args.score, args.expert, args.threshold, args.n_notes)
    
