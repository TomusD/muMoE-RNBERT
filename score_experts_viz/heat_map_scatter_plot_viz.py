"""
This script analyzes expert specialization based on pitch, grouped by note name,
and generates scatter plots and heatmaps to visualize the results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12
})

def midi_to_note(midi_pitch):
    # Convert MIDI pitch to note name
    note_names = ['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B']
    note_index = midi_pitch % 12
    return f"{note_names[note_index]}"

def analyze_expert_pitch_specialization(inter_path, output_dir, threshold):
    """
    Analyze expert specialization based on pitch, grouped by note name
    
    inter_path (str): Path to the input CSV file containing interpretation data
    output_dir (str): Directory to save output plots
    threshold (float): Threshold for filtering expert coefficients.

    """

    filename = os.path.basename(inter_path).replace('.csv', '')
    num_experts = filename.split('_')[3]
    tfmethod = filename.split('_')[0]
    dedouble_method = filename.split('_')[6]
    layer = filename.split('_')[5]
    
    score_data = pd.read_csv(inter_path)
    print(f"Loaded data with {len(score_data)} rows")
    coeff_cols = [col for col in score_data.columns if 'coeff_expert_' in col]

    # Convert pitch to note name
    score_data['note_name'] = [midi_to_note(pitch) for pitch in score_data['pitch']]
    note_names = sorted(score_data['note_name'].unique())
    print(f"Found {len(note_names)} unique note names")
    
    # For each note name, calculate average expert activation
    note_expert_data = {}
    for note in note_names:
        note_rows = score_data[score_data['note_name'] == note]
        expert_means = {col: note_rows[col].mean() for col in coeff_cols}
        note_expert_data[note] = expert_means
    
    # Find most active expert for each note
    note_experts = {}
    for note, expert_values in note_expert_data.items():
        max_expert = max(expert_values, key=expert_values.get)
        max_value = expert_values[max_expert]
        note_experts[note] = (max_expert, max_value)

    # Plot the mean expert activation results
    plt.figure(figsize=(14, 8))
    
    # Sort note names
    standard_order = ['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B']
    sorted_notes = sorted(note_experts.keys(), key=lambda x: standard_order.index(x))

    # Get expert indices and values
    expert_indices = [int(note_experts[n][0].replace("coeff_expert_", "")) for n in sorted_notes]
    expert_values = [note_experts[n][1] for n in sorted_notes]

    # Create scatter plot
    x_positions = list(range(len(sorted_notes)))
    plt.scatter(x_positions, expert_indices, s=[v*100 for v in expert_values], alpha=0.7)
    plt.title("Experts Specialization by Note with Mean value of Coefficients")
    plt.xlabel("Notes")
    plt.ylabel("Expert Index")
    plt.xticks(x_positions, sorted_notes)
    plt.ylim(0, int(num_experts) + 5)
    
    # Add annotations to show expert numbers
    for x, y in zip(x_positions, expert_indices):
        plt.annotate(f"Expert {y}", (x, y), xytext=(0, 7), 
                     textcoords='offset points', ha='center')
        
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir + f'/Scatter_plot_from_{num_experts}_experts_with_{tfmethod}_{dedouble_method}_method_layer_{layer}', dpi=300)
    print(f"\nSaved the plot as 'Scatter_plot_from_{num_experts}_experts_with_{tfmethod}_{dedouble_method}_method_layer_{layer}.png' to {output_dir}")
    plt.close()

    # Plot heatmap of expert activations
    plt.figure(figsize=(24, 20))
    
    # Create a matrix for heatmap
    valid_notes = sorted(note_expert_data.keys(), key=lambda x: standard_order.index(x))
    matrix = np.zeros((len(valid_notes), len(coeff_cols)))
    for i, note in enumerate(valid_notes):
        for j, expert in enumerate(coeff_cols):
            matrix[i, j] = note_expert_data[note][expert]
    
    # Plot heatmap
    plt.imshow(matrix, aspect='auto', cmap='inferno')
    plt.title(f'All Expert Activations by Note')
    plt.xlabel("Expert Index")
    plt.ylabel("Note")
    plt.xticks(range(len(coeff_cols)), [i for i in range(int(num_experts))])
    plt.yticks(range(len(valid_notes)), list(reversed(valid_notes)))
    plt.tight_layout()
    plt.savefig(output_dir + f'/Heat_Map_from_{num_experts}_experts_with_{tfmethod}_{dedouble_method}_method_layer_{layer}', dpi=300)
    print(f"\nSaved the plot as 'Heat_Map_from_{num_experts}_experts_with_{tfmethod}_{dedouble_method}_method_layer_{layer}.png' to {output_dir}")
    plt.close()

    # Plot results based on coefficient threshold
    if threshold > 0:

        note_expert_data = {}
        for note in note_names:
            note_rows = score_data[score_data['note_name'] == note]
            expert_counts = {col: (note_rows[col] > threshold).sum() for col in coeff_cols}
            note_expert_data[note] = expert_counts

        # Find most active expert for each note
        note_experts = {}
        for note, expert_values in note_expert_data.items():
            max_expert = max(expert_values, key=expert_values.get)
            max_value = expert_values[max_expert]
            note_experts[note] = (max_expert, max_value)

        # Plot the thresholded expert activation results
        plt.figure(figsize=(14, 8))
        
        # Sort note names
        sorted_notes = sorted(note_experts.keys(), key=lambda x: standard_order.index(x))

        # Get expert indices and values
        expert_indices = [int(note_experts[n][0].replace("coeff_expert_", "")) for n in sorted_notes]
        expert_values = [note_experts[n][1] for n in sorted_notes]

        # Create scatter plot
        x_positions = list(range(len(sorted_notes)))
        plt.scatter(x_positions, expert_indices, s=[v/100 for v in expert_values], alpha=0.7)
        plt.title(f"Experts Specialization by Note with coefficient Threshold ≥ {threshold}")
        plt.xlabel("Notes")
        plt.ylabel("Expert Index")
        plt.xticks(x_positions, sorted_notes)
        if int(num_experts) <= 50:
            plt.ylim(0, int(num_experts))
        else:
            plt.ylim(0, int(num_experts) + 20)

        # Add annotations to show expert numbers
        for x, y in zip(x_positions, expert_indices):
            plt.annotate(f"Expert {y}", (x, y), xytext=(0, 7), 
                        textcoords='offset points', ha='center')
            
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_dir + f'/Scatter_plot_from_{num_experts}_experts_with_{tfmethod}_{dedouble_method}_method_and_{threshold}_threshold_layer_{layer}.png', dpi=300)
        print(f"\nSaved the plot as 'Scatter_plot_from_{num_experts}_experts_with_{tfmethod}_{dedouble_method}_method_and_{threshold}_threshold_layer_{layer}.png' to {output_dir}")
        plt.close()

        # Plot heatmap of expert activations
        plt.figure(figsize=(24, 20))
        
        # Create a matrix for heatmap
        valid_notes = sorted(note_expert_data.keys(), key=lambda x: standard_order.index(x))
        matrix = np.zeros((len(valid_notes), len(coeff_cols)))
        for i, note in enumerate(valid_notes):
            for j, expert in enumerate(coeff_cols):
                matrix[i, j] = note_expert_data[note][expert]
        
        # Plot heatmap
        plt.imshow(matrix, aspect='auto', cmap='inferno')
        plt.title(f'All Expert Activations by Note with Coefficients > {threshold}')
        plt.xlabel("Expert Index")
        plt.ylabel("Note")
        plt.xticks(range(len(coeff_cols)), [i for i in range(int(num_experts))])
        plt.yticks(range(len(valid_notes)), list(reversed(valid_notes)))
        plt.tight_layout()
        plt.savefig(output_dir + f'/Heat_Map_from_{num_experts}_experts_with_{tfmethod}_{dedouble_method}_method_and_{threshold}_threshold_layer_{layer}.png', dpi=300)
        print(f"\nSaved the plot as 'Heat_Map_from_{num_experts}_experts_with_{tfmethod}_{dedouble_method}_method_and_{threshold}_threshold_layer_{layer}.png' to {output_dir}")
        plt.close()



if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--inter-path', type=str, required=True, help='Path to the input CSV file containing interpretation data')
    argparse.add_argument('--output-dir', type=str, required=True, help='Directory to save output plots')
    argparse.add_argument('--threshold', type=float, default=0.1, help='Threshold for expert activation')
    args = argparse.parse_args()

    analyze_expert_pitch_specialization(args.inter_path, args.output_dir, args.threshold)