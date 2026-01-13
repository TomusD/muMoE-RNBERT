import pandas as pd
import numpy as np
from scipy.stats import entropy
import argparse
from tqdm import tqdm

def midi_to_note(midi_pitch):
    note_names = ['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B']
    
    # Calculate octave and note 
    note_index = midi_pitch % 12
    
    # Return note name with octave
    return f"{note_names[note_index]}"

parser = argparse.ArgumentParser()
parser.add_argument('--inter-path', type=str, required=True, help='Path to the input CSV file containing interpretation data')
parser.add_argument('--activation-threshold', type=float, default=0.4, help='Activation threshold for expert coefficients')
parser.add_argument('--RN', action='store_true', default=False, help='Use Roman Numeral representation instead of Key')
parser.add_argument('--ST', action='store_true', default=False, help='Use Sustaining Tones representation instead of Key')
args = parser.parse_args()

# Load Data
df = pd.read_csv(args.inter_path)
df_clean = df.dropna(subset=['key_pc_pred_label', 'mode_pred_label']).copy()
df_clean = df.dropna(subset=['primary_alteration_primary_degree_secondary_alteration_secondary_degree_pred_label']).copy()
df_clean = df.dropna(subset=['inversion_pred_label']).copy()
df_clean = df.dropna(subset=['quality_pred_label']).copy()

# Create Key_Label column
df_clean['key_pc_pred_label'] = [midi_to_note(pitch) for pitch in df_clean['key_pc_pred_label'].astype(int)]
df_clean['Key_Label'] = df_clean['key_pc_pred_label'].astype(str) + "_" + df_clean['mode_pred_label'].astype(str)

# Create Roman_Numeral column
df_clean['Roman_Numeral'] = df_clean['key_pc_pred_label'].astype(str) + "/" + \
                            df_clean['primary_alteration_primary_degree_secondary_alteration_secondary_degree_pred_label'].astype(str) + "/" + \
                            df_clean['inversion_pred_label'].astype(str) + "/" + \
                            df_clean['quality_pred_label'].astype(str)

# Create Sustaining_Tones column
df_clean['Sustaining_Tones'] = df_clean['harmony_onset_pred_label'].astype(str) + "/" + \
                               df_clean['chord_tone_pred_label'].astype(str)


print(f"Data Loaded: {len(df)} rows total.")
print(f"Data After Cleaning NaN Labels: {len(df_clean)} rows.")
print(f"Unique Keys found: {df_clean['Key_Label'].nunique()}")
print(f"Unique Roman Numerals found: {df_clean['Roman_Numeral'].nunique()}")


# Identify Expert Columns
expert_cols = [col for col in df_clean.columns if col.startswith('coeff_expert_')]
print(f"Experts found: {len(expert_cols)}")

# Calculate Baseline random expert entropy
if args.RN:
    print("Calculating baselines using Roman_Numeral distribution.")
    dataset_dist = df_clean['Roman_Numeral'].value_counts(normalize=True)
if args.ST:
    print("Calculating baselines using Sustaining_Tones distribution.")
    dataset_dist = df_clean['Sustaining_Tones'].value_counts(normalize=True)
if not args.RN and not args.ST:
    print("Calculating baselines using Key_Label distribution.")
    dataset_dist = df_clean['Key_Label'].value_counts(normalize=True)
theoretical_max_entropy = np.log2(len(dataset_dist))
random_baseline_entropy = entropy(dataset_dist, base=2)

print(f"\n--- Baselines ---")
print(f"Theoretical Max Entropy (Uniform Distribution): {theoretical_max_entropy:.4f}")
print(f"Random Baseline Entropy (Dataset Distribution): {random_baseline_entropy:.4f}")
print("-" * 60)

# Calculate Metrics for Each Expert
expert_metrics = []
for exp_col in tqdm(expert_cols):

    # Get rows where this expert is above the activation threshold
    active_data = df_clean[df_clean[exp_col] >= args.activation_threshold]
    
    # Skip dead experts
    if len(active_data) < 600:
        continue
    
    # Get the distribution of Keys or Roman Numerals for this specific expert
    if args.RN:
        expert_dist = active_data['Roman_Numeral'].value_counts(normalize=True)
    if args.ST:
        expert_dist = active_data['Sustaining_Tones'].value_counts(normalize=True)
    if not args.RN and not args.ST:
        expert_dist = active_data['Key_Label'].value_counts(normalize=True)
    #print(exp_col, expert_dist)
    
    # Calculate Entropy
    exp_entropy = entropy(expert_dist, base=2)
    
    # Assign Clarity Score
    clarity = expert_dist.max()
    top_class = expert_dist.idxmax()
    
    # Comparison to Baseline (Ratio)
    entropy_ratio = exp_entropy / random_baseline_entropy
    
    expert_metrics.append({
        'Expert': exp_col,
        'Specialization_Entropy': exp_entropy,
        'Clarity_Score': clarity,
        'Primary_Class': top_class,
        'Entropy_Ratio': entropy_ratio,
        'Activation_Count': len(active_data)
    })

# Create DataFrame
metrics_df = pd.DataFrame(expert_metrics)
print(f"Calculated metrics for {len(metrics_df)} active experts.")

# Print Results
metrics_df = metrics_df.sort_values(by='Clarity_Score', ascending=False)

print(f"\n--- Top 10 Most Specialized Experts ---")
print(metrics_df)

print(f"\n--- Aggregate Results ---")
print(f"Mean Expert Entropy:      {metrics_df['Specialization_Entropy'].mean():.4f} (Target < {random_baseline_entropy:.4f})")
print(f"Mean Clarity Score:       {metrics_df['Clarity_Score'].mean():.4f}")
print(f"Avg Entropy Ratio vs Rand: {metrics_df['Entropy_Ratio'].mean():.4f}% of random baseline")
print(f"Avg Improvement vs Rand:  {100 * (1 - metrics_df['Entropy_Ratio'].mean()):.1f}% reduction in uncertainty")