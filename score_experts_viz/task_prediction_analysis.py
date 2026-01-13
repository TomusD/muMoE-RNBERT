import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

def key_name(index):
    note_names = ['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B']
    return f"{note_names[index]}"


def analyze_predictions(inter_path, output_dir, expert_index, threshold, degrees=None, chord_factors=None, chord_tone=None, harmony_onset=None, 
                          bass_pcs=None, inversion=None, quality=None, key=None, RN=None):
    """
    Analyzes the number of most active roman numerals in combination and separately for a given expert.

    Args:
        inter_path (str): Path to the interpretation data csv file.
        output_dir (str): Directory to save output plots.
        expert_index (int): Index of the expert to analyze.
        threshold (float): Threshold for filtering expert coefficients.
        degrees (bool): If True, analyze the top degrees for the expert.
        chord_factors (bool): If True, analyze the top chord factors for the expert.
        chord_tone (bool): If True, analyze the top chord tones for the expert.
        harmony_onset (bool): If True, analyze the top harmony onset labels for the expert.
        bass_pcs (bool): If True, analyze the top bass pitch classes for the expert.
        inversion (bool): If True, analyze the top inversions for the expert.
        quality (bool): If True, analyze the top qualities for the expert.
        key (bool): If True, analyze the top keys for the expert.
        RN (bool): If True, analyze the top roman numerals for the expert.
    """
    filename = os.path.basename(inter_path).replace('.csv', '')
    num_experts = filename.split('_')[3]
    tfmethod = filename.split('_')[0]
    dedouble_method = filename.split('_')[6]
    layer = filename.split('_')[5]
    color='#1f77b4'

    # Check if the expert index is valid
    if expert_index > (int(num_experts)-1) or expert_index < 0:
        raise ValueError(f"Invalid expert index {expert_index}. Must be between 0 and {int(num_experts) - 1}.")

    pred_data = []
    for chunk in pd.read_csv(inter_path, chunksize=50000):
        score_chunk = chunk[chunk[f'coeff_expert_{expert_index}'] >= threshold]
        if not score_chunk.empty:
            pred_data.append(score_chunk)
    
    # Concatenate all chunks into a single DataFrame
    score = pd.concat(pred_data)
    coeff_df = score[['note_index', 'score_id', 'csv_path', f'coeff_expert_{expert_index}', 'chord_factors_pred_label', 'chord_tone_pred_label', 'harmony_onset_pred_label',
                      'bass_pcs_pred_label', 'primary_alteration_primary_degree_secondary_alteration_secondary_degree_pred_label',
                      'inversion_pred_label', 'quality_pred_label', 'key_pc_pred_label', 'mode_pred_label']]

    # Degree Analysis
    if degrees:
        top_results = coeff_df.groupby(['primary_alteration_primary_degree_secondary_alteration_secondary_degree_pred_label']).size().sort_values(ascending=False).head(10)
        print(f"Result: Top degrees for expert {expert_index}:")
        print(top_results)

        # Plot the Results
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 12
        })
        plt.figure(figsize=(14, 8))
        barplot = sns.barplot(x=range(len(top_results)), y=top_results.values, color=color, legend=False)
        plt.title(f'Top degrees for expert {expert_index} with Coeff Threshold ≥ {threshold}')
        plt.xticks(ticks=range(len(top_results)), labels=top_results.index, rotation=45, ha='right')
        plt.xlabel('Degrees')
        plt.ylabel(f'Number of Degrees above Threshold')
        plt.tight_layout()
        
        # Add values on top of bars
        for p in barplot.patches:
            barplot.annotate(format(p.get_height(), '.0f'), 
                                (p.get_x() + p.get_width() / 2., p.get_height()), 
                                ha = 'center', va = 'center', 
                                xytext = (0, 9), 
                                textcoords = 'offset points')

        plt.savefig(output_dir + f'/top_degrees_for_expert_{expert_index}_with_{tfmethod}_{dedouble_method}_method_and_{threshold}_threshold_layer_{layer}.pdf', dpi=300)
        print(f"\nSaved the plot as 'top_degrees_for_expert_{expert_index}_with_{tfmethod}_{dedouble_method}_method_and_{threshold}_threshold_layer_{layer}.pdf' to {output_dir}")
        plt.close()

    if chord_factors:
        top_results = coeff_df.groupby(['chord_factors_pred_label']).size().sort_values(ascending=False).head(10)
        print(f"Result: Top Chord Factors for expert {expert_index}:")
        print(top_results)

        # Plot the Results
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 12
        })
        plt.figure(figsize=(14, 8))
        barplot = sns.barplot(x=range(len(top_results)), y=top_results.values, color=color, legend=False)
        plt.title(f'Top Chord Factors for expert {expert_index} with Coeff Threshold ≥ {threshold}')
        plt.xticks(ticks=range(len(top_results)), labels=top_results.index, rotation=45, ha='right')
        plt.xlabel('Chord Factors')
        plt.ylabel(f'Number of Chord Factors above Threshold')
        plt.tight_layout()
        
        # Add values on top of bars
        for p in barplot.patches:
            barplot.annotate(format(p.get_height(), '.0f'), 
                                (p.get_x() + p.get_width() / 2., p.get_height()), 
                                ha = 'center', va = 'center', 
                                xytext = (0, 9), 
                                textcoords = 'offset points')

        plt.savefig(output_dir + f'/top_chord_factors_for_expert_{expert_index}_with_{tfmethod}_{dedouble_method}_method_and_{threshold}_threshold_layer_{layer}.pdf', dpi=300)
        print(f"\nSaved the plot as 'top_chord_factors_for_expert_{expert_index}_with_{tfmethod}_{dedouble_method}_method_and_{threshold}_threshold_layer_{layer}.pdf' to {output_dir}")
        plt.close()

    if chord_tone:
        top_results = coeff_df.groupby(['chord_tone_pred_label']).size().sort_values(ascending=False).head(10)
        print(f"Result: Top Chord Tones for expert {expert_index}:")
        print(top_results)

        # Plot the Results
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 12
        })
        plt.figure(figsize=(14, 8))
        barplot = sns.barplot(x=range(len(top_results)), y=top_results.values, color=color, legend=False)
        plt.title(f'Top Chord Tones for expert {expert_index} with Coeff Threshold ≥ {threshold}')
        plt.xticks(ticks=range(len(top_results)), labels=top_results.index, rotation=45, ha='right')
        plt.xlabel('Chord Tones')
        plt.ylabel(f'Number of Chord Tones above Threshold')
        plt.tight_layout()
        
        # Add values on top of bars
        for p in barplot.patches:
            barplot.annotate(format(p.get_height(), '.0f'), 
                                (p.get_x() + p.get_width() / 2., p.get_height()), 
                                ha = 'center', va = 'center', 
                                xytext = (0, 9), 
                                textcoords = 'offset points')

        plt.savefig(output_dir + f'/top_chord_tones_for_expert_{expert_index}_with_{tfmethod}_{dedouble_method}_method_and_{threshold}_threshold_layer_{layer}.pdf', dpi=300)
        print(f"\nSaved the plot as 'top_chord_tones_for_expert_{expert_index}_with_{tfmethod}_{dedouble_method}_method_and_{threshold}_threshold_layer_{layer}.pdf' to {output_dir}")
        plt.close()
    
    if harmony_onset:
        top_results = coeff_df.groupby(['harmony_onset_pred_label']).size().sort_values(ascending=False).head(10)
        print(f"Result: Top Harmony Onset for expert {expert_index}:")
        print(top_results)

        # Plot the Results
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 12
        })
        plt.figure(figsize=(14, 8))
        barplot = sns.barplot(x=range(len(top_results)), y=top_results.values, color=color, legend=False)
        plt.title(f'Top Harmony Onset for expert {expert_index} with Coeff Threshold ≥ {threshold}')
        plt.xticks(ticks=range(len(top_results)), labels=top_results.index, rotation=45, ha='right')
        plt.xlabel('Harmony Onsets')
        plt.ylabel(f'Number of Harmony Onsets above Threshold')
        plt.tight_layout()
        
        # Add values on top of bars
        for p in barplot.patches:
            barplot.annotate(format(p.get_height(), '.0f'), 
                                (p.get_x() + p.get_width() / 2., p.get_height()), 
                                ha = 'center', va = 'center', 
                                xytext = (0, 9), 
                                textcoords = 'offset points')

        plt.savefig(output_dir + f'/top_harmony_onset_for_expert_{expert_index}_with_{tfmethod}_{dedouble_method}_method_and_{threshold}_threshold_layer_{layer}.pdf', dpi=300)
        print(f"\nSaved the plot as 'top_harmony_onset_for_expert_{expert_index}_with_{tfmethod}_{dedouble_method}_method_and_{threshold}_threshold_layer_{layer}.pdf' to {output_dir}")
        plt.close()

    if bass_pcs:
        top_results = coeff_df.groupby(['bass_pcs_pred_label']).size().sort_values(ascending=False).head(10)
        print(f"Result: Top Bass Pitch Classes for expert {expert_index}:")
        print(top_results)

        # Plot the Results
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 12
        })
        plt.figure(figsize=(14, 8))
        barplot = sns.barplot(x=range(len(top_results)), y=top_results.values, color=color, legend=False)
        plt.title(f'Top Bass Pitch Classes for expert {expert_index} with Coeff Threshold ≥ {threshold}')
        plt.xticks(ticks=range(len(top_results)), labels=top_results.index, rotation=45, ha='right')
        plt.xlabel('Bass Pitch Classes')
        plt.ylabel(f'Number of Bass Pitch Classes above Threshold')
        plt.tight_layout()
        
        # Add values on top of bars
        for p in barplot.patches:
            barplot.annotate(format(p.get_height(), '.0f'), 
                                (p.get_x() + p.get_width() / 2., p.get_height()), 
                                ha = 'center', va = 'center', 
                                xytext = (0, 9), 
                                textcoords = 'offset points')

        plt.savefig(output_dir + f'/top_bass_pitch_classes_for_expert_{expert_index}_with_{tfmethod}_{dedouble_method}_method_and_{threshold}_threshold_layer_{layer}.pdf', dpi=300)
        print(f"\nSaved the plot as 'top_bass_pitch_classes_for_expert_{expert_index}_with_{tfmethod}_{dedouble_method}_method_and_{threshold}_threshold_layer_{layer}.pdf' to {output_dir}")
        plt.close()

    if inversion:
        top_results = coeff_df.groupby(['inversion_pred_label']).size().sort_values(ascending=False).head(10)
        print(f"Result: Top Inversions for expert {expert_index}:")
        print(top_results)

        # Plot the Results
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 12
        })
        plt.figure(figsize=(14, 8))
        barplot = sns.barplot(x=range(len(top_results)), y=top_results.values, color=color, legend=False)
        plt.title(f'Top Inversions for expert {expert_index} with Coeff Threshold ≥ {threshold}')
        plt.xticks(ticks=range(len(top_results)), labels=top_results.index, rotation=45, ha='right')
        plt.xlabel('Inversions')
        plt.ylabel(f'Number of Inversions above Threshold')
        plt.tight_layout()
        
        # Add values on top of bars
        for p in barplot.patches:
            barplot.annotate(format(p.get_height(), '.0f'), 
                                (p.get_x() + p.get_width() / 2., p.get_height()), 
                                ha = 'center', va = 'center', 
                                xytext = (0, 9), 
                                textcoords = 'offset points')

        plt.savefig(output_dir + f'/top_inversions_for_expert_{expert_index}_with_{tfmethod}_{dedouble_method}_method_and_{threshold}_threshold_layer_{layer}.pdf', dpi=300)
        print(f"\nSaved the plot as 'top_inversions_for_expert_{expert_index}_with_{tfmethod}_{dedouble_method}_method_and_{threshold}_threshold_layer_{layer}.pdf' to {output_dir}")
        plt.close()
    
    if quality:
        top_results = coeff_df.groupby(['quality_pred_label']).size().sort_values(ascending=False).head(10)
        print(f"Result: Top Qualities for expert {expert_index}:")
        print(top_results)

        # Plot the Results
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 12
        })
        plt.figure(figsize=(14, 8))
        barplot = sns.barplot(x=range(len(top_results)), y=top_results.values, color=color, legend=False)
        plt.title(f'Top Qualities for expert {expert_index} with Coeff Threshold ≥ {threshold}')
        plt.xticks(ticks=range(len(top_results)), labels=top_results.index, rotation=45, ha='right')
        plt.xlabel('Qualities')
        plt.ylabel(f'Number of Qualities above Threshold')
        plt.tight_layout()
        
        # Add values on top of bars
        for p in barplot.patches:
            barplot.annotate(format(p.get_height(), '.0f'), 
                                (p.get_x() + p.get_width() / 2., p.get_height()), 
                                ha = 'center', va = 'center', 
                                xytext = (0, 9), 
                                textcoords = 'offset points')

        plt.savefig(output_dir + f'/top_qualities_for_expert_{expert_index}_with_{tfmethod}_{dedouble_method}_method_and_{threshold}_threshold_layer_{layer}.pdf', dpi=300)
        print(f"\nSaved the plot as 'top_qualities_for_expert_{expert_index}_with_{tfmethod}_{dedouble_method}_method_and_{threshold}_threshold_layer_{layer}.pdf' to {output_dir}")
        plt.close()

    if key:
        top_results = coeff_df.groupby(['key_pc_pred_label','mode_pred_label']).size().sort_values(ascending=False).head(10)
        print(f"Result: Top Keys for expert {expert_index}:")
        print(top_results)

        formatted_labels = [f"{key_name(int(key))}/_{mode}" for key, mode in top_results.index]
        print(f"Formatted Labels: {formatted_labels}")

        # Plot the Results
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 12
        })
        plt.figure(figsize=(14, 8))
        barplot = sns.barplot(x=range(len(top_results)), y=top_results.values, color=color, legend=False)
        plt.title(f'Top Keys for expert {expert_index} with Coeff Threshold ≥ {threshold}')
        plt.xticks(ticks=range(len(top_results)), labels=formatted_labels, rotation=45, ha='right')
        plt.xlabel('Keys')
        plt.ylabel(f'Number of Keys above Threshold')
        plt.tight_layout()
        
        # Add values on top of bars
        for p in barplot.patches:
            barplot.annotate(format(p.get_height(), '.0f'), 
                                (p.get_x() + p.get_width() / 2., p.get_height()), 
                                ha = 'center', va = 'center', 
                                xytext = (0, 9), 
                                textcoords = 'offset points')

        plt.savefig(output_dir + f'/top_keys_for_expert_{expert_index}_with_{tfmethod}_{dedouble_method}_method_and_{threshold}_threshold_layer_{layer}.pdf', dpi=300)
        print(f"\nSaved the plot as 'top_keys_for_expert_{expert_index}_with_{tfmethod}_{dedouble_method}_method_and_{threshold}_threshold_layer_{layer}.pdf' to {output_dir}")
        plt.close()

    if RN:
        top_results = coeff_df.groupby(['key_pc_pred_label','mode_pred_label','primary_alteration_primary_degree_secondary_alteration_secondary_degree_pred_label',
                                        'inversion_pred_label', 'quality_pred_label']).size().sort_values(ascending=False).head(20)
        print(f"Result: Top Roman Numerals for expert {expert_index}:")
        print(top_results)

        formatted_labels = [f"{key_name(int(key))}_{mode}/{degree}/{inversion}/{quality}" for key, mode, degree, inversion, quality in top_results.index]
        print(f"Formatted Labels: {formatted_labels}")

        # Plot the Results
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 12
        })
        plt.figure(figsize=(14, 8))
        barplot = sns.barplot(x=range(len(top_results)), y=top_results.values, color=color, legend=False)
        plt.title(f'Top Roman Numerals for expert {expert_index} with Coeff Threshold ≥ {threshold}')
        plt.xticks(ticks=range(len(top_results)), labels=formatted_labels, rotation=45, ha='right')
        plt.xlabel('Roman Numerals')
        plt.ylabel(f'Number of Roman Numerals above Threshold')
        plt.tight_layout()
        
        # Add values on top of bars
        for p in barplot.patches:
            barplot.annotate(format(p.get_height(), '.0f'), 
                                (p.get_x() + p.get_width() / 2., p.get_height()), 
                                ha = 'center', va = 'center', 
                                xytext = (0, 9), 
                                textcoords = 'offset points')

        plt.savefig(output_dir + f'/top_RN_for_expert_{expert_index}_with_{tfmethod}_{dedouble_method}_method_and_{threshold}_threshold_layer_{layer}.pdf')
        print(f"\nSaved the plot as 'top_RN_for_expert_{expert_index}_with_{tfmethod}_{dedouble_method}_method_and_{threshold}_threshold_layer_{layer}.pdf' to {output_dir}")
        plt.close()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--inter-path", required=True, help="Path to the csv file containing ALL expert coefficients.")
    parser.add_argument("--output-dir", default='.', help="Directory to save output plots (default: current directory).")
    parser.add_argument("--threshold", type=float, help="Threshold for expert coefficients.")
    parser.add_argument("--expert", type=int, default=0, help="Expert index to analyze (default: 0).")
    parser.add_argument("--degrees", action='store_true', help="If set, analyze the top degrees for the expert.")
    parser.add_argument("--chord-factors", action='store_true', help="If set, analyze the top chord factors for the expert.")
    parser.add_argument("--chord-tone", action='store_true', help="If set, analyze the top chord tones for the expert.")
    parser.add_argument("--harmony-onset", action='store_true', help="If set, analyze the top harmony onset labels for the expert.")
    parser.add_argument("--bass-pcs", action='store_true', help="If set, analyze the top bass pitch classes for the expert.")
    parser.add_argument("--inversion", action='store_true', help="If set, analyze the top inversions for the expert.")
    parser.add_argument("--quality", action='store_true', help="If set, analyze the top qualities for the expert.")
    parser.add_argument("--key", action='store_true', help="If set, analyze the top keys for the expert.")
    parser.add_argument("--RN", action='store_true', help="If set, analyze the top roman numerals for the expert.")
    parser.add_argument("--all-analysis", action='store_true', help="If set, perform all analyses for the expert.")
    args = parser.parse_args()

    if args.degrees:
        print(f"Plotting the degrees that the expert {args.expert} was most active on, with threshold ≥ {args.threshold}")
        analyze_predictions(args.inter_path, args.output_dir, args.expert, args.threshold, degrees=args.degrees)
    if args.chord_factors:
        print(f"Plotting the chord factors that the expert {args.expert} was most active on, with threshold ≥ {args.threshold}")
        analyze_predictions(args.inter_path, args.output_dir, args.expert, args.threshold, chord_factors=args.chord_factors)
    if args.chord_tone:
        print(f"Plotting the chord tones that the expert {args.expert} was most active on, with threshold ≥ {args.threshold}")
        analyze_predictions(args.inter_path, args.output_dir, args.expert, args.threshold, chord_tone=args.chord_tone)
    if args.harmony_onset:
        print(f"Plotting the harmony onset labels that the expert {args.expert} was most active on, with threshold ≥ {args.threshold}")
        analyze_predictions(args.inter_path, args.output_dir, args.expert, args.threshold, harmony_onset=args.harmony_onset)
    if args.bass_pcs:
        print(f"Plotting the bass pitch classes that the expert {args.expert} was most active on, with threshold ≥ {args.threshold}")
        analyze_predictions(args.inter_path, args.output_dir, args.expert, args.threshold, bass_pcs=args.bass_pcs)
    if args.inversion:
        print(f"Plotting the inversions that the expert {args.expert} was most active on, with threshold ≥ {args.threshold}")
        analyze_predictions(args.inter_path, args.output_dir, args.expert, args.threshold, inversion=args.inversion)
    if args.quality:
        print(f"Plotting the qualities that the expert {args.expert} was most active on, with threshold ≥ {args.threshold}")
        analyze_predictions(args.inter_path, args.output_dir, args.expert, args.threshold, quality=args.quality)
    if args.key:
        print(f"Plotting the keys that the expert {args.expert} was most active on, with threshold ≥ {args.threshold}")
        analyze_predictions(args.inter_path, args.output_dir, args.expert, args.threshold, key=args.key)
    if args.RN:
        print(f"Plotting the roman numerals that the expert {args.expert} was most active on, with threshold ≥ {args.threshold}")
        analyze_predictions(args.inter_path, args.output_dir, args.expert, args.threshold, RN=args.RN)
    if args.all_analysis:
        print(f"Plotting all analysis types for the expert {args.expert} with threshold ≥ {args.threshold}")
        analyze_predictions(args.inter_path, args.output_dir, args.expert, args.threshold, degrees=True, chord_factors=True, chord_tone=True, 
                          harmony_onset=True, bass_pcs=True, inversion=True, quality=True, key=True, RN=True)