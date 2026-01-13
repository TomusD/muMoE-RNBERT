import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_key(file_path):

    # Read only necessary columns
    cols = ['key', 'granular_key', 'tonicization']
    df = pd.read_csv(file_path, usecols=lambda c: c in cols)
    df = df[df['key'] != 'na']

    # Extract modulations and tonicizations
    modulations = df['key'].tolist()
    df['tonicization'] = pd.to_numeric(df['tonicization'], errors='coerce').fillna(0)
    tonic_rows = df[df['tonicization'] != 0]['granular_key'].dropna().tolist()

    return modulations, tonic_rows

def key_diversity():

    input_path = "/home/tomus/github_wsl/Thesis/RNBERT/rnbert/saved_datasets_chkpts/datasets/rnbert_csvs"
    output_dir = '/home/tomus/github_wsl/Thesis/RNBERT/rnbert/score_experts_viz'

    fifths_order = [
    'Dbb', 'bbb',
    'Abb', 'fb',
    'Ebb', 'cb',
    'Bbb', 'gb',
    'Fb',  'db',

    'Cb', 'ab',      
    'Gb', 'eb',     
    'Db', 'bb',     
    'Ab', 'f',      
    'Eb', 'c',    
    'Bb', 'g',   
    'F',  'd',   


    'C',  'a',


    'G',  'e',
    'D',  'b',       
    'A',  'f#',      
    'E',  'c#',      
    'B',  'g#',      
    'F#', 'd#',      
    'C#', 'a#',      

    'G#', 'e#',
    'D#', 'b#',      
    'A#', 'f##',     
    'E#',     
    'B#',    
    'F##',     
    ]

    # Search for CSV files recursively
    print(f"Searching for CSV files in {input_path}...")
    files = glob.glob(os.path.join(input_path, "**", "*.csv"), recursive=True)
    print(f"Found {len(files)} CSV files.")

    all_modulations = []
    all_tonicizations = []

    # Process each file to extract keys
    for file in files:
        mods, tonics = get_key(file)
        all_modulations.extend(mods)
        all_tonicizations.extend(tonics)

    # Create DataFrames for plotting
    df_mod = pd.DataFrame({'Key': all_modulations, 'Context': 'Modulation'})
    df_ton = pd.DataFrame({'Key': all_tonicizations, 'Context': 'Tonicization'})
    
    full_df = pd.concat([df_mod, df_ton], ignore_index=True)

    plt.figure(figsize=(15, 8))
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 12
    })
    
    # Log scale for y-axis as in the reference image
    plot = sns.countplot(data=full_df, x='Key', hue='Context', order=fifths_order)
    plot.set_yscale("log")
    
    plt.title("Key Diversity in Dataset (Modulation vs Tonicization)")
    plt.xlabel("Keys")
    plt.ylabel("Total counts (log scale)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir + "/dataset_key_diversity.pdf")
    print(f"Plot saved as dataset_key_diversity.pdf to {output_dir}")


if __name__ == "__main__":
     key_diversity()    