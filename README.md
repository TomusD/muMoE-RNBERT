# μMoE-RNBERT: Interpretable Music Harmonic Analysis Through Multilinear Mixture of Experts

This repository contains the official implementation of the model described in the ICASSP 2026 paper **"Interpretable Music Harmonic Analysis Through Multilinear Mixture of Experts"**.

If you find this work useful in your research, please cite the paper: 

```bibtex
@INPROCEEDINGS{11463689,
  author={Triantafyllou, Thanasis and Nicolaou, Mihalis A. and Panagakis, Yannis},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Interpretable Music Harmonic Analysis Through Multilinear Mixture of Experts}, 
  year={2026},
  pages={16082-16086},
  doi={10.1109/ICASSP55912.2026.11463689}}


```
Or my thesis, which has a bit more information about the project and some ablation studies:

```bibtex
@mastersthesis{uoadl:5325645,
    BIBTEX_ENTRY = "masterthesis",
    year = "2026",
    school = "National and Kapodistrian University of Athens",
    author = "Triantafyllou Athanasios",
    title = "Interpretable and Efficient Music Harmonic Analysis Using Multilinear Mixture of Experts and Parameter-Efficient Fine-Tuning Methods"
}

```

## Acknowledgments

This work builds directly upon the **RNBERT** architecture. We extend their state-of-the-art RNA model by replacing standard feed-forward layers with Multilinear Mixture of Experts (μMoE) layers for a more interpretable model, without sacrificing performance.

* **Original RNBERT Repository:** [https://github.com/malcolmsailor/rnbert](https://github.com/malcolmsailor/rnbert)

## 1. Installation & Setup

To reproduce our results, you must first create a working RNBERT environment. Please follow the installation instructions in the [RNBERT README](https://github.com/malcolmsailor/rnbert) to download the datasets and preprocess the data.

You may need to run the following additional commands to configure the environment for μMoE-RNBERT:

### Environment Configuration

Ensure your Python path includes the necessary submodules for data representation. This may be useful when building the data(step 2 in RNBERT README) :

```bash
export PYTHONPATH=write_seqs:reprs:music_df

```

### Dependency Installation

You may need `torch-scatter` and `torch-sparse`. Note that the installation URL depends on your specific PyTorch and CUDA versions. For example, for PyTorch 2.0.1 and CUDA 11.8:

```bash
# Example for Torch 2.0.1 + CUDA 11.8
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

```

*Check [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for the URL matching your system.*

Once you have completed the RNBERT setup (i.e., you can fine-tune a basic RNBERT model), you are ready to train a μMoE-RNBERT model and reproduce the results or build on it.

---

## 2. Training μMoE-RNBERT

This command fine-tunes the MusicBERT backbone using the μMoE architecture. It freezes the first 9 pre-trained layers and applies the Multilinear Mixture of Experts (CP decomposition) to the final 3 layers, using 48 experts as described in the paper.

> **Important:** Ensure you have downloaded the pre-trained MusicBERT checkpoint before running this command.

```bash
python musicbert_fork/training_scripts/train_chord_tones.py \
    -a base \
    -d ${RNDATA_ROOT-${HOME}/datasets}/rnbert_rn_uncond_data_bin \
    --multitask \
    --validate-interval-updates 2500 \
    --lr 0.00025 \
    --freeze-layers 9 \
    --use-mumoe \
    --mumoe-method CP \
    --mumoe-layers 3 \
    --mumoe-num-experts 48 \
    --total-updates 50000 \
    --warmup-updates 2500

```

---

## 3. Extracting Expert Interpretability Data

Once training is complete, use this command to run inference on the test set and extract the expert coefficients. This script saves a CSV file containing the activation values for every expert on every note, which we will use for the visualizations and analysis. Unfortunately, it's slow, and as we increase the number of experts, it gets slower.

```bash
python musicbert_fork/eval_scripts/save_multi_task_predictions.py \
    --data-dir /path/to/your/data_bin \
    --checkpoint /path/to/your/best_checkpoint.pt \
    --output-folder /path/to/save/predictions \
    --interpretation-data-path /path/to/save/csv_file \
    --layer-coeffs 11 \
    --task musicbert_multitask_sequence_tagging \
    --batch-size 4 \
    --overwrite

```

---

## 4. Post-Processing: Deduplicating Expert Coefficients

Because the model uses a segmented window, we have overlapping coefficients, some notes appear multiple times in the raw CSV output. This script deduplicates these duplicates by averaging the coefficients or dropping duplicates, to create a clean, coefficients set for analysis.

```bash
python score_experts_viz/dedouble_csv.py \
    --inter-path /path/to/your/raw_interpretation_data.csv \
    --output-dir /path/to/output_directory \
    --mean

```

Use **`--mean`** to average overlapping coefficients or **`--drop`** to keep only the first occurrence.

---

## 5. Quantitative Analysis: Expert Specialization

To analyze what musical concepts experts are learning, use the following script. This generates bar charts showing the count of occurrences of predicted labels when a specific expert is active(above the threshold value). 

This corresponds to the "Quantitative Analysis" section of the paper, where we count label occurrences above a coefficient threshold. **The expert index won't be the same as the paper because of randomness in the model, but the patterns will be.**

```bash
python score_experts_viz/task_prediction_analysis.py \
    --inter-path /path/to/deduplicated_data_mean.csv \
    --output-dir /path/to/save/plots \
    --expert 17 \
    --threshold 0.4 \
    --all-analysis

```

* **`--expert`**: The index of the expert you want to analyze (e.g., `17`).
* **`--threshold`**: The activation value above which an expert is active (e.g., `0.4`).

---

## 6. Qualitative Analysis: Piano Roll Visualization

To inspect the model's behavior on specific musical scores, use this script to generate piano roll visualizations. These plots overlay the expert's activation as a heatmap onto each note of the musical score, allowing you to see exactly where and when an expert activates (e.g., during specific key modulations). **The expert index won't be the same as the paper because of randomness in the model, but the patterns will be.**

This corresponds to the "Qualitative Analysis" section of the paper.

```bash
python score_experts_viz/expert_analysis.py \
    --inter-path /path/to/deduplicated_data_mean.csv \
    --output-dir /path/to/save/pianorolls \
    --score BI73 \
    --expert 17 \
    --threshold 0.4 \
    --n-notes 50000 \
    --notes-analysis

```

* **`--score`**: The specific Score name from the dataset you want to visualize (e.g., `BI73`, which is Chopin's Mazurka No.1 (BI 73)). Must be an exact match of the name of the score in the dataset.
* **`--expert`**: The index of the expert you want to analyze (e.g., `17`).
* **`--threshold`**: Coefficients below this value will appear black/inactive in the piano roll.
* **`--n-notes`**: Limits the visualization to a specific number of notes.

---

## Resources & Visualizations

To enable direct analysis of our findings without the need for training or heavy inference, we provide pre-computed expert coefficient files and a collection of visualizations.

### Available Resources:

* **Expert Coefficient CSVs:** Ready-to-use data for analysis. **[Download Here](https://drive.google.com/file/d/1Wr8UE2NEjE0cWyxhH3Qeae8gHXRd--2x/view?usp=sharing)**
    * `coeff_experts_48.csv` (Used in main paper analysis)
    * `coeff_experts_256.csv`
    * `coeff_experts_1024.csv`


* **Extended Analysis:**
    * **Qualitative:** Additional piano roll heatmaps for various experts and scores. **[Download Here](https://drive.google.com/drive/folders/1KwWs4h1KsSwv5QH58ZuxVYPmCQ_EMcuM?usp=sharing)**
    * **Quantitative:** Full bar charts for expert specializations across different tasks and harmonic functions. **[Download Here](https://drive.google.com/drive/folders/1AKWqigeSvJQbpI9IlTOg0RfSmEoGqj4u?usp=sharing)**

---
