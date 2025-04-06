# Neural Collaborative Filtering (NCF) 

This repository contains a complete movie recommendation pipeline built using Neural Collaborative Filtering (NCF) for assignment 1 at Leiden University.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
  - [Step 1: Convert DAT Files to CSV](#step-1-convert-dat-files-to-csv)
  - [Step 2: Data Preprocessing and Negative Sampling](#step-2-data-preprocessing-and-negative-sampling)
  - [Step 3: Model Training](#step-3-model-training)
  - [Step 4: Model Evaluation](#step-4-model-evaluation)
- [Project Structure](#project-structure)
- [Notes](#notes)

## Overview

This project implements an NCF model that combines Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP) components for movie recommendation. Positive interactions are defined as ratings ≥ 4, and negative samples are generated to balance the dataset. The project is built using PyTorch along with standard data processing libraries.

## Requirements

- Python 3.7+
- [PyTorch](https://pytorch.org/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/)
- [tqdm](https://github.com/tqdm/tqdm)

You can install the required packages via pip:

```bash
pip install torch pandas numpy scikit-learn tqdm
```

## Data Preparation
Place DAT Files:
Place your original MovieLens .dat files (ratings.dat, users.dat, movies.dat) in the ./dataset/ directory.

Conversion to CSV:
The script can convert the .dat files to CSV format. This step is optional if you already have the CSV files.

Preprocessing:

The ratings are preprocessed to convert user IDs to 0-based indexing.

Only positive interactions (ratings ≥ 4) are kept and labeled as 1.

Negative samples (default: 8 negatives per positive) are generated to create a balanced dataset.

The processed data is split into training, validation, and test sets.

## Project Structure
.
├── dataset/
│   ├── ratings.dat
│   ├── users.dat
│   ├── movies.dat
│   ├── ratings.csv              # Generated from ratings.dat
│   ├── users.csv                # Generated from users.dat
│   ├── movies.csv               # Generated from movies.dat
│   ├── editing_ratings.csv      # Preprocessed positive interactions
│   ├── with_negatives.csv       # Combined positive and negative samples
│   ├── train.csv                # Training split
│   ├── val.csv                  # Validation split
│   └── test.csv                 # Test split
├── <your_script_name>.py        # Main pipeline script (data processing, training, evaluation)
├── README.md
└── requirements.txt             # (Optional) Generated with: pip freeze > requirements.txt



## Usage

Run the main pipeline script by executing:

```bash
python ncf_merged.py
```

## Experiment

### Step 1: Convert DAT Files to CSV
Prompt: "Convert DAT files to CSV? (y/n):"

Action: Converts the MovieLens .dat files to CSV files and saves them in the ./dataset/ directory.

Note: Run this step only if the CSV files do not already exist.

### Step 2: Data Preprocessing and Negative Sampling
Prompt: "Preprocess ratings data? (y/n):"

Action:

Converts the ratings to 0-based indexing.

Filters out only positive interactions (ratings ≥ 4).

Generates negative samples and combines them with positive interactions.

Splits the dataset into training (70%), validation (15%), and test (15%) sets.

Note: Skip this step if you already have processed CSV files (train.csv, val.csv, test.csv) in the dataset folder.

### Step 3: Model Training
Prompt: "Train model? (y/n):"

Action:

Creates PyTorch datasets and dataloaders using the processed data.

Initializes the NCF model with default parameters.

Trains the model using Binary Cross-Entropy Loss with the Adam optimizer.

Uses early stopping (patience is configurable) and saves the best model as ncf_model.pt.

Configuration:

Training batch size is set to 1024.

Learning rate, number of epochs, and early stopping patience can be adjusted in the script.

### Step 4: Model Evaluation
Prompt: "Evaluate model? (y/n):"

Action:

Loads the best model (ncf_model.pt).

Evaluates the model on the test set using Recall@10 and NDCG@10 metrics.

Prints the evaluation results.


## Notes

If you cannot run the merged version, you can run individual files.

1. python convert.py
2. python preprocess.py
3. python negatives.py
4. python splitting.py
5. python train.py
6. python evaluate.py