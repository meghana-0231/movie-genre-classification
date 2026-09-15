# Movie Genre Classification

A multiclass movie genre classification system that predicts a movie's genre from its plot summary using Natural Language Processing and Machine Learning.

## Project Overview

Movie genre classification can be treated as a text classification problem where the movie plot is used as the input and the genre is the target.

This project implements an end-to-end machine learning pipeline including:

- Data understanding and exploratory analysis
- Data quality and duplicate analysis
- Text preprocessing
- TF-IDF feature engineering
- Model comparison
- Error analysis
- Class imbalance handling
- Hyperparameter tuning
- Final model evaluation
- Prediction on new movie plots

## Dataset

The project uses the IMDb Genre Classification Dataset.

The training data contains 54,214 records across 27 movie genres.

The main fields are:

- `ID` — movie identifier
- `TITLE` — movie title
- `GENRE` — target class
- `PLOT` — movie plot summary

The supplied test set contains 54,200 records with corresponding labels provided separately.

## Data Quality

Initial analysis identified duplicate plot descriptions and a small number of plot groups with conflicting genre labels.

For model development:

- Conflicting plot groups were excluded from the modeling dataset.
- Exact duplicate `PLOT + GENRE` combinations were removed.
- The original source files were preserved unchanged.

The resulting modeling dataset contained 54,071 records with unique plot texts and no remaining conflicting genre labels.

## Methodology

### 1. Data Preparation

The raw movie records were loaded and validated for:

- Missing values
- Duplicate records
- Duplicate plot texts
- Conflicting genre labels
- Train/test plot overlap

### 2. Feature Engineering

TF-IDF was used to convert movie plot summaries into numerical feature vectors.

The vectorizer used:

- Unigrams and bigrams
- Minimum document frequency filtering
- Maximum document frequency filtering
- Sublinear term frequency
- A maximum vocabulary size of 100,000 features

### 3. Models Evaluated

The following models were compared:

- Majority-class baseline
- Multinomial Naive Bayes
- Logistic Regression
- Linear SVM

### 4. Model Improvement

Because the dataset is highly imbalanced across genres, class weighting was introduced for the Linear SVM.

Multiple values of `C` were evaluated:

- `C = 0.5`
- `C = 1.0`
- `C = 2.0`

The final configuration was selected using Macro F1 as the primary comparison metric.

## Final Model

**Balanced Linear SVM**

Configuration:

- Algorithm: Linear SVM
- `C = 0.5`
- `class_weight = "balanced"`

### Internal Validation Performance

- Accuracy: approximately 58.71%
- Macro F1: approximately 39.79%
- Weighted F1: approximately 58.55%

### Official Supplied-Test Benchmark

- Accuracy: approximately 58.25%
- Macro F1: approximately 39.10%
- Weighted F1: approximately 58.3%

The supplied dataset contains some repeated plot texts between training and test data, including cases with differing genre sets. Therefore, the official test result is presented as a supplied-dataset benchmark rather than a completely leakage-free generalization estimate.

## Example Prediction

The trained pipeline was used to predict genres for new movie plot summaries.

Pipeline:

`Movie Plot → TF-IDF → Balanced Linear SVM → Predicted Genre`

Example inputs included:

- The Missing Case
- The Last Concert
- Battle for the Kingdom

## Project Structure

```text