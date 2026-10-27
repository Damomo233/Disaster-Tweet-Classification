# Disaster Tweet Classification: TF-IDF Baselines vs. DistilBERT

This project explores binary disaster tweet classification using both classical machine learning and transformer-based NLP methods.

The workflow includes exploratory data analysis, text preprocessing, stratified dataset splitting, TF-IDF-based machine learning baselines, and DistilBERT fine-tuning. Model performance is evaluated using F1 score on validation and held-out test sets.


## Dataset Source

Kaggle Web：https://www.kaggle.com/competitions/nlp-getting-started/overview

This project is based on Kaggle's **Natural Language Processing with Disaster Tweets** competition (`nlp-getting-started`).The original dataset and competition task are provided by Kaggle. 



## Dataset

The Kaggle training dataset contains **7,613 labeled tweets** with the following fields:

- `id`
- `keyword`
- `location`
- `text`
- `target`

The target variable indicates whether a tweet refers to a real disaster.

The original class distribution is:

| Target | Count |
|---|---:|
| 0 | 4,342 |
| 1 | 3,271 |

## Data Preprocessing

The preprocessing pipeline performs:

- Missing-value filling
- Duplicate tweet removal
- HTML entity decoding
- URL removal
- User mention (`@user`) removal
- Extra whitespace removal
- Lowercasing
- Combination of `keyword`, `location`, and cleaned tweet text into a single modeling feature



## Dataset Split

The cleaned dataset is divided into stratified training, validation, and test sets using a fixed random seed.

| Split | Samples |
|---|---:|
| Training | 6,001 |
| Validation | 751 |
| Test | 751 |

Stratification preserves a similar class distribution across all three subsets.

## Classical Machine Learning Models

Four classical models are implemented using Scikit-learn pipelines:

1. Support Vector Machine
2. Logistic Regression
3. K-Nearest Neighbors
4. Random Forest

Each model uses TF-IDF features with the following configuration:

- English stop-word removal
- Unigrams and bigrams: `ngram_range=(1, 2)`
- Minimum document frequency: `min_df=2`
- Sublinear term frequency scaling
- Maximum vocabulary size: `20,000`

The models are trained on the same training split and evaluated using F1 score.

## DistilBERT

A pretrained `distilbert-base-uncased` model is fine-tuned for binary sequence classification using Hugging Face Transformers.

Main training configuration:

```text
Model: distilbert-base-uncased
Maximum sequence length: 32
Learning rate: 2e-5
Batch size: 32
Epochs: 3
Number of labels: 2
```

Tokenized inputs include both `input_ids` and `attention_mask`.

The model is optimized using Adam with sparse categorical cross-entropy loss.

## Results

| Model | Validation F1 | Test F1 |
|---|---:|---:|
| DistilBERT | **0.758** | **0.767** |
| Logistic Regression | 0.738 | 0.763 |
| SVM | 0.725 | 0.740 |
| Random Forest | 0.701 | 0.716 |
| KNN | 0.661 | 0.642 |

DistilBERT achieved the highest validation and test F1 scores. However, the improvement over TF-IDF with Logistic Regression was relatively small: approximately **0.004 F1** on the test set.

This result suggests that classical TF-IDF-based models remain highly competitive for this relatively small tweet classification dataset, while transformer fine-tuning provides a modest additional improvement.


## Project Structure

```text
Kaggle_ML/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── notebooks/
│   └── Kaggle_ML.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Environment

The final notebook was run with:

```text
Python 3.12.10
```

Main libraries used include:

- NumPy
- pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / tf-keras
- Hugging Face Transformers

The notebook enables legacy Keras compatibility before importing the TensorFlow-based Hugging Face model:

```python
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
```

## Running the Project

Create and activate a Python environment, then install the project dependencies:

```bash
pip install -r requirements.txt
```

Place the dataset files inside:

```text
data/
```

The notebook automatically resolves the project root whether it is launched from the project root or from the `notebooks/` directory.

Then open and run:

```text
notebooks/Kaggle_ML.ipynb
```


## Summary
This project compares traditional sparse-text classification approaches with transformer fine-tuning under a shared dataset split and evaluation framework.

The final results show that DistilBERT produces the strongest overall performance, while Logistic Regression with TF-IDF achieves nearly comparable test performance with substantially simpler modeling.