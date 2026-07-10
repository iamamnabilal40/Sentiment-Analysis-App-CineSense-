# CineSense Movie Review Sentiment Analyzer

CineSense is an end to end natural language processing (NLP) application
that classifies movie reviews as **positive** or **negative** in real
time. It combines a classic, interpretable machine learning pipeline
(TF-IDF + Logistic Regression) with hyperparameter tuning and an
interactive Streamlit interface, making it a complete demonstration of
how a machine learning idea goes from raw text to a usable, deployed
application.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Design Decisions](#design-decisions)
- [Limitations](#limitations)
- [Possible Extensions](#possible-extensions)
- [Author](#author)

---

## Overview

Sentiment analysis is one of the most common real world applications of
NLP used in product reviews, social media monitoring, customer
feedback analysis, and content moderation. CineSense focuses on a
narrower, well defined version of this problem: classifying movie
reviews as positive or negative, and doing every part of the pipeline
properly rather than relying on a single black box call.

Rather than only training a model in a notebook, CineSense wraps the
model in a working web app, so predictions, confidence scores, and
model performance metrics are all visible and usable in one place
which is what makes it "portfolio ready" rather than just a script.

## Features

- **End to end pipeline** from labeled data, to training, to
  evaluation, to a live deployed interface, with no missing steps.
- **TF-IDF vectorization** using unigrams and bigrams to capture both
  individual sentiment words ("brilliant", "boring") and short phrases
  ("not good", "highly recommended").
- **Logistic Regression classifier**, chosen for being fast, accurate
  on text classification tasks, and easy to interpret compared to
  deep learning alternatives.
- **Hyperparameter tuning** via `GridSearchCV` with 5 fold
  cross validation, searching over vectorizer settings and
  regularization strength.
- **Live prediction interface** type any review and instantly see
  whether it's classified positive or negative, along with a
  confidence percentage.
- **Model performance dashboard** accuracy, precision, recall, F1
  score, and a confusion matrix, all generated from a proper held out
  test split rather than training data.
- **One click retraining** directly from the app interface.
- **Clean, styled UI** with color coded results (green for positive,
  red for negative) for quick visual feedback.

## Project Structure

```
cinesense/
├── app.py                    # Streamlit web application (UI + inference)
├── train_model.py             # Training pipeline, grid search, evaluation
├── dataset.py                  # Labeled movie review dataset
├── requirements.txt            # Python dependencies
├── sentiment_model.joblib      # Saved trained model (generated on first run)
├── metrics.json                # Saved evaluation metrics (generated on first run)
└── README.md
```

## Tech Stack

| Layer              | Tool / Library                              |
|---------------------|-----------------------------------------------|
| Language            | Python 3.9+                                   |
| ML / NLP             | scikit-learn (TfidfVectorizer, LogisticRegression, GridSearchCV) |
| Data handling        | pandas, joblib                               |
| Web interface        | Streamlit                                    |
| Model persistence    | joblib (serialized pipeline)                 |

## How It Works

1. **Dataset (`dataset.py`)**
   A set of hand written movie reviews (clearly positive or negative in
   tone) is combined with template generated reviews built from
   sentiment-bearing adjectives and sentence templates. This
   augmentation step increases the size and variety of the training
   data, which is a common technique for bootstrapping small,
   from scratch NLP datasets when a large labeled corpus isn't
   available.

2. **Training (`train_model.py`)**
   - Text is split into an 80/20 train/test split, stratified by
     label so both classes are represented proportionally.
   - A `Pipeline` combines a `TfidfVectorizer` (English stop words
     removed, unigrams + bigrams) with a `LogisticRegression`
     classifier.
   - `GridSearchCV` searches over `tfidf__max_features`,
     `tfidf__min_df`, and `clf__C` using 5-fold cross-validation to
     find the combination that generalizes best.
   - The best model is evaluated on the held-out test set (accuracy,
     precision, recall, F1, confusion matrix), then refit on the
     **full** dataset before being saved so the deployed model
     benefits from every available labeled example.
   - Metrics and the trained pipeline are saved to disk (`metrics.json`
     and `sentiment_model.joblib`) so the app doesn't need to retrain
     on every launch.

3. **Application (`app.py`)**
   - Loads the saved pipeline and metrics (training automatically if
     they don't exist yet).
   - **Try It tab** a text box where any review can be typed in,
     returning a Positive/Negative label with a confidence score and a
     visual probability bar.
   - **Model Performance tab** displays accuracy, precision, recall,
     F1 score, the confusion matrix, and the best hyperparameters found
     during grid search, plus a button to retrain the model from
     within the app.
   - **About tab** a summary of the project, pipeline, and possible
     extensions, for anyone viewing the deployed app who wants context
     without reading the source code.

## Installation

```bash
# 1. Clone or download this project folder
cd cinesense

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

### Train the model directly (optional)

The app trains the model automatically the first time it runs if no
saved model is found, but you can also train it manually:

```bash
python train_model.py
```

This prints the evaluation metrics to the console and saves:
- `sentiment_model.joblib` the trained, ready to use pipeline
- `metrics.json` accuracy, precision, recall, F1, and confusion matrix

### Run the app

```bash
streamlit run app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`)
open it in your browser to use the interface.

### Run in Google Colab

Since Colab doesn't run Streamlit natively without a tunneling
service, a plain Python notebook version of the training and
prediction code (no UI) can be used instead for quick experimentation
and demonstration inside a notebook.

## Model Performance

On the held out test split, CineSense currently achieves:

| Metric      | Score   |
|--------------|---------|
| Accuracy     | ~92%    |
| Precision    | ~92%    |
| Recall       | ~92%    |
| F1 Score     | ~92%    |

These numbers are generated fresh each time the model is trained and
are also visible live in the app's **Model Performance** tab, along
with the exact confusion matrix and the hyperparameters selected by
grid search.

## Design Decisions

- **Why TF-IDF + Logistic Regression instead of a deep learning
  model?** For a dataset of this size, a classical pipeline trains in
  seconds, is easy to explain end to end, and avoids the overfitting
  risk that comes with training a large neural network on a small
  corpus. It also keeps the project fully runnable without a GPU.
- **Why generate template based reviews?** A purely hand-written
  dataset of ~80 reviews was too small for the model to generalize
  well (cross-validation accuracy around 55–60%). Augmenting with
  templated reviews built from a broad vocabulary of sentiment
  adjectives increased the dataset to several hundred examples and
  raised cross validation accuracy above 90%, without requiring an
  external dataset download.
- **Why grid search instead of default hyperparameters?** Regularization
  strength and vectorizer vocabulary size have a meaningful impact on
  this kind of small to medium text classification task, so tuning
  them systematically (rather than guessing) gives a more defensible,
  reproducible result.

## Limitations

- The dataset, while augmented, is still synthetic/template-assisted
  in part rather than sourced from a large real world review corpus,
  so performance on genuinely diverse, informal, or sarcastic reviews
  (e.g. real social media text) would likely be lower than the
  reported test accuracy.
- The model only distinguishes binary sentiment (positive/negative)
  and does not detect neutral or mixed sentiment.
- Logistic Regression with TF IDF does not capture deep contextual
  meaning the way a transformer based model (e.g. BERT) would for
  example, sarcasm or negation across long sentences can still be
  misclassified.

## Possible Extensions

- Swap in a large, real world dataset (e.g. the IMDB 50k reviews
  dataset) for more robust, production-grade performance.
- Replace TF-IDF + Logistic Regression with a fine tuned BERT or
  DistilBERT model for improved handling of context and negation.
- Add batch CSV upload so multiple reviews can be scored at once and
  exported with predicted labels.
- Add multi class sentiment (positive / neutral / negative) or a
  1–5 star rating prediction instead of binary classification.
- Deploy publicly via Streamlit Community Cloud, Hugging Face Spaces,
  or a similar hosting service so the app is accessible via a public
  link rather than only running locally.

## Author

**Amna Bilal**
Robotics & Intelligent System student, Bahria University
