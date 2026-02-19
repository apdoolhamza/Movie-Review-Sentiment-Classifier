<h1 align="center">
  Movie Review Sentiment Analysis Checker  
</h1>

<p align="center">
Production-Oriented NLP Pipeline

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![View Notebook](https://img.shields.io/badge/📓%20Notebook-File-blue)](https://nbviewer.org/github/apdoolhamza/Movie-Review-Sentiment-Classifier/blob/main/Notebooks/Movie_review_sentiment_analysis.ipynb)
[![Live Demo on Hugging Face](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/apdoolhamza/MovieSentimentAI)
[![Documentation PDF](https://img.shields.io/badge/📘%20Project_Documentation-PDF-blue)](https://github.com/apdoolhamza/Movie-Review-Sentiment-Classifier/blob/main/docs/Sentiment_analysis_report.pdf)
</p>

A clean, efficient, and production-ready binary sentiment classifierfor movie reviews using classic machine learning techniques (TF-IDF + LinearSVC + calibration).  
Achieves ~88–90% F1-score on the IMDB 50k dataset while remaining lightweight, explainable, and GPU-free.

<p align="center">
  <a href="https://huggingface.co/spaces/apdoolhamza/MovieSentimentAI">
    <img src="Screenshots/Interface.jpg" width="700"/>
  </a>
</p>

[![Live Demo on Hugging Face](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/apdoolhamza/MovieSentimentAI)

## Features

- Comprehensive EDA with word clouds & length distributions
- TF-IDF with bigrams + sublinear scaling
- Hyperparameter tuning with HalvingGridSearchCV
- Probability calibration (isotonic method)
- Full evaluation suite: classification report, confusion matrix, ROC & PR curves
- Model persistence with joblib
- Quick inference function


## Project Directory Structure

```
Movie-sentiment-checker/
│
├── Model/
│   └── sentiment_model_calibrated.joblib  # Trained & calibrated ML model
│
├── Notebooks/
│   └── movie_review_sentiment_analysis.ipynb # EDA + Training pipeline
│
├── Screenshots/
│   └── (EDA plots, confusion matrix, word clouds, etc.)
│
├── docs/
│   └── Sentiment_analysis_report.pdf  # Full technical documentation
│
├── License           # Project license
├── README.md         # Project overview & usage
├── app.py            # Gradio deployment app
└── requirements.txt  # Python dependencies
```

## Installation

```bash
# Recommended: use a virtual environment
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
import joblib

# Load the saved model
model = joblib.load("sentiment_model_calibrated.joblib")

def predict_sentiment(text):
    # The cleaning function is defined in the notebook
    cleaned = clean_text(text)
    proba = model.predict_proba([cleaned])[0][1]
    label = "Positive" if proba > 0.5 else "Negative"
    return label, round(proba, 4)

# Examples
print(predict_sentiment("This movie is absolutely fantastic!"))
print(predict_sentiment("Terrible acting and boring plot. Complete waste."))
```
## Expected output:
```python
('Positive', 0.89102)
('Negative', 0.1125)
```

## Model Performance

| Metric              | Value                  |
|---------------------|------------------------|
| Accuracy            | 0.9148                 |
| F1-score (macro)    | 0.9148                 |
| ROC-AUC             | 0.9728                 |
| Training time       | 4–10 minutes (CPU)      |
| Inference speed     | < 5 ms / review        |

## License

This project is licensed under the MIT License – see the LICENSE file for details.

## Acknowledgments

* IMDB dataset: [Maas et al., 2011](https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/IMDB-Dataset.csv)
* Inspiration & best practices from scikit-learn documentation

## Contact / Contributing

Feel free to open an issue or submit a pull request.

## Author

```
Apdoolmajeed Hamza (apdoolhamza)
AI/ML Engineer | Full-stack Web Developer
```
- LinkedIn: https://www.linkedin.com/in/apdoolhamza/
- GitHub:   https://github.com/apdoolhamza/
