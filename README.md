# NewsLens — News Topic Classifier (BBC News dataset)

A Flask FAI case study using **TF-IDF + Multinomial Naive Bayes** to classify news articles.

## Recommended dataset: BBC News Classification

Use the BBC full-text dataset containing **2,225 articles across five topics**:
- Business
- Entertainment
- Politics
- Sport
- Tech

Download:
https://www.kaggle.com/datasets/yufengdev/bbc-fulltext-and-category

Alternative dataset page:
https://huggingface.co/datasets/SetFit/bbc-news

**Dataset note:** BBC article text is copyrighted by the BBC. Use it for your academic demonstration in accordance with the dataset source's terms; do not redistribute the raw articles as part of your project ZIP.

## 1. Install dependencies

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

## 2. Add the BBC dataset

1. Download the dataset from the link above.
2. Extract it.
3. Copy its CSV file into this project's `data/` folder. Do not name it `news_sample.csv`; that filename is reserved for the prepared output.
4. Run:

```bash
python prepare_dataset.py
```

The script detects common column names such as `text`, `article`, or `content` and `category`, then creates `data/news_sample.csv`.

## 3. Run the Flask app

```bash
python app.py
```

Open http://127.0.0.1:5000

## Features

- Classify a pasted news article
- Batch-classify a CSV of headlines/articles
- Explore topic distribution
- See test-set accuracy, precision, recall, F1-score, and confusion matrix

## Algorithm

1. Split labelled articles into training and test sets.
2. TF-IDF converts article text into numeric word/phrase features.
3. Multinomial Naive Bayes learns how word patterns relate to topics.
4. Evaluate on held-out test articles and display the predicted topic.

The app uses the article text stored in the `title` column internally; this column name is retained for compatibility with the starter version. For the BBC dataset it contains full article text, not just headlines.

## CSV upload format

Use one text column named `title`, `headline`, `text`, or `article`.

```csv
title
"Researchers report a new development in technology..."
"Company earnings increase as markets respond..."
```

## Important limitations

- BBC News articles are from a particular time period and publisher; the model may not generalize to modern articles or other publishers.
- A prediction probability is a model estimate, not a guarantee of correctness.
- Avoid claiming a particular accuracy before running the app and recording the actual held-out test metrics.

## Project structure

```text
fai_news_topic_classifier/
├── app.py
├── prepare_dataset.py
├── requirements.txt
├── README.md
├── data/
│   └── news_sample.csv
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── batch.html
│   └── evaluation.html
└── static/
    └── style.css
```
