import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, flash, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "news_sample.csv")
MODEL_PATH = os.path.join(BASE_DIR, "news_model.joblib")
app = Flask(__name__)
app.secret_key = "change-this-for-production"

# Bundled starter data lets the app run immediately. For meaningful evaluation,
# replace it with the UCI News Aggregator dataset using prepare_dataset.py.
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["title", "category"])
    df["title"] = df["title"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    return df[df["title"].ne("") & df["category"].ne("")]

def train_model():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["title"], df["category"], test_size=0.25, random_state=42,
        stratify=df["category"] if df["category"].value_counts().min() >= 2 else None
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english",
                                  ngram_range=(1, 2), max_features=20000)),
        ("nb", MultinomialNB(alpha=0.5))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    labels = sorted(df["category"].unique().tolist())
    report = classification_report(y_test, predictions, labels=labels,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, predictions, labels=labels).tolist()
    joblib.dump(model, MODEL_PATH)
    return model, df, {
        "accuracy": round(float(accuracy) * 100, 2),
        "rows": len(df),
        "classes": labels,
        "report": report,
        "cm": cm,
        "labels": labels
    }

def get_model_and_stats(force=False):
    # Train on startup for simplicity, ensuring the selected dataset is used.
    return train_model()

@app.route("/", methods=["GET", "POST"])
def index():
    model, df, stats = get_model_and_stats()
    prediction = None
    if request.method == "POST":
        article = request.form.get("article", "").strip()
        if not article:
            flash("Please enter a headline or article first.", "warning")
        else:
            pred = model.predict([article])[0]
            probs = model.predict_proba([article])[0]
            classes = model.named_steps["nb"].classes_
            ranked = sorted(zip(classes, probs), key=lambda item: item[1], reverse=True)
            prediction = {
                "text": article,
                "topic": pred,
                "confidence": round(float(max(probs)) * 100, 1),
                "alternatives": [(label, round(float(prob) * 100, 1)) for label, prob in ranked[:4]]
            }
    topic_counts = df["category"].value_counts().to_dict()
    return render_template("index.html", stats=stats, prediction=prediction,
                           topic_counts=topic_counts)

@app.route("/batch", methods=["GET", "POST"])
def batch():
    model, df, stats = get_model_and_stats()
    results = None
    if request.method == "POST":
        upload = request.files.get("file")
        if not upload or not upload.filename:
            flash("Choose a CSV file to upload.", "warning")
        else:
            try:
                uploaded_df = pd.read_csv(upload)
                # Accept title, headline, text, or article as the text column.
                candidates = ["title", "headline", "text", "article"]
                column = next((c for c in candidates if c in uploaded_df.columns.str.lower()), None)
                if column is None:
                    flash("CSV needs a text column named title, headline, text, or article.", "danger")
                else:
                    actual_column = next(c for c in uploaded_df.columns if c.lower() == column)
                    texts = uploaded_df[actual_column].fillna("").astype(str)
                    preds = model.predict(texts)
                    probs = model.predict_proba(texts)
                    classes = model.named_steps["nb"].classes_
                    results = []
                    for text, pred, probrow in zip(texts, preds, probs):
                        best = float(max(probrow)) * 100
                        results.append({"text": text, "topic": pred, "confidence": round(best, 1)})
            except Exception as exc:
                flash(f"Could not read that CSV: {exc}", "danger")
    return render_template("batch.html", stats=stats, results=results)

@app.route("/evaluation")
def evaluation():
    model, df, stats = get_model_and_stats()
    labels = stats["labels"]
    matrix = stats["cm"]
    report_rows = []
    for label in labels:
        item = stats["report"].get(label, {})
        report_rows.append({
            "label": label,
            "precision": round(item.get("precision", 0) * 100, 1),
            "recall": round(item.get("recall", 0) * 100, 1),
            "f1": round(item.get("f1-score", 0) * 100, 1),
            "support": int(item.get("support", 0))
        })
    return render_template("evaluation.html", stats=stats, report_rows=report_rows,
                           matrix=matrix, labels=labels)

if __name__ == "__main__":
    app.run(debug=True)
