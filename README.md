# News Topic Classifier Using Naive Bayes

## Project Overview

The **News Topic Classifier** is a machine learning-based web application that automatically categorizes news articles into different topics using **Multinomial Naive Bayes** and **TF-IDF (Term Frequency–Inverse Document Frequency)**. Developed using Flask, the application provides an interactive interface where users can enter news text, upload CSV files for batch classification, and explore model performance.

This project demonstrates the practical application of Natural Language Processing (NLP) and supervised machine learning for automated news categorization.

## Objectives

- Automatically classify news articles into predefined categories.
- Apply NLP techniques to convert text into numerical features.
- Implement the Multinomial Naive Bayes classification algorithm.
- Develop an interactive web application using Flask.
- Evaluate model performance using standard classification metrics.

## Features

- **Single Article Classification:** Enter a headline or article and predict its topic.
- **Batch Classification:** Upload a CSV file to classify multiple news articles.
- **Topic Exploration:** Explore the distribution of categories in the dataset.
- **Model Evaluation:** View accuracy, precision, recall, F1-score, and the confusion matrix.
- **Interactive Interface:** Access the model through a user-friendly web dashboard.

## Machine Learning Approach

The project uses the following pipeline:

1. **Text Preprocessing:** Prepare news text for analysis.
2. **TF-IDF Vectorization:** Convert words and phrases into numerical feature vectors.
3. **Multinomial Naive Bayes:** Train a supervised classifier to predict news categories.
4. **Model Evaluation:** Evaluate predictions using a held-out test dataset.
5. **Flask Integration:** Display predictions and evaluation results through a web interface.

## Dataset

The project supports the **BBC News Classification dataset**, which contains approximately 2,225 news articles across five categories:

- Business
- Entertainment
- Politics
- Sport
- Technology

Dataset source: [BBC Full Text and Category Dataset – Kaggle](https://www.kaggle.com/datasets/yufengdev/bbc-fulltext-and-category)

The dataset must be downloaded separately and prepared using the included `prepare_dataset.py` script. Refer to the project instructions before running the application with the BBC dataset.

## Technologies Used

- **Programming Language:** Python
- **Web Framework:** Flask
- **Machine Learning:** Scikit-learn
- **Natural Language Processing:** TF-IDF
- **Algorithm:** Multinomial Naive Bayes
- **Data Processing:** Pandas
- **Model Serialization:** Joblib
- **Frontend:** HTML, CSS, Bootstrap, JavaScript

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/news-topic-classifier.git
cd news-topic-classifier
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

Download and extract the BBC News dataset, place its CSV file in the project's `data/` folder, and run:

```bash
python prepare_dataset.py
```

### 4. Run the Application

```bash
python app.py
```

### 5. Open the Website

Navigate to:

```text
http://127.0.0.1:5000
```

## Evaluation Metrics

The application reports the following metrics:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted category assignments that are correct.
- **Recall:** Proportion of actual category examples correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Breakdown of correct and incorrect predictions by category.

The reported metrics depend on the dataset and test split used.

## Future Enhancements

- Compare Naive Bayes with Logistic Regression and Support Vector Machines.
- Add interactive charts for category-wise performance.
- Support additional datasets and news categories.
- Improve text preprocessing and hyperparameter tuning.
- Deploy the application on a cloud hosting platform.

## Academic Purpose

This project was developed as a **Fundamentals of Artificial Intelligence (FAI) case study** to demonstrate supervised machine learning, natural language processing, text classification, model evaluation, and web application development.

## License and Dataset Usage

The project code may be shared under a suitable open-source license. The BBC news articles are third-party content; check the dataset's licensing and usage terms before redistributing the raw dataset.

**Keywords:** Machine Learning, Natural Language Processing, News Classification, Naive Bayes, TF-IDF, Flask, Python, Text Mining.
