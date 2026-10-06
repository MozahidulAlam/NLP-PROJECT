# 📩 SMS Spam Detection using Machine Learning

A machine learning project for classifying SMS/text messages as **Spam** or **Ham (Legitimate)** using Natural Language Processing (NLP).

The project demonstrates a complete text-classification workflow, including data exploration, text preprocessing, TF-IDF feature extraction, model training, evaluation, custom prediction, and model serialization.

---

## 📌 Project Overview

Spam messages are unwanted messages that may contain advertisements, fraudulent offers, suspicious links, or other unwanted content.

In this project, I developed a machine learning model that learns patterns from labeled SMS messages and predicts whether a new message is:

* 🟢 **Ham** — legitimate message
* 🔴 **Spam** — unwanted/spam message

The project uses **TF-IDF** to convert text into numerical features and **Multinomial Naive Bayes** for classification.

---

## 🎯 Objectives

The main objectives of this project are:

1. Explore and understand the SMS dataset.
2. Clean and preprocess text data.
3. Convert text into numerical features using TF-IDF.
4. Train a text classification model.
5. Evaluate model performance using classification metrics.
6. Predict whether new messages are spam or legitimate.
7. Save the trained model and TF-IDF vectorizer for future use.
8. Analyze words that are strongly associated with spam messages.

---

## 🧠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Regular Expressions
* Joblib
* Jupyter Notebook / Google Colab

---

## 🔄 Machine Learning Workflow

```text
Raw SMS Dataset
       ↓
Exploratory Data Analysis
       ↓
Text Cleaning
       ↓
Label Encoding
       ↓
Train-Test Split
       ↓
TF-IDF Feature Extraction
       ↓
Multinomial Naive Bayes
       ↓
Prediction
       ↓
Model Evaluation
       ↓
Custom Message Prediction
       ↓
Save Model & Vectorizer
```

---

## 🔍 Exploratory Data Analysis

Before training the model, I examined:

* Dataset dimensions
* Column names
* Missing values
* Duplicate records
* Class distribution
* Basic dataset information

A visualization was also created to compare the number of Spam and Ham messages.

---

## 🧹 Text Preprocessing

The text preprocessing pipeline includes:

* Converting text to lowercase
* Removing special characters
* Removing numbers
* Removing extra whitespace

Example:

```text
Congratulations! You won $1000!!!
```

After preprocessing:

```text
congratulations you won
```

---

## 🔢 TF-IDF Feature Extraction

Machine learning algorithms cannot directly understand raw text.

Therefore, TF-IDF (Term Frequency-Inverse Document Frequency) is used to transform text into numerical feature vectors.

The vectorizer is configured with:

* English stop-word removal
* `max_df=0.95`
* `min_df=2`

The TF-IDF vectorizer is fitted only on the training data and then used to transform the test data.

---

## 🤖 Machine Learning Model

### Multinomial Naive Bayes

The project uses the **Multinomial Naive Bayes** algorithm.

It is particularly suitable for text classification problems because it works effectively with discrete/count-based or TF-IDF-style text features.

The model is trained using the TF-IDF representation of the training messages.

---

## 📊 Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

The confusion matrix helps visualize:

* True Positives
* True Negatives
* False Positives
* False Negatives

---

## 🔮 Custom Prediction

The project also allows a user to enter a new message.

The message goes through the following process:

```text
Input Message
     ↓
Text Cleaning
     ↓
TF-IDF Transformation
     ↓
Model Prediction
     ↓
Spam / Ham
     ↓
Prediction Probability
```

Example:

```text
Enter your message:

Congratulations! You have won a free prize.
```

The model returns a predicted class along with the estimated probability for Ham and Spam.

---

## 💾 Model Saving

The trained model and TF-IDF vectorizer are saved using Joblib:

```python
joblib.dump(model, "spam_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
```

This allows the trained components to be reused later without retraining the model from scratch.

---

## 🔎 Spam Word Analysis

The project also investigates which words have a strong association with Spam messages.

The analysis compares the learned log probabilities of words between the Spam and Ham classes.

This provides a basic form of model interpretation and helps identify words that contribute strongly to Spam classification.

---

## 📁 Project Structure

```text
sms-spam-detection-ml/
│
├── README.md
├── notebooks/
│   └── lab_spmEmail.ipynb
│
├── models/
│   ├── spam_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── requirements.txt
└── .gitignore
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/sms-spam-detection-ml.git
cd sms-spam-detection-ml
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the notebook

Open:

```text
notebooks/lab_spmEmail.ipynb
```

You can run it using Jupyter Notebook, JupyterLab, or Google Colab.

---

## 📦 Requirements

The main Python libraries used in this project are:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

---

## 📈 Future Improvements

Possible improvements for future versions include:

* Comparing multiple machine learning algorithms
* Hyperparameter tuning
* Better text preprocessing
* Handling multilingual SMS messages
* Testing additional NLP feature representations
* Building a web interface for real-time prediction
* Deploying the model as an API
* Adding a larger and more diverse dataset

---

## 👨‍💻 About the Project

This project was developed as part of my practical learning in **Machine Learning, Natural Language Processing, and Data Science**.

It helped me understand how a complete NLP classification pipeline works—from raw text data to preprocessing, feature engineering, model training, evaluation, and prediction.

---

## 📄 License

This repository contains my implementation and learning work. Please check the original dataset's terms and license before redistributing the dataset itself.
