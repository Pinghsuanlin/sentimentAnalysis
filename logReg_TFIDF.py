import pandas as pd
import re #built in Python module for regular expressions
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import os
print(os.getcwd())
print(os.listdir(r"D:\portfolio\sentiment_BERT"))

# --- 1. Data Loading and Preprocessing ---
df = pd.read_csv('.\input\imdb.csv')

# Sampling a smaller subset for faster processing
df = df.sample(n=10000, random_state=42).reset_index(drop=True)

# Function to clean text: remove HTML tags, non-alphabetic characters, and lower case
def clean_text(text):
    text = re.sub('<br />', ' ', text) # Remove HTML break tags
    text = re.sub('[^a-zA-Z]', ' ', text) # Keep only letters
    return text.lower().strip()

# Apply cleaning to the review column
df['review'] = df['review'].apply(clean_text)

# Encode target variable: 'positive' -> 1, 'negative' -> 0
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

df.sentiment.value_counts() # 1-0 distribution is quite balanced

# Define features (X) and target (y)
X = df['review']
y = df['sentiment']

# Reasoning: Split the data into training and testing sets to evaluate the model 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 2. Feature Engineering (TF-IDF) ---
# print("Step 2: Training TF-IDF Vectorizer...")

# Setting min_df=5 removes very rare words
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5, stop_words='english')

# Fit the vectorizer on the training data to prevent data leakage
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
# Transform the test data using the fitted vectorizer
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# --- 3. Model Training (Logistic Regression) ---
# print("Step 3: Training Logistic Regression Model...")

log_reg = LogisticRegression(solver='liblinear', random_state=42)

# Train the model
log_reg.fit(X_train_tfidf, y_train)

# --- 4. Evaluation ---
# print("Step 4: Model Evaluation")

# Predict on the test set
y_pred = log_reg.predict(X_test_tfidf)

# Reasoning: The classification report provides key metrics (Precision, Recall, F1-score) 
# for each class, giving a comprehensive view of the model's performance.
print("\nClassification Report (Logistic Regression + TF-IDF):")
print(classification_report(y_test, y_pred))

"""
Test Classification Report (Logistic Regression + TF-IDF):
              precision    recall  f1-score   support

           0       0.89      0.84      0.86       992
           1       0.85      0.90      0.87      1008

    accuracy                           0.87      2000
   macro avg       0.87      0.87      0.87      2000
weighted avg       0.87      0.87      0.87      2000
"""