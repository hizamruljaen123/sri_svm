# Sentiment Analysis on YouTube & GoTube Comments Using SVM (Manual Implementation)

This project shows how to perform **sentiment analysis** on video comments using a **Support Vector Machine (SVM)** classifier.

---

## Steps Overview

1. **Data Collection**  
   - Collect comments from YouTube or GoTube via API or CSV export.  
   - Columns typically include: `comment_text`, `like_count`, `author`, etc.

2. **Data Preprocessing**  
   - Convert text to lowercase.  
   - Remove punctuation, URLs, emojis, stopwords.  
   - Tokenize text and optionally use stemming/lemmatization.

3. **Feature Extraction**  
   - Convert text into **numerical features** using:
     - Bag-of-Words (CountVectorizer)  
     - TF-IDF (TfidfVectorizer)  

4. **SVM Training**  
   - Use **scikit-learn's SVM** with a linear kernel.  
   - Split dataset into training and testing sets.  

5. **Prediction & Evaluation**  
   - Predict sentiment: Positive, Negative, Neutral  
   - Evaluate using **accuracy, precision, recall, F1-score**

---

## Python Example (Simplified)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import re
import string

# Load dataset
data = pd.read_csv('youtube_gotube_comments.csv')  # columns: comment_text, sentiment_label

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # remove URLs
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    return text

data['cleaned'] = data['comment_text'].apply(clean_text)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['cleaned'])
y = data['sentiment_label']  # 0=Negative, 1=Positive, 2=Neutral

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train, y_train)

# Predict
y_pred = svm_model.predict(X_test)

# Evaluate
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
