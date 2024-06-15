import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
news_data = pd.read_csv('C:/Users/Hitesh/Downloads/fake_news/train.csv')

# Data preprocessing
news_data.fillna('', inplace=True)
news_data['content'] = news_data['title'] + ' ' + news_data['author']

# Define stemming function
def stemming(content):
    if isinstance(content, str):
        porter_stemmer = PorterStemmer()
        sc = re.sub('[^a-zA-Z]', ' ', content)
        sc = sc.lower().split()
        stop_words = set(stopwords.words('english'))
        sc = [porter_stemmer.stem(word) for word in sc if word not in stop_words]
        sc = ' '.join(sc)
        return sc
    else:
        return ''

# Apply stemming
import nltk
nltk.download('stopwords')
news_data['content'] = news_data['content'].apply(stemming)

# Define features and labels
X = news_data['content'].values
Y = news_data['label'].values

# Vectorize the text data
vect = TfidfVectorizer()
vect.fit(X)
X = vect.transform(X)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=10, test_size=0.2, stratify=Y)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(x_train, y_train)

# Predict and evaluate the model
train_predictions = model.predict(x_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print("Training Accuracy:", train_accuracy)

test_predictions = model.predict(x_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print("Test Accuracy:", test_accuracy)
