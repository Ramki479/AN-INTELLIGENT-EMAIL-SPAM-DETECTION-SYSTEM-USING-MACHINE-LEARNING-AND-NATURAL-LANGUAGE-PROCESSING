import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


class Model:
    def __init__(self):
        # Load dataset
        self.df = pd.read_csv("Cleaned_Data.csv")

        self.df["Email"] = self.df["Email"].astype(str)

        X = self.df["Email"]
        y = self.df["Label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=10
        )

        self.vectorizer = TfidfVectorizer()
        X_train_vec = self.vectorizer.fit_transform(X_train)

        self.nb = MultinomialNB()
        self.lr = LogisticRegression(max_iter=1000)
        self.rf = RandomForestClassifier(n_estimators=50)
        self.knn = KNeighborsClassifier(n_neighbors=9)
        self.svm = SVC(probability=True)

        self.nb.fit(X_train_vec, y_train)
        self.lr.fit(X_train_vec, y_train)
        self.rf.fit(X_train_vec, y_train)
        self.knn.fit(X_train_vec, y_train)
        self.svm.fit(X_train_vec, y_train)

    def get_vector(self, text):
        return self.vectorizer.transform([text])

    def get_prediction(self, vector):
        preds = [
            self.nb.predict(vector)[0],
            self.lr.predict(vector)[0],
            self.rf.predict(vector)[0],
            self.knn.predict(vector)[0],
            self.svm.predict(vector)[0],
        ]

        return "Spam" if preds.count(1) >= 3 else "Non-Spam"

    def get_probabilities(self, vector):
        return [
            self.nb.predict_proba(vector)[0],
            self.lr.predict_proba(vector)[0],
            self.rf.predict_proba(vector)[0],
            self.knn.predict_proba(vector)[0],
            self.svm.predict_proba(vector)[0],
        ]
