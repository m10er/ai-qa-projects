# api/data.py

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

# Basit örnek veri seti
data = {
    'text': [
        "Win money now", "Hello friend", "Claim your free prize", 
        "How are you?", "Get rich quick", "Let's meet for lunch",
        "Buy now", "Your invoice", "Cheap meds available", 
        "Let's catch up soon"
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1: spam, 0: not spam
}

df = pd.DataFrame(data)

# Split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)

# Model pipeline
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

model.fit(X_train, y_train)

# Metrics
def get_metrics():
    y_pred = model.predict(X_test)
    return {
        'precision': round(precision_score(y_test, y_pred), 2),
        'recall': round(recall_score(y_test, y_pred), 2),
        'f1_score': round(f1_score(y_test, y_pred), 2)
    }

def predict(text):
    return int(model.predict([text])[0])
