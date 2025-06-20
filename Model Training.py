from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pandas as pd
import pickle

# Load CSV
df = pd.read_csv("D:/backup work/FYP/FYP 2/movies.csv")

# Create pipeline
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train
model.fit(df['description'], df['genre'])

# Save model
with open("genre_model.pkl", "wb") as f:
    pickle.dump(model, f)
