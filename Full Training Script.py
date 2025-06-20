from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pandas as pd
import pickle

# Load your CSV (check if columns are correct)
df = pd.read_csv("D:/backup work/FYP/FYP 2/movies.csv")
df.columns = df.columns.str.strip()  # Optional cleanup

# Create model
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train
model.fit(df['description'], df['genre'])

# Save model
with open("genre_model.pkl", "wb") as f:
    pickle.dump(model, f)
