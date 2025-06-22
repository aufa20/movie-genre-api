import pandas as pd

# Load movies CSV
df = pd.read_csv("movies.csv")

# Drop rows with missing values in key columns
df = df.dropna(subset=["description", "genre"])

with open("movie_train.txt", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        genre = str(row["genre"]).strip().replace(" ", "_")  # fastText uses no spaces in labels
        desc = str(row["description"]).strip().replace("\n", " ")
        f.write(f"__label__{genre} {desc}\n")
