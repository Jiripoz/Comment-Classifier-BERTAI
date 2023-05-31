import pandas as pd


df = pd.read_csv("src/b2w.csv")

df["Sentiment"] = df["rating"].apply(
    lambda score: "positive" if score > 3 else "neutral" if score == 3 else "negative"
)
df["Sentiment"] = df["Sentiment"].map({"positive": 1, "neutral": 0.5, "negative": 0})

df = df[["review_text_processed", "Sentiment"]]

df.drop(df.tail(130000).index, inplace=True)

df.to_csv("src/clear_b2w.csv")
