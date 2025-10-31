# 8) Train Word2Vec & FastText (combined cleaned_text)
from gensim.models import Word2Vec, FastText
import pandas as pd
from pathlib import Path

files = [
    "labeled-sentiment_2col.xlsx",
    "test__1__2col.xlsx",
    "train__3__2col.xlsx",
    "train-00000-of-00001_2col.xlsx",
    "merged_dataset_CSV__1__2col.xlsx",
]

sentences = []
for f in files:
    df = pd.read_excel(f, usecols=["cleaned_text"])
    sentences.extend(df["cleaned_text"].astype(str).str.split().tolist())

Path("embeddings").mkdir(exist_ok=True)

w2v = Word2Vec(
    sentences=sentences,
    vector_size=300,
    window=5,
    min_count=3,
    sg=1,
    negative=10,
    epochs=10
)
w2v.save("embeddings/word2vec.model")

ft = FastText(
    sentences=sentences,
    vector_size=300,
    window=5,
    min_count=3,
    sg=1,
    min_n=3,
    max_n=6,
    epochs=10
)
ft.save("embeddings/fasttext.model")

print("Saved embeddings.")
