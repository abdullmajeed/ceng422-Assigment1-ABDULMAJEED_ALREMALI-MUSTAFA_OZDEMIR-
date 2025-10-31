# 9) Compare Word2Vec vs FastText (simple metrics)
# Evaluate coverage, synonym/antonym similarity and nearest neighbors; report per domain if possible.

import pandas as pd
from gensim.models import Word2Vec, FastText
import re

w2v = Word2Vec.load("embeddings/word2vec.model")
ft  = FastText.load("embeddings/fasttext.model")

seed_words = [
    "yaxşı", "pis", "çox", "bahalı", "ucuz", "mükəmməl", "dəhşət",
    "<PRICE>", "<RATING_POS>"
]
syn_pairs = [("yaxşı", "ola"), ("bahalı", "qiymətli"), ("ucuz", "sərfəli")]
ant_pairs = [("yaxşı", "pis"), ("bahalı", "ucuz")]

def lexical_coverage(model, tokens):
    vocab = model.wv.key_to_index
    return sum(1 for t in tokens if t in vocab) / max(1, len(tokens))

files = [
    "labeled-sentiment_2col.xlsx",
    "test__1__2col.xlsx",
    "train__3__2col.xlsx",
    "train-00000-of-00001_2col.xlsx",
    "merged_dataset_CSV__1__2col.xlsx",
]

def read_tokens(f):
    df = pd.read_excel(f, usecols=["cleaned_text"])
    return [t for row in df["cleaned_text"].astype(str) for t in row.split()]

print("== Lexical coverage (per dataset) ==")
for f in files:
    toks = read_tokens(f)
    cov_w2v = lexical_coverage(w2v, toks)
    cov_ft  = lexical_coverage(ft, toks)  # FT still embeds OOV via subwords
    print(f"{f}: W2V={cov_w2v:.3f}, FT(vocab)={cov_ft:.3f}")

from numpy import dot
from numpy.linalg import norm

def cos(a, b): return float(dot(a, b) / (norm(a) * norm(b)))

def pair_sim(model, pairs):
    vals = []
    for a, b in pairs:
        try:
            vals.append(model.wv.similarity(a, b))
        except KeyError:
            pass
    return sum(vals) / len(vals) if vals else float('nan')

syn_w2v = pair_sim(w2v, syn_pairs)
syn_ft  = pair_sim(ft,  syn_pairs)
ant_w2v = pair_sim(w2v, ant_pairs)
ant_ft  = pair_sim(ft,  ant_pairs)

print("\n== Similarity (higher better for synonyms; lower better for antonyms) ==")
print(f"Synonyms: W2V={syn_w2v:.3f}, FT={syn_ft:.3f}")
print(f"Antonyms: W2V={ant_w2v:.3f}, FT={ant_ft:.3f}")
print(f"Separation (Syn - Ant): W2V={(syn_w2v - ant_w2v):.3f}, FT={(syn_ft - ant_ft):.3f}")

def neighbors(model, word, k=5):
    try:
        return [w for w, _ in model.wv.most_similar(word, topn=k)]
    except KeyError:
        return []

print("\n== Nearest neighbors (qualitative) ==")
for w in seed_words:
    print(f"\nW2V NN for '{w}':", neighbors(w2v, w))
    print(f"FT  NN for '{w}':", neighbors(ft,  w))

# (Optional) domain drift if you train domain-specific models separately:
# drift(word, model_a, model_b) = 1 - cos(vec_a, vec_b)
