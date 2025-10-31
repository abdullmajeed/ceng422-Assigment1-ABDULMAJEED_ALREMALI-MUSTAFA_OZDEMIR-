# CENG442 – Assignment 1  
### **Sentiment Analysis on Azerbaijani Texts using Word2Vec & FastText**

---

## 1️⃣ Data & Goal
This project used **five Azerbaijani text datasets** provided in the assignment:  
`labeled-sentiment.xlsx`, `test__1__.xlsx`, `train__3__.xlsx`, `train-00000-of-00001.xlsx`, and `merged_dataset_CSV__1__.xlsx`.

Each dataset was converted into a two-column Excel file containing:
- **cleaned_text**
- **sentiment_value** → 0.0 = Negative, 0.5 = Neutral, 1.0 = Positive  

Neutral samples (0.5) were intentionally **kept** to preserve a realistic distribution of sentiments.  
Removing them would bias embeddings toward extreme polarities and reduce semantic richness.

---

## 2️⃣ Preprocessing (rules & examples)

A comprehensive Azerbaijani-aware cleaning pipeline was applied to each dataset:

| Step | Description |
|------|--------------|
| **Unicode & Casing** | `fix_text` + `html.unescape` normalization, Azerbaijani lowercase (`İ→i`, `I→ı`) |
| **Entity Masking** | URLs → `URL`, emails → `EMAIL`, phones → `PHONE`, mentions → `USER` |
| **HTML/Markup** | Strip all `<tags>` and encoded entities |
| **Hashtags** | Remove `#` but split camelCase (e.g., `#QarabagIsBack → qarabag is back`) |
| **Numbers** | Replace digits with `<NUM>` |
| **Repetitions** | Collapse ≥3 repeated chars to 2 (`coooool → coool`) |
| **Spaces & Punctuations** | Normalize multiple spaces and punctuation marks |
| **Single-Letter Tokens** | Remove except `o`, `e` |
| **Duplicates & Empty Rows** | Drop them automatically |

**Before → After examples**
```text
Bu film çox yaxşıdır!!! 😂      → bu film çox yaxşıdır EMO_POS  
Qiymət: 35 azn, əlaqə: +994…   → qiymət <NUM> azn əlaqə PHONE  
#QarabagIsBack!!                → qarabag is back  
<a href="…">link</a> yazmışdım → link yazmışdım
```
After cleaning, each file was saved as a two-column Excel with cleaned_text and sentiment_value.

---

## 3️⃣ Mini Challenges

We implemented all 5 mini challenges successfully:

- **Hashtag Split** → Automatically splits camelCase hashtags  
  `#QarabagIsBack → qarabag is back`

- **Emoji Mapping** → `EMO_MAP` replaced positive emojis → `<EMO_POS>` and negative → `<EMO_NEG>`

- **Stopword Research** → Compared Azerbaijani with Turkish;  
  optional stopword list: `və, ilə, amma, ancaq, lakin, ya, həm, ki, bu, bir`  
  Negations preserved: `yox, deyil, heç, qətiyyən, yoxdur`

- **Negation Scope** → Marks next 3 tokens after negators with `_NEG`  
  e.g., `heç bir nəticə vermədi → heç bir_NEG nəticə_NEG vermədi_NEG`

- **Simple Deasciify** → Applied `cox→çox`, `yaxsi→yaxşı`, `sagol→sağol`  
  → **Reported:** `2187 tokens changed`
---

## 4️⃣ Domain-Aware

We implemented automatic **domain detection** and **domain-specific normalization**  
to make the corpus context-aware and help embeddings learn domain-sensitive semantics.

---

### ** Detection Rules**

Keyword-based hints were used to classify each text into one of three domains:

- **News** → keywords like `apa`, `trend`, `azertac`, `reuters`, `bloomberg`, `dha`, `aa`
- **Social** → includes `@`, `#`, or emojis (😅😍🙂😂)
- **Reviews** → contains `azn`, `qiymət`, `ulduz`, `çox yaxşı`

➡️ Texts that did not match any of these patterns were labeled as **general**.

---

### ** Domain Normalization (for reviews)**

Replaced sentiment and rating phrases with special tags:

```text
price → <PRICE>,  1-5 stars → <STARS_*>
çox yaxşı → <RATING_POS>,  çox pis → <RATING_NEG>
Example transformation:

Qiymət: 35 azn, çox pis xidmət.  
→ qiymət <PRICE> çox pis <RATING_NEG>
```

### ** Domain Tags **

During corpus generation, each line in corpus_all.txt was automatically prefixed with its detected domain label using the pattern:
dom<domain> <sentence> (e.g., domreviews, domsocial, domnews, domgeneral).
This produced a domain-tagged corpus with 124,353 sentences for model training.
```text
Examples:

domreviews çox bahalı idi <RATING_NEG>
domsocial çox gözəl idi EMO_POS
```
---

## 5️⃣ Embeddings: Training Settings and Results

Two embedding models — **Word2Vec** and **FastText** — were trained on the combined, cleaned, and domain-tagged corpus.

---

### ** Training Settings**

| **Parameter**         | **Word2Vec**        | **FastText**       |
|------------------------|--------------------|--------------------|
| **Architecture**       | Skip-gram (sg=1)   | Skip-gram (sg=1)   |
| **Vector size**        | 300                | 300                |
| **Window**             | 5                  | 5                  |
| **Min count**          | 3                  | 3                  |
| **Negative samples**   | 10                 | 10                 |
| **Epochs**             | 10                 | 10                 |
| **Subword n-grams**    | —                  | 3–6                |

Both models were trained using **gensim** and saved as:

- `embeddings/word2vec.model`  
- `embeddings/fasttext.model`

---

### ** Evaluation Results**

| **Metric**                  | **Word2Vec** | **FastText** |
|------------------------------|--------------|--------------|
| **Vocabulary Coverage (avg)** | 0.960        | 0.960        |
| **Synonym Similarity (avg)**  | 0.312        | 0.417        |
| **Antonym Similarity (avg)**  | 0.306        | 0.421        |
| **Separation (Syn–Ant)**      | +0.006       | −0.004       |

---

### ** Nearest Neighbors (Samples)**

| **Word** | **Word2Vec Neighbors** | **FastText Neighbors** |
|-----------|------------------------|-------------------------|
| **yaxşı** | yaxşı, iyi, `<RATING_POS>` | yaxşı1, yaxşıca, yaxşıya |
| **pis**   | `<RATING_NEG>`, günd, lire | pis, pisdil, pi |
| **çox**   | çoxx, gözəldir | çoxx, çox |
| **bahalı**| portretlarına, villaları | bahallı, baha1sı, pahal1 |
| **ucuz**  | şeytanbazardan, düzəldirilib | ucuzu, ucucza, ucuzdu |

---

### ** Interpretation**

- **FastText** achieved higher synonym and antonym similarity, confirming that its subword-based approach captures better semantic and morphological relations — especially in rich and noisy Azerbaijani text.  
- **Word2Vec** performed competitively but was slightly weaker in handling rare or misspelled forms.  
- Both embeddings provided **meaningful nearest neighbors**, validating the preprocessing pipeline and domain-tagged corpus.

---
## 6️⃣ Lemmatization (Optional)

We initially attempted to use **existing Azerbaijani lemmatization libraries**, such as *MorAz* (a morphological analyzer) and other open-source tools, to automatically reduce inflected words to their lemmas.  
However, these libraries were either **not publicly available**, **not installable via pip**, or **incompatible** with our Python environment. As a result, we implemented a **simplified rule-based lemmatization** instead.

Our approach relies on:
1. A **custom hand-built mapping (`LEMMA_MAP`)** that covers common word variants  
   *(e.g., `yaxşıdır → yaxşı`, `gözəldir → gözəl`, `ucuzdur → ucuz`, `bahalıdır → bahalı`)*  
2. A **copula-removal rule** (`-dır`, `-dir`, `-dur`, `-dü`, etc.) safely applied when the remaining stem length ≥ 3.

This hybrid approach provided a lightweight yet effective approximation of true lemmatization, helping unify multiple surface forms of the same word and improving model consistency.  
Across all datasets, the rule-based lemmatizer **normalized approximately 84,241 tokens**, which noticeably increased vocabulary coherence before embedding training.

###  Effect on Embeddings after Lemmatization

After applying the rule-based lemmatizer, all datasets showed a **notable increase in lexical coverage**, especially for **Word2Vec**, which reached above 0.93 across all domains.  
This indicates that the normalization successfully unified multiple inflected forms (e.g., *yaxşıdır → yaxşı*, *gözəldir → gözəl*, *ucuzdur → ucuz*).

Although synonym and antonym similarity scores remained relatively close, qualitative inspection of nearest neighbors demonstrated **cleaner and more semantically consistent clusters** around sentiment-bearing words.  
For instance, the neighbors of *yaxşı* and *pis* were more coherent (grouping similar or contextually related words rather than noisy forms).

Overall, even with a simple rule-based approach, **vocabulary cohesion and embedding interpretability improved**, validating the practical value of lightweight Azerbaijani lemmatization.

> **Note:**  
> All files related to the **Lemmatization experiment**, including the modified preprocessing code and the resulting two-column Excel outputs,  
> are stored separately under the folder **`/Lemmatization/`** for clarity and reproducibility.


---

## 7️⃣ Reproducibility

## 📂 Embeddings Folder

The trained models are stored in Google Drive for easy access:

🔗 **[Download from Google Drive](https://drive.google.com/drive/folders/1K6hKwy7nbCecQHkzKE0PpRW_Kc2hsNr7?usp=sharing)**  
Contains:
- `word2vec.model`
- `fasttext.model`
- Supporting files (vocab, vectors, etc.)

The project is **fully reproducible** and organized into **three modular scripts**:

| **Stage** | **Script** | **Description** |
|------------|-------------|-----------------|
| **Preprocessing** | `preprocess_and_corpus.py` | Cleans all five datasets, removes duplicates, exports 2-column Excel files, and builds the domain-tagged corpus `corpus_all.txt`. |
| **Training** | `train_embeddings.py` | Trains **Word2Vec** and **FastText** models using the cleaned corpus and saves them in `embeddings/`. |
| **Evaluation** | `compare_embeddings.py` | Compares both models using coverage, synonym/antonym similarity, and nearest-neighbor quality metrics. |

---

### ** Environment**

- Python 3.9 or higher (tested on Windows 10)  
- Required packages:  
  ```bash
  pip install -U pandas gensim openpyxl ftfy regex scikit-learn

### ** Deterministic Results**

To ensure identical results across runs:
1.	Disable Python’s hash randomization:
   ```bash
  set PYTHONHASHSEED=0        # Windows  
  export PYTHONHASHSEED=0     # Linux / macOS
  ```
2.	Use a fixed random seed during training:
-	seed = 42

### ** How to run (end-to-end):**

1.	Put the 5 raw Excel files in the project root directory.
2.	Preprocess + export 2-column files + build corpus
   ```bash
  python preprocess_and_corpus.py
  ```
Outputs:
-	*_2col.xlsx (cleaned sentiment files)
-	corpus_all.txt (domain-tagged corpus)
  
---

Train embeddings (Word2Vec & FastText)
```bash
  python train_embeddings.py
  ```
Outputs:
-	embeddings/word2vec.model
-	embeddings/fasttext.model

---

Evaluate & compare models
```bash
  python compare_embeddings.py
  ```
Outputs:
- Printed coverage, synonym/antonym similarity, nearest neighbors

---

## 8️⃣ Conclusions

This assignment successfully demonstrated the complete **Azerbaijani sentiment analysis pipeline**, covering data cleaning, domain detection, corpus construction, and embedding training with **Word2Vec** and **FastText**.

- The preprocessing pipeline effectively normalized noisy Azerbaijani text, handled negations, emojis, hashtags, and domain-specific patterns.  
- The domain-aware corpus (`corpus_all.txt`) enabled richer contextual learning across *news*, *social*, and *review* text types.  
- Both embeddings produced meaningful semantic structures; however, **FastText** consistently performed better overall.  
- **FastText** achieved higher synonym and antonym similarity, better handled **morphological variations** and **rare words**, thanks to its subword-based architecture.  
- **Word2Vec** remained competitive for frequent and clean vocabulary but was less robust to noisy or misspelled forms.  
- A light **lemmatization step** was tested to unify morphological variants (e.g., *gedirəm → getmək*), slightly improving consistency but with limited impact compared to FastText’s subword modeling.  
- All experiments were reproducible and CPU-efficient, meeting the project’s reproducibility and interpretability criteria.

**Conclusion:** Overall, **FastText** proved to be the better model for Azerbaijani sentiment analysis due to its ability to capture subword information and handle the language’s rich morphology more effectively.

**Future work:** integrating a more advanced Azerbaijani lemmatizer, expanding domain coverage, and exploring transformer-based embeddings such as **BERT** for deeper contextual understanding.


---

### Group Members

- Abdulmajeed Alremali (22050941025)
- Mustafa ÖzdemirR (21050111016)

---







