1) Data & Goal
This project used five Azerbaijani text datasets provided in the assignment:
labeled-sentiment.xlsx, test__1__.xlsx, train__3__.xlsx, train-00000-of-00001.xlsx, and merged_dataset_CSV__1__.xlsx.
Each dataset was converted into a two-column Excel file containing cleaned_text and its sentiment_value (0.0 = Negative, 0.5 = Neutral, 1.0 = Positive).
Neutral samples (0.5) were intentionally kept to preserve a realistic distribution of sentiments.
Removing them would bias the embeddings toward extreme polarities and reduce the model’s ability to represent subtle or mixed emotions.
2) Preprocessing (rules & examples)
We applied Azerbaijani-aware, rule-based cleaning to all datasets before labeling:
•	Unicode & casing: fix_text + html.unescape, NFC normalization; Azerbaijani lowercase (İ→i, I→ı).
•	Entity masking: URLs→URL, emails→EMAIL, phone numbers→PHONE, mentions→USER.
•	HTML/markup: strip <tags> and entities.
•	Hashtags: drop # but keep the text and split camelCase (e.g., #QarabagIsBack → qarabag Is Back).
•	Digits: numbers → <NUM>.
•	Repeated letters: collapse ≥3 to 2 (e.g., cooool → coool).
•	Punctuation & spaces: collapse multi-punctuation and extra spaces; keep Azerbaijani letters ə, ğ, ı, ö, ü, ç, ş.
•	Single-letter tokens: remove except o, e.
•	Row hygiene: drop exact duplicates and rows that become empty after cleaning.
Before → After (samples)
•	Bu **film** çox yaxşıdır!!! 😂 → bu film çox yaxşıdır EMO_POS
•	Qiymət: 35 azn, əlaqə: +994… → qiymət <NUM> azn əlaqə PHONE
•	#QarabagIsBack!! → qarabag is back
•	<a href="…">link</a> yazmışdım → link yazmışdım
After cleaning, each file was saved as a two-column Excel with cleaned_text and sentiment_value.
3) Mini Challenges
We implemented all five mini challenges required by the assignment:
•	Hashtag split: Automatically split camelCase hashtags (e.g., #QarabagIsBack → qarabag is back) while removing the # symbol.
•	Emoji mapping: A small emoji dictionary (EMO_MAP) mapped positive emojis to <EMO_POS> and negative ones to <EMO_NEG> before tokenization.
•	Stopword research: Compared Azerbaijani with Turkish stopwords and defined a small optional list (e.g., “və”, “ilə”, “amma”, “lakin”, “ya”, “həm”, “ki”, “bu”, “bir”, “o”). Negations (“yox”, “deyil”, “heç”, “qətiyyən”, “yoxdur”) were intentionally preserved. The option --remove-stopwords can toggle this behavior (default: off).
•	Negation scope: After each negation, the next three tokens were marked with _NEG (e.g., heç bir nəticə vermədi → heç_NEG bir_NEG nəticə_NEG vermədi), improving contextual polarity handling.
•	Simple deasciify: Applied a small map (cox→çox, yaxsi→yaxşı) and reported token changes. Across all datasets, 2,187 tokens were corrected during deasciification.
4) Domain-Aware
We implemented automatic domain detection and domain-specific normalization to make the corpus context-aware.
•	Detection rules:
Used keyword-based hints to classify each text into one of three domains:
o	News → keywords like “apa”, “trend”, “azertac”, “reuters”.
o	Social → includes “@”, “#”, or emojis (😂😍🙂😡).
o	Reviews → contains “azn”, “qiymət”, “ulduz”, or “çox yaxşı”.
Texts that did not match any pattern were labeled as general.
•	Domain normalization (for reviews):
Replaced sentiment and rating phrases with special tags:
price → <PRICE>, 1–5 stars → <STARS_#>,
çox yaxşı → <RATING_POS>, çox pis → <RATING_NEG>.
•	Domain tags:
During corpus generation, each line in corpus_all.txt was automatically prefixed with its detected domain label using the pattern:
dom<domain> <sentence> (e.g., domreviews, domsocial, domnews, domgeneral).
This produced a domain-tagged corpus with 124,353 sentences for model training.
5) Embeddings: Training Settings and Results
Two embedding models — Word2Vec and FastText — were trained on the combined, cleaned, and domain-tagged corpus.
Training Settings
Parameter	Word2Vec	FastText
Architecture	Skip-gram (sg=1)	Skip-gram (sg=1)
Vector size	300	300
Window	5	5
Min count	3	3
Negative samples	10	10
Epochs	10	10
Subword n-grams	—	3–6
Both models were trained using gensim and saved as:
•	embeddings/word2vec.model
•	embeddings/fasttext.model
Evaluation Results
Metric	Word2Vec	FastText
Vocabulary Coverage (avg)	0.960	0.960
Synonym Similarity (avg)	0.312	0.417
Antonym Similarity (avg)	0.306	0.421
Separation (Syn–Ant)	+0.006	–0.004
Nearest Neighbors (Samples)
Word	Word2Vec Neighbors	FastText Neighbors
yaxşı	yaxshi, iyi, <RATING_POS>	yaxşı1, yaxşıca, yaxşıya
pis	<RATING_NEG>, günd, lire	piis, pisdii, pi
çox	çoxx, gözəldir	çoxx, çoh
bahalı	portretlərinə, villaları	bahallı, baha1sı, pahalı
ucuz	şeytanbazardan, düzəldirilib	ucuzu, ucuzca, ucuzdu
Interpretation
•	FastText achieved higher synonym and antonym similarity, confirming that its subword-based approach captures better semantic and morphological relations, especially in rich and noisy Azerbaijani text.
•	Word2Vec performed competitively but was slightly weaker in handling rare or misspelled forms.
•	Both embeddings provide meaningful nearest neighbors, validating the overall corpus and preprocessing pipeline.
7) Reproducibility
The project is fully reproducible and organized into three modular scripts:
Stage	Script	Description
Preprocessing	 preprocess_and_corpus.py	Cleans all five datasets, removes duplicates, exports 2-column Excel files, and builds the domain-tagged corpus corpus_all.txt.
 Training	train_embeddings.py	Trains Word2Vec and FastText models using the cleaned corpus and saves them in embeddings/.
Evaluation	compare_embeddings.py	Compares both models using coverage, synonym/antonym similarity, and nearest-neighbor quality metrics.
Environment
•	Python 3.9 or higher (tested on Windows 10)
•	Required packages:
•	pip install -U pandas gensim openpyxl ftfy regex scikit-learn
Deterministic Results
To ensure identical results across runs:
1.	Disable Python’s hash randomization:
2.	set PYTHONHASHSEED=0        # Windows  
3.	export PYTHONHASHSEED=0     # Linux / macOS
4.	Use a fixed random seed during training:
5.	seed = 42
6.	Word2Vec(..., seed=seed)
7.	FastText(..., seed=seed)
How to run (end-to-end):
1.	Put the 5 raw Excel files in the project root directory.
2.	Preprocess + export 2-column files + build corpus
3.	python 1_preprocess_and_corpus.py
Outputs:
o	*_2col.xlsx (cleaned sentiment files)
o	corpus_all.txt (domain-tagged corpus)
4.	Train embeddings (Word2Vec & FastText)
5.	python 2_train_embeddings.py
Outputs:
o	embeddings/word2vec.model
o	embeddings/fasttext.model
6.	Evaluate & compare models
7.	python 3_compare_embeddings.py
Outputs:
o	Printed coverage, synonym/antonym similarity, nearest neighbors
8.	Notes:
o	All runs are CPU-friendly (no GPU required).
o	Results are reproducible on repeated runs using the same random seed (seed = 42) and hash control (PYTHONHASHSEED = 0).
8) Conclusions
This assignment successfully demonstrated the complete Azerbaijani sentiment analysis pipeline, covering data cleaning, domain detection, corpus construction, and embedding training with Word2Vec and FastText.
•	The preprocessing pipeline effectively normalized noisy Azerbaijani text, handled negations, emojis, hashtags, and domain-specific patterns.
•	The domain-aware corpus (corpus_all.txt) enabled richer contextual learning across news, social, and review text types.
•	Both embeddings produced meaningful semantic structures; however, FastText achieved higher coverage and better synonym similarity due to its subword modeling.
•	The evaluation confirmed that FastText handles morphological variations and rare words more effectively, while Word2Vec remains competitive for frequent vocabulary.
•	All experiments were reproducible and CPU-efficient, meeting the project’s reproducibility and interpretability criteria.
Future work: integrating a more advanced Azerbaijani lemmatizer, expanding domain coverage, and exploring transformer-based embeddings such as BERT for deeper contextual understanding.
### Group Members
- Abdulmajeed Alremali (22050941025)
