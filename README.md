# High-Volume Sentiment Engineering  
## Team Project: IMDb Review Distillation for Automated Sentiment Modeling

**Team Members:** Christian Shannon, Ashley Love, Kirsten Livingston, and Mugtaba Awad  

---

## 1. Project Overview

This project delivers a modular Python pipeline that ingests, remediates, and validates the 50,000-record Stanford IMDb Large Movie Review dataset to create a high-fidelity Golden Asset for sentiment modeling.  
The pipeline replaces manual, ad hoc text cleaning with a reproducible SDLC that consolidates 50,000 decentralized `.txt` files, removes linguistic noise, preserves negation, and verifies data integrity through a set of visual audit gates.

Key dataset notes:  
- 50,000 movie reviews (25,000 positive, 25,000 negative) split into `train/test` and `pos/neg` folders.  
- Source: Stanford IMDb Large Movie Review Dataset (Maas et al.).  
- Unstructured text-only `.txt` files, with HTML artifacts and no unified master table at the outset.

---

## 2. Business Problem & Impact

### Business Problem

High-volume text streams (e.g., consumer reviews) are often scattered across thousands of files and polluted with HTML, filler words, and inconsistent cleaning.  
Manual sentiment analysis does not scale and can silently break semantics—especially when naive stop-word removal drops negation words like “not” or “never,” flipping polarity and misrepresenting consumer voice.

**Professional Hypothesis**  
A modular, automated distillation pipeline that decouples ingestion, normalization, and feature engineering will:

- Provide a single, stable Source of Truth for sentiment audits.  
- Reduce feature explosion while preserving high-intensity sentiment markers and perfect 50/50 class balance.  
- Deliver a Golden Asset that supports both high-accuracy models and transparent, visual validation for stakeholders.

### Business / Analytical Impact

Using the distilled Golden Asset with TF–IDF bi-grams and a tuned Logistic Regression model:

- Balanced dataset: **25,000 positive / 25,000 negative** reviews preserved end-to-end.  
- Final distilled Logistic Regression model (on 25k train / 25k test) achieves:  
  - Accuracy: **≈89%**.  
  - True Positives: **11,146** (≈89.1% of positive reviews).  
  - True Negatives: **11,123** (≈88.9% of negative reviews).  
- Feature space reduced from an initial noisy vocabulary (~100k tokens) to **20,000** highest-intensity terms without sacrificing performance.

This provides a scalable blueprint for transforming noisy, ungoverned text into a clinically styled, model-ready asset suitable for sentiment auditing, CX analytics, and downstream predictive modeling.

---

## 3. Data Pipeline

The pipeline is organized into three core phases, each aligned with a project role and notebook.

### 3.1 Phase 1 – High-Volume Acquisition & Ingestion

Implemented in `01_Data_Wrangling_IMDb_Master-2.ipynb` by **Ashley Love (Data Wrangler)**.  

**Objective**  
Rebuild the ingestion core by converting 50,000 decentralized `.txt` files into a single, labeled master dataframe.

**Key Steps**

- **Directory Mapping & File Traversal**  
  - Use `os` and `glob` to walk the `aclImdb` directory tree for both `train` and `test`, covering `pos` and `neg` subfolders.  
  - Programmatically ingest each review file and capture folder-derived metadata (train/test, sentiment).

- **Label Engineering**  
  - Assign a binary sentiment label based on folder:  
    - `pos` → `sentimenttarget = 1`  
    - `neg` → `sentimenttarget = 0`  
  - Ensures a perfectly balanced 25k / 25k class split.

- **Raw Source-of-Truth Consolidation**  
  - Concatenate all records into a single Pandas dataframe.  
  - Persist as `rawacquisitiondump.csv`, creating a unified Raw Source of Truth for the team.

**Phase 1 Audit**

- Total Source Files Ingested: **50,000**.  
- Acquisition Strategy: automated directory walk (no manual file manipulation).  
- Class Distribution: **25,000 positive / 25,000 negative** (100% label integrity).  
- Status: **PHASE 1 AUDIT SUCCESSFUL**.

---

### 3.2 Phase 2 – Linguistic Normalization

Implemented in `01_Data_Wrangling_IMDb_Master-2.ipynb` (Phase 2) by **Ashley Love (Data Wrangler)**.  

**Objective**  
Normalize raw text into a machine-ready format while preserving the full census and preventing null-related data loss.

**Key Steps**

- **HTML De-noising**  
  - Regex-based stripping of `<br>` tags and similar structural artifacts.

- **Linguistic Standardization**  
  - Lowercase all text to enforce case consistency.  
  - Strip punctuation and normalize whitespace to stabilize tokenization.

- **Null-Safety & Integrity Audit**  
  - Replace non-string or empty entries with a neutral placeholder to avoid dropped rows.  
  - Compute word counts and confirm 0 nulls in the normalized text column.

**Phase 2 Audit**

- Total Records Processed: **50,000**.  
- Average Review Length: **≈228 words**.  
- Null Values in `reviewtextclean`: **0**.  
- Status: **DATA IS NORMALIZED AND READY FOR PHASE 3**.

**Output Asset**

- `cleanreviewsfinal.csv`  
  - Columns: `group` (train/test), `sentimenttarget`, `wordcount`, `reviewtextclean`.

---

### 3.3 Phase 2.5 – Advanced Linguistic Distillation

Implemented in `01_Data_Wrangling_IMDb_Master-2.ipynb` (Phase 2.5) by **Ashley Love (Data Engineer)**.  

**Objective**  
Reduce dimensionality and computational overhead by removing high-frequency, low-sentiment tokens, while preserving sentiment-bearing language for modeling.

**Key Steps**

- **Stop-Word Removal**  
  - Apply a curated stop-word set targeting fillers like “the”, “and”, “is”, etc.  
  - Leave sentiment-rich adjectives, adverbs, and domain-specific terms intact.

- **Distilled Text Column**  
  - Transform `reviewtextclean` into `textdistilled`, providing a compact representation optimized for vectorization.  
  - Retain both columns in the Golden Asset to support raw vs. distilled model comparisons.

- **Efficiency & Signal Audit**  
  - Compute pre- and post-distillation word counts.  
  - Validate that high-intensity sentiment terms remain prominent.

**Phase 2.5 Metrics**

- Total Noise Tokens Removed: **5,381,886**.  
- Avg Words/Review Before: **228.0**.  
- Avg Words/Review After: **≈120.3**.  
- Status: **DATA IS DISTILLED AND OPTIMIZED FOR MACHINE LEARNING**.

**Output Asset (Golden Asset)**

- `imdbmodelreadyfinal.csv`  
  - Columns: `group`, `sentimenttarget`, `wordcount`, `reviewtextclean`, `textdistilled`.

---

## 4. Modeling Approach

Modeling is implemented in `02_Sentiment_Modeling_Christian-2-3.ipynb` by **Christian Shannon (Data Scientist)**.

### 4.1 Problem Framing & Split

- Binary classification: `sentimenttarget` (0 = negative, 1 = positive).  
- Train/test split: **25,000 / 25,000** reviews, maintaining perfect class balance.  
- Experiments conducted on both raw and distilled variants, with the distilled TF–IDF Logistic Regression selected as the primary model.

### 4.2 Feature Engineering: TF–IDF with Bi-grams

- Vectorization on `textdistilled` using `TfidfVectorizer` with:  
  - `ngram_range=(1, 2)` to capture both unigrams and bi-grams (e.g., “not good”).  
  - Max features capped at **20,000** to reduce feature explosion and computational cost.

This design explicitly models local context and preserves negation patterns that are critical for accurate sentiment classification.

### 4.3 Models

- **Baseline Models**  
  - Naive Bayes and Logistic Regression evaluated on both raw and distilled text.  

- **Primary Model: Distilled TF–IDF Logistic Regression**  
  - Trained on 25,000 distilled reviews.  
  - Evaluated on a 25,000-record test set with balanced classes.

### 4.4 Performance & Interpretation

**Performance (Distilled Logistic Regression)**

- Accuracy: **≈89.1%**.  
- True Negatives: **11,123** (≈88.9% of negative reviews).  
- True Positives: **11,146** (≈89.1% of positive reviews).  

These results show that the distilled Golden Asset delivers high, symmetric performance on both classes while operating on a compact 20k-feature space.

**Feature Importance**

- Top positive terms include: `great`, `excellent`, `perfect`, `wonderful`, `amazing`.  
- Top negative terms include: `worst`, `bad`, `awful`, `boring`, `waste`, `terrible`.  

The feature importance table confirms that the model focuses on intuitive sentiment markers, and that the distillation pipeline successfully removed HTML and filler without erasing critical emotional language.

---

## 5. Visual Analytics & Golden Asset Validation

Visual auditing is implemented in `03_Visual_Auditing_Kirsten-4.ipynb` by **Kirsten Livingston (Data Visualizer)**.

### 5.1 Semantic Word Clouds (Task A)

**Goal**  
Visually confirm that the Golden Asset is free from non-semantic noise and dominated by sentiment-bearing tokens.

**Approach**

- Create separate subsets for positive (`sentimenttarget = 1`) and negative (`sentimenttarget = 0`) reviews.  
- Generate word clouds using `textdistilled` for each class.  

**Findings**

- Positive clouds highlight high-intensity terms (e.g., “excellent”, “amazing”, “fun”, “favorite”).  
- Negative clouds focus on terms like “worst”, “awful”, “boring”, “waste”, “disappointing”.  
- No HTML tags or stopwords appear, confirming successful distillation.

---

### 5.2 Comparative Frequency Histograms (Task B)

**Goal**  
Show that distillation reduces linguistic filler while preserving the statistical shape of review lengths.

**Approach**

- Compute length of `reviewtextclean` vs. `textdistilled` (word counts).  
- Plot overlaid histograms and dual-axis views of review length distributions.

**Findings**

- Distilled reviews are shorter on average (~120 vs. 228 words) but follow a similar distribution shape across the 50,000-review census.  
- This indicates that the pipeline removed noise without altering the underlying population structure.

---

### 5.3 Model Performance Visuals (Task C)

**Goal**  
Provide stakeholder-ready visuals confirming that predictive performance remains high on the distilled Golden Asset.

**Approach**

- Plot confusion matrix for the distilled Logistic Regression model.  
- Summarize accuracy, precision, recall, and F1-score.

**Findings**

- The confusion matrix shows balanced true positives and true negatives (~89% each).  
- The distilled model maintains benchmark-level IMDb sentiment performance while using fewer features and a smaller training set.

---

## 6. Technical Stack

Core libraries and tools used across the pipeline:

- **Data & Wrangling**
  - `pandas`, `numpy` for ingestion, consolidation, and transformation.  
  - `os`, `glob`, `re`, `string` for directory traversal and text normalization.

- **NLP & Modeling**
  - `scikit-learn` (`TfidfVectorizer`, `LogisticRegression`, metrics, model selection).  

- **Visualization**
  - `matplotlib`, `seaborn` for histograms and confusion matrix plots.  
  - `wordcloud.WordCloud` for semantic word clouds.

- **Project Structure**
  - Jupyter notebooks for each project phase, plus a white paper (`p3_m3.docx`) documenting the SDLC and stakeholder Q&A.

---

## 7. Ethical & Governance Considerations

Summarized from the white paper and stakeholder appendix:

- **Semantic Integrity & Negation Preservation**  
  - The pipeline design explicitly respects negation (e.g., “not”, “never”) to avoid flipping sentiment and misreporting consumer experiences.  
  - Stop-word lists are curated and reviewed rather than blindly applied.

- **Bias & Class Balance**  
  - The dataset maintains a **50/50** positive/negative split, reducing the risk of biased models that overfit majority sentiment.  

- **Transparency & Reproducibility**  
  - Each phase (ingestion, normalization, distillation, modeling, visualization) is modular, version-controlled, and auditable.  
  - Visual audit gates (word clouds, histograms, confusion matrix) are required prior to any production deployment.

- **Responsible Use**  
  - The Golden Asset and models are intended as decision-support tools for sentiment auditing and CX analytics, not as opaque black boxes.  
  - Stakeholders are encouraged to pair automated insights with qualitative review sampling.

---

## 8. Repository Structure

A representative layout based on the project files:

```text
.
├─ README.md                                     # This file
├─ data/
│  └─ Data Source/                               # Stanford IMDb directory tree (external source)
├─ notebooks/
│  ├─ 01_Data_Wrangling_IMDb_Master.ipynb        # Ingestion, normalization, distillation
│  ├─ 02_Sentiment_Modeling_Christian 2.ipynb    # TF–IDF + Logistic Regression
│  └─ 03_Visual_Auditing_Kirsten.ipynb           # Word clouds, histograms, confusion matrix
└─ reports/
   └─ High Volume Sentiment Engineering.pdf      # White paper and stakeholder documentation
