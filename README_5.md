# 📊 EDA & Predictive Modeling — Academic Performance of Irish Students

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13-4C72B0?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-green?style=for-the-badge)

**A rigorous, end-to-end data science pipeline applied to the UCI Student Performance Dataset**
*Covering EDA · Feature Engineering · Visualisation · Regression · Classification*

---

> *"The goal is to turn data into information, and information into insight."*
> — Carly Fiorina

---

**Author:** Aman Aaryan &nbsp;
**Institution:** KIIT University, Department of Information Technology &nbsp;|&nbsp; **Date:** March 2026

</div>

---

## 📌 Table of Contents

| # | Section |
|---|---------|
| 1 | [Project Overview](#-project-overview) |
| 2 | [Pipeline Architecture](#-pipeline-architecture) |
| 3 | [Dataset Description](#-dataset-description) |
| 4 | [Feature Dictionary](#-feature-dictionary) |
| 5 | [Phase Breakdown](#-phase-breakdown) |
| 6 | [Visualisations](#-visualisations) |
| 7 | [Model Results](#-model-results) |
| 8 | [Key Findings & Insights](#-key-findings--insights) |
| 9 | [Tech Stack](#-tech-stack) |
| 10 | [Project Structure](#-project-structure) |
| 11 | [How to Run](#-how-to-run) |
| 12 | [Limitations & Future Work](#-limitations--future-work) |
| 13 | [References](#-references) |

---

## 🎯 Project Overview

This project applies a **complete 7-phase Exploratory Data Analysis (EDA) pipeline** to the UCI Machine Learning Repository's *Student Performance Dataset* — a real-world educational dataset collected from two secondary schools in Portugal during the 2005–2006 academic year.

The study investigates **what factors drive student academic success** across Mathematics and Portuguese Language, combining rigorous statistical analysis with machine learning predictive modeling.

### 🧭 Core Objectives

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PROJECT OBJECTIVES                           │
├──────┬──────────────────────────────────────────────────────────────┤
│  01  │  Master two real-world CSV datasets through systematic EDA   │
│  02  │  Engineer meaningful features from raw demographic data      │
│  03  │  Construct a multi-phase visual narrative (9 figure sets)    │
│  04  │  Merge datasets via composite key for cross-subject analysis │
│  05  │  Quantify lifestyle (alcohol, romance) impacts on grades     │
│  06  │  Train & evaluate Linear + Logistic Regression models        │
│  07  │  Deliver reproducible notebook + executive dashboard         │
└──────┴──────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Pipeline Architecture

The project follows a structured 7-phase pipeline — each phase building upon the previous:

```
╔══════════════════════════════════════════════════════════════════════════╗
║                    END-TO-END EDA PIPELINE                               ║
╠══════════╦═══════════════════════════════════╦══════════════════════════╣
║  PHASE   ║         OPERATION                 ║       OUTPUT             ║
╠══════════╬═══════════════════════════════════╬══════════════════════════╣
║  01      ║  Data Ingestion                   ║  df_mat, df_por loaded   ║
║          ║  Load CSVs, inspect structure     ║  395 + 649 records       ║
╠══════════╬═══════════════════════════════════╬══════════════════════════╣
║  02      ║  Preprocessing & Feature Eng.     ║  3 new features added    ║
║          ║  Clean, engineer, unify           ║  df_all (1,044 rows)     ║
╠══════════╬═══════════════════════════════════╬══════════════════════════╣
║  03      ║  Univariate Analysis              ║  9-panel distribution    ║
║          ║  Distributions of all variables   ║  figure                  ║
╠══════════╬═══════════════════════════════════╬══════════════════════════╣
║  04a     ║  Bivariate Analysis               ║  6-panel relationship    ║
║          ║  Key predictors vs G3             ║  figure                  ║
╠══════════╬═══════════════════════════════════╬══════════════════════════╣
║  04b     ║  Correlation Heatmap              ║  Full Pearson matrix     ║
║          ║  Pearson r for all numeric pairs  ║  lower-triangle view     ║
╠══════════╬═══════════════════════════════════╬══════════════════════════╣
║  04c     ║  Pairplot                         ║  5×5 scatterplot matrix  ║
║          ║  Multi-variable overview          ║  coloured by grade label ║
╠══════════╬═══════════════════════════════════╬══════════════════════════╣
║  05      ║  Relational Merging               ║  df_merged: dual-        ║
║          ║  Inner join on 13-col key         ║  enrolled students       ║
╠══════════╬═══════════════════════════════════╬══════════════════════════╣
║  06      ║  Social & Lifestyle Analysis      ║  6-panel lifestyle       ║
║          ║  Alcohol, romance, aspirations    ║  figure                  ║
╠══════════╬═══════════════════════════════════╬══════════════════════════╣
║  07      ║  Executive Dashboard +            ║  Regression R²=0.771     ║
║          ║  Predictive Modeling              ║  Classification F1=0.891 ║
╚══════════╩═══════════════════════════════════╩══════════════════════════╝
```

### Data Flow Diagram

```
  [student-mat.csv]          [student-por.csv]
        │                           │
        ▼                           ▼
   df_mat (395×33)           df_por (649×33)
        │                           │
        ├──── Feature Engineering ──┤
        │    ┌────────────────────┐ │
        │    │ + grade_label      │ │
        │    │ + avg_parent_edu   │ │
        │    │ + grade_improvement│ │
        │    └────────────────────┘ │
        │                           │
        ├──────── pd.concat() ──────┤
        │                           │
        ▼                           ▼
   df_all (1044×37)       df_merged (inner join
        │                  on 13-col composite key)
        │                           │
        ▼                           ▼
  [EDA Phases 3–6]       [Cross-subject Analysis]
        │
        ▼
  [Regression Model]  →  MAE=1.394 | RMSE=2.165 | R²=0.771
  [Classification]    →  Accuracy=86.1% | F1=0.891
        │
        ▼
  [Executive Dashboard]
```

---

## 📂 Dataset Description

| Property | Mathematics | Portuguese |
|----------|-------------|------------|
| **File** | `student-mat.csv` | `student-por.csv` |
| **Students** | 395 | 649 |
| **Features** | 33 columns | 33 columns |
| **Delimiter** | Semicolon `;` | Semicolon `;` |
| **Schools** | GP, MS | GP, MS |
| **Grade Range** | 0 – 20 | 0 – 20 |
| **Target Variable** | G3 (Final Grade) | G3 (Final Grade) |
| **Missing Values** | ✅ None | ✅ None |
| **Duplicates** | ✅ None | ✅ None |

### Target Variable — G3 Grade Distribution

```
Grade Distribution Summary:

                  MATHEMATICS              PORTUGUESE
  Mean    ──────── 10.42          vs       11.91 ─────────
  Median  ──────── 11.00          vs       12.00 ─────────
  Std Dev ──────── 4.58           vs        3.23 ─────────
  Fail %  ──────── ~33%           vs        ~16% ─────────

  Grade Categories:
  ┌──────────────┬──────────────┬──────────────────────────┐
  │  Range       │  Label       │  Interpretation          │
  ├──────────────┼──────────────┼──────────────────────────┤
  │   0 –  9     │  ❌ Fail     │  Below minimum threshold │
  │  10 – 12     │  ✔️  Pass    │  Meets minimum standard  │
  │  13 – 15     │  ⭐ Good     │  Above average           │
  │  16 – 20     │  🏆 Excellent│  Outstanding achievement │
  └──────────────┴──────────────┴──────────────────────────┘

  Overall distribution (both subjects combined):
  Fail      ██████████░░░░░░░░░░  22.0%
  Pass      ███████████████░░░░░  39.0%
  Good      ██████████░░░░░░░░░░  27.3%
  Excellent ████░░░░░░░░░░░░░░░░  11.7%
```

### Dataset Source

> **Cortez, P. & Silva, A.** (2008). *Using Data Mining to Predict Secondary School Student Performance.*
> Proceedings of 5th FUBUTEC Conference, Porto, Portugal.
> 📎 [UCI ML Repository — Student Performance](https://archive.ics.uci.edu/ml/datasets/student+performance)

---

## 📋 Feature Dictionary

### Demographic Features

| Feature | Type | Scale | Description |
|---------|------|-------|-------------|
| `school` | Binary | GP / MS | Gabriel Pereira or Mousinho da Silveira |
| `sex` | Binary | F / M | Student gender |
| `age` | Numeric | 15 – 22 | Student age |
| `address` | Binary | U / R | Urban or Rural home address |
| `famsize` | Binary | LE3 / GT3 | Family size ≤3 or >3 |
| `Pstatus` | Binary | T / A | Parents living together or apart |

### Socio-Academic Features

| Feature | Type | Scale | Description |
|---------|------|-------|-------------|
| `Medu` | Ordinal | 0 – 4 | Mother's education level |
| `Fedu` | Ordinal | 0 – 4 | Father's education level |
| `Mjob` | Categorical | 5 values | Mother's occupation |
| `Fjob` | Categorical | 5 values | Father's occupation |
| `reason` | Categorical | 4 values | Reason for school choice |
| `guardian` | Categorical | 3 values | Legal guardian |
| `traveltime` | Ordinal | 1 – 4 | Home-to-school travel time |
| `studytime` | Ordinal | 1 – 4 | Weekly study hours |
| `failures` | Numeric | 0 – 4 | Number of past class failures |

### Support & Activity Features

| Feature | Type | Description |
|---------|------|-------------|
| `schoolsup` | Binary | Extra school educational support |
| `famsup` | Binary | Family educational support |
| `paid` | Binary | Extra paid tutoring classes |
| `activities` | Binary | Extra-curricular activities |
| `nursery` | Binary | Attended nursery school |
| `higher` | Binary | Aspires to pursue higher education |
| `internet` | Binary | Internet access at home |
| `romantic` | Binary | Currently in a romantic relationship |

### Social & Lifestyle Features

| Feature | Type | Scale | Description |
|---------|------|-------|-------------|
| `famrel` | Ordinal | 1 – 5 | Family relationship quality |
| `freetime` | Ordinal | 1 – 5 | Free time after school |
| `goout` | Ordinal | 1 – 5 | Going out with friends frequency |
| `Dalc` | Ordinal | 1 – 5 | Weekday alcohol consumption |
| `Walc` | Ordinal | 1 – 5 | Weekend alcohol consumption |
| `health` | Ordinal | 1 – 5 | Current health status |
| `absences` | Numeric | 0 – 93 | Number of school absences |

### Grade Variables

| Feature | Type | Scale | Description |
|---------|------|-------|-------------|
| `G1` | Numeric | 0 – 20 | First period grade |
| `G2` | Numeric | 0 – 20 | Second period grade |
| `G3` | Numeric | 0 – 20 | **Final grade — TARGET VARIABLE** |

### 🔧 Engineered Features

| Feature | Formula | Purpose |
|---------|---------|---------|
| `grade_label` | Binned G3 → {Fail, Pass, Good, Excellent} | Classification & visual grouping |
| `avg_parent_edu` | `(Medu + Fedu) / 2` | Single household education index |
| `grade_improvement` | `G3 − G1` | Academic trajectory over the year |

---

## 🔬 Phase Breakdown

### Phase 1 — Data Ingestion

```python
# Critical: Portuguese & Math CSVs use semicolons, NOT commas
df_mat = pd.read_csv('student-mat.csv', sep=';')   # 395 × 33
df_por = pd.read_csv('student-por.csv', sep=';')   # 649 × 33
```

Key structural inspection methods applied:

| Method | Purpose |
|--------|---------|
| `df.head()` | Visual sanity check — correct parsing |
| `df.info()` | Column names, types, non-null counts |
| `df.describe()` | Descriptive stats — detect outliers |
| `df.isnull().sum()` | Missing value audit |
| `df.duplicated().sum()` | Duplicate detection |

---

### Phase 2 — Preprocessing & Feature Engineering

```
Preprocessing Checklist:
  ✅ Missing Values  → 0 in Math | 0 in Portuguese
  ✅ Duplicates      → 0 in Math | 0 in Portuguese
  ✅ grade_label     → Binned G3 into 4 performance categories
  ✅ avg_parent_edu  → (Medu + Fedu) / 2  [range: 0.0 – 4.0]
  ✅ grade_improvement → G3 − G1          [trajectory metric]
  ✅ Unified df_all  → pd.concat([df_mat, df_por]) → 1044 × 37
```

---

### Phase 3 — Univariate Distribution Analysis

Examined 9 individual variable distributions:

```
  Variables Analysed:
  ┌─────────────────────┬──────────────────────────────────────────┐
  │ G3 (Math)           │ Bimodal — spike at 0 (non-sitters)       │
  │ G3 (Portuguese)     │ Near-normal, mean=11.9, tighter spread   │
  │ studytime           │ Right-skewed — majority study 2–5 hrs    │
  │ absences            │ Strongly right-skewed, max=75            │
  │ age                 │ Concentrated 15–18, rare >19             │
  │ failures            │ 73% have zero failures                   │
  │ grade_improvement   │ Centred near 0; heavier left tail in Math│
  │ sex                 │ 591 Female vs 453 Male (combined)        │
  │ grade_label         │ Pass=39% > Fail=22% > Good=27% > Exc=12%│
  └─────────────────────┴──────────────────────────────────────────┘
```

---

### Phase 4 — Bivariate & Multivariate Analysis

#### 4a — Key Predictors vs Final Grade

| Predictor | Relationship | Insight |
|-----------|-------------|---------|
| **Study Time** | Positive, non-linear | Studytime=3 (5–10h) yields best median grades |
| **Past Failures** | Strong negative | 0 failures → median ~11; 3 failures → median <6 |
| **Parental Education** | Positive linear | Portuguese shows steeper gradient than Math |
| **Gender** | Subject-dependent | Males outperform in Math; Females in Portuguese |
| **Address** | Urban advantage | Urban students score ~1–2 pts higher consistently |
| **Internet Access** | Positive | ~0.8 pt advantage in Portuguese; ~0.5 in Math |

#### 4b — Pearson Correlation Heatmap Findings

```
  Top Correlations with G3 (Mathematics):

  G2          ████████████████████████████████████  r = +0.90  ⬆ Very Strong
  G1          ████████████████████████████████       r = +0.80  ⬆ Very Strong
  grade_impr  ████████████████████████████           r = +0.70  ⬆ Strong
  G2 ↔ G1     █████████████████████████████████████ r = +0.85  ⬆ Very Strong
  failures    ██████████████░░░░░░░░░░░░░░░░░░░░░░  r = −0.36  ⬇ Moderate
  studytime   ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  r = +0.25  ⬆ Weak
  Dalc        ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  r = −0.22  ⬇ Weak
  Dalc ↔ Walc ██████████████████████████            r = +0.65  ⬆ Strong
  goout ↔ Walc████████████████                      r = +0.42  ⬆ Moderate

  Interpretation Scale:
  |r| 0.00–0.19 → Negligible  |  0.20–0.39 → Weak
  |r| 0.40–0.59 → Moderate    |  0.60–0.79 → Strong
  |r| 0.80–1.00 → Very Strong
```

#### 4c — Pairplot Insight

- **Strongest visual class separation** occurs along G1, G2, G3 axes
- Excellent (green) students cleanly cluster at high grade values
- Fail (red) students cluster at low values with some overlap into Pass
- `studytime` and `absences` show weaker but visible separation

---

### Phase 5 — Relational Merging

```python
# 13-column composite key for identifying dual-enrolled students
merge_keys = ['school','sex','age','address','famsize','Pstatus',
              'Medu','Fedu','Mjob','Fjob','reason','nursery','internet']

df_merged = pd.merge(df_mat, df_por, on=merge_keys, suffixes=('_math','_por'))
```

| Finding | Value |
|---------|-------|
| Cross-subject correlation (Math G3 vs Por G3) | r ≈ 0.480 |
| Regression equation | y = 0.30x + 9.38 |
| Grade differential (Por − Math) distribution | Centred slightly above 0 |
| Interpretation | Portuguese grades marginally higher; ability is generalised |

---

### Phase 6 — Social & Lifestyle Analysis

```
  ALCOHOL CONSUMPTION IMPACT (Mathematics — Weekday Dalc):

  Dalc=1 (Very Low)  → Median G3 ≈ 11  ████████████████████  ✅
  Dalc=2             → Median G3 ≈ 10  ██████████████████░░  ⚠️
  Dalc=3             → Median G3 ≈  9  ████████████████░░░░  ⚠️
  Dalc=4             → Median G3 ≈  8  ██████████████░░░░░░  ❌
  Dalc=5 (Very High) → Median G3 ≈  8  ██████████████░░░░░░  ❌
                                         Below Pass Threshold ↑

  KEY INSIGHT: Dalc ≥ 3 → Median grade falls BELOW pass threshold
```

| Social Variable | Effect on G3 | Direction |
|----------------|-------------|-----------|
| Weekday Alcohol (Dalc) | Strong | ⬇ Negative |
| Weekend Alcohol (Walc) | Moderate | ⬇ Negative |
| Going Out (goout) | Weak | ⬇ Negative |
| Romantic Relationship | Mild (~0.5–1.0 pts) | ⬇ Negative |
| Higher Education Aspiration | Very Strong | ⬆ Positive |
| Family Educational Support | Counterintuitive ⚠️ | Confounded |

> ⚠️ **Confounding Alert:** Family support appears negatively correlated due to selection bias — support is given to already-struggling students. This is **not** evidence that support hurts grades.

---

## 🖼️ Visualisations

| Figure | Title | Key Insight |
|--------|-------|-------------|
| `phase3_univariate.png` | Univariate Distribution Analysis | G3 bimodal in Math; right-skewed absences |
| `phase4a_bivariate.png` | Bivariate — Key Predictors vs G3 | Failures most damaging; urban advantage visible |
| `phase4b_heatmap.png` | Pearson Correlation Heatmap | G1/G2/G3 r>0.80; multicollinearity present |
| `phase4c_pairplot.png` | Pairplot by Grade Label | Clear G1/G2/G3 cluster separation |
| `phase5_merged.png` | Cross-Subject Comparative Analysis | r=0.48 between Math & Portuguese G3 |
| `phase6_social.png` | Social & Lifestyle Factors | Alcohol & aspirations dominate lifestyle effects |
| `regression_results.png` | Regression Diagnostics | R²=0.771; residuals near-normally distributed |
| `classification_results.png` | Classification — Confusion Matrix | F1=0.891; high precision for Pass class |
| `executive_dashboard.png` | Executive Summary Dashboard | Full project summary in one figure |

---

## 📈 Model Results

### Regression Model — Predicting G3 (Linear Regression)

```
  Features Used: G1, G2, studytime, failures, absences,
                 avg_parent_edu, Dalc, Walc, goout

  Train / Test Split: 80% / 20%  |  random_state = 42
```

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MAE** | `1.3941` | Predictions off by ~1.4 grade points on average |
| **MSE** | `4.6883` | Squared error; sensitive to large deviations |
| **RMSE** | `2.1652` | ~68% of predictions within ±2.17 grade points |
| **R²** | `0.7714` | Model explains **77.1%** of grade variance |
| **CV R² (5-fold)** | `0.8030` | Robust generalisation confirmed |

#### Feature Coefficients (Regression)

```
  G2              ████████████████████████████████████████  +1.00  ⬆ Strongest
  G1              █████████████░░░░░░░░░░░░░░░░░░░░░░░░░░  +0.22  ⬆ Positive
  goout           ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  +0.18  ⬆ Positive
  absences        ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  +0.07  ⬆ Slight
  Walc            █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  +0.05  ⬆ Slight
  studytime       ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  −0.10  ⬇ Negative*
  avg_parent_edu  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  −0.13  ⬇ Negative*
  Dalc            ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  −0.16  ⬇ Negative
  failures        ████████████████░░░░░░░░░░░░░░░░░░░░░░░  −0.41  ⬇ Strongest

  * Sign affected by multicollinearity with G1/G2 — interpret with caution
```

---

### Classification Model — Pass/Fail (Logistic Regression)

```
  Binary Target: Pass (G3 ≥ 10) = 1  |  Fail (G3 < 10) = 0
  Preprocessing: StandardScaler applied before model fitting
```

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Accuracy** | `0.861` | 86.1% of all predictions correct |
| **Precision** | `0.938` | 93.8% of predicted-Pass students actually passed |
| **Recall** | `0.849` | 84.9% of actual-Pass students correctly identified |
| **F1-Score** | `0.891` | Balanced precision-recall trade-off |

#### Confusion Matrix Breakdown

```
                    Predicted FAIL    Predicted PASS
                  ┌────────────────┬────────────────┐
  Actual FAIL     │   TN = 23      │   FP = 3       │
                  ├────────────────┼────────────────┤
  Actual PASS     │   FN = 8       │   TP = 45      │
                  └────────────────┴────────────────┘

  ✅ True Negatives  (23): Correctly identified failing students
  ✅ True Positives  (45): Correctly identified passing students
  ⚠️  False Positives ( 3): Failing students wrongly predicted as Pass
  ⚠️  False Negatives ( 8): Passing students missed by model
```

---

## 💡 Key Findings & Insights

```
┌─────┬──────────────────────────────────────────────────────────────────────┐
│ 01  │ PRIOR GRADES DOMINATE                                                │
│     │ G1 (r=0.80) and G2 (r=0.90) are the strongest predictors of G3.    │
│     │ → Implication: Intervene EARLY — waiting until year-end is too late │
├─────┼──────────────────────────────────────────────────────────────────────┤
│ 02  │ PORTUGUESE IS EASIER THAN MATHEMATICS                                │
│     │ Mean G3: 11.91 (Por) vs 10.42 (Math) | Fail%: 16% vs 33%           │
│     │ → Math has larger variance — more high achievers AND more failures   │
├─────┼──────────────────────────────────────────────────────────────────────┤
│ 03  │ ALCOHOL MEASURABLY HARMS GRADES                                      │
│     │ Dalc ≥ 3 → median grade falls below pass threshold in Mathematics   │
│     │ → Weekday drinking more damaging than weekend consumption            │
├─────┼──────────────────────────────────────────────────────────────────────┤
│ 04  │ ASPIRATION IS A POWERFUL PREDICTOR                                   │
│     │ Students aspiring to higher education score 2–3 pts higher on avg   │
│     │ → Likely captures intrinsic motivation — a causal driver            │
├─────┼──────────────────────────────────────────────────────────────────────┤
│ 05  │ URBAN ADVANTAGE IS COMPOSITE                                         │
│     │ Urban students consistently outperform rural peers                  │
│     │ → Effect of internet, tutoring access, shorter commute combined      │
├─────┼──────────────────────────────────────────────────────────────────────┤
│ 06  │ FAMILY SUPPORT PARADOX                                               │
│     │ famsup shows weak negative raw correlation — a textbook confound    │
│     │ → Support given to struggling students creates selection bias        │
├─────┼──────────────────────────────────────────────────────────────────────┤
│ 07  │ ACADEMIC ABILITY IS GENERALISED                                      │
│     │ Math G3 vs Portuguese G3 Pearson r ≈ 0.48 for dual-enrolled        │
│     │ → ~42% of variance shared — subject-specific factors explain rest   │
├─────┼──────────────────────────────────────────────────────────────────────┤
│ 08  │ FAILURES COMPOUND NEGATIVELY                                         │
│     │ 0 failures → median ~11 | 1 failure → ~8 | 2+ failures → ~6        │
│     │ → Each failure triggers a negative feedback loop of disengagement   │
└─────┴──────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Library | Version | Role |
|---------|---------|------|
| ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) | 3.10+ | Core language |
| **pandas** | 2.x | Data manipulation, DataFrame operations |
| **numpy** | 2.x | Numerical computing, array operations |
| **matplotlib** | 3.x | Low-level plot construction |
| **seaborn** | 0.13 | Statistical visualisation (heatmaps, violins, pairplots) |
| **scikit-learn** | 1.x | ML models, metrics, train/test split, StandardScaler |
| **jupyter** | 1.x | Interactive notebook environment |
| **nbconvert** | latest | Notebook → HTML / PDF export |

---

## 📁 Project Structure

```
Student_EDA_Project/
│
├── 📓 student_performance_eda.ipynb       ← Main analysis notebook (all 7 phases)
├── 📄 student_performance_eda.html        ← Exported HTML version
│
├── 📊 Data/
│   ├── student-mat.csv                    ← Mathematics dataset (395 students)
│   └── student-por.csv                    ← Portuguese dataset (649 students)
│
├── 🖼️ Figures/
│   ├── phase3_univariate.png              ← Phase 3: Univariate distributions
│   ├── phase4a_bivariate.png             ← Phase 4a: Key predictor relationships
│   ├── phase4b_heatmap.png               ← Phase 4b: Pearson correlation heatmap
│   ├── phase4c_pairplot.png              ← Phase 4c: Multi-variable pairplot
│   ├── phase5_merged.png                 ← Phase 5: Cross-subject comparison
│   ├── phase6_social.png                 ← Phase 6: Lifestyle factor analysis
│   ├── regression_results.png            ← Regression diagnostics & coefficients
│   ├── classification_results.png        ← Confusion matrix & classification metrics
│   └── executive_dashboard.png          ← Phase 7: Full executive summary
│
├── 📝 Student_Performance_EDA_Complete_Guide.pdf  ← Full written report (37 pages)
└── 📖 README.md                          ← This file
```

---

## ▶️ How to Run

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter nbconvert
```

### Clone & Launch

```bash
# 1. Clone the repository
git clone https://github.com/thestethoguy/student-performance-eda.git
cd student-performance-eda

# 2. Launch Jupyter Notebook
jupyter notebook student_performance_eda.ipynb

# 3. Run all cells sequentially (Kernel → Restart & Run All)
```

### Export Options

```bash
# Export to HTML
python -m nbconvert --to html student_performance_eda.ipynb

# Export to PDF (requires chromium)
pip install nbconvert[webpdf] playwright
playwright install chromium
python -m nbconvert --to webpdf student_performance_eda.ipynb
```

---

## ⚠️ Limitations & Future Work

### Current Limitations

| Limitation | Impact | Mitigation Path |
|------------|--------|----------------|
| Small sample (395/649 students) | Limited generalisation | Multi-school, multi-country data |
| Single academic year snapshot | No longitudinal trends | Cohort tracking study |
| Self-reported lifestyle data | Potential under-reporting bias | Objective behavioural measures |
| Portuguese school context only | Cultural non-generalisability | Cross-national replication |
| No direct SES index | SES proxied by address/parental edu | Direct income/SES measurement |

### Future Extensions

```
  Planned Improvements:
  ┌─────────────────────────┬────────────────────────────────────────────────┐
  │ Technique               │ Expected Benefit                               │
  ├─────────────────────────┼────────────────────────────────────────────────┤
  │ Random Forest / XGBoost │ Capture non-linear feature interactions        │
  │ PCA                     │ Reduce 33 features, enable 2D visualisation    │
  │ SHAP Values             │ Per-instance explainability beyond coefficients│
  │ GridSearchCV            │ Optimise Logistic Regression regularisation C  │
  │ SMOTE                   │ Handle Fail class imbalance                    │
  │ NLP (if text data)      │ Sentiment/topic extraction from responses      │
  └─────────────────────────┴────────────────────────────────────────────────┘
```

---

## 📚 References

| Source | Citation |
|--------|---------|
| **Primary Dataset** | Cortez, P. & Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance.* FUBUTEC 2008, Porto. |
| **pandas** | McKinney, W. (2010). *Data Structures for Statistical Computing in Python.* SciPy 2010. |
| **numpy** | Harris, C.R. et al. (2020). *Array programming with NumPy.* Nature, 585, 357–362. |
| **matplotlib** | Hunter, J.D. (2007). *Matplotlib: A 2D Graphics Environment.* Computing in Science & Engineering, 9(3), 90–95. |
| **seaborn** | Waskom, M. (2021). *seaborn: statistical data visualization.* JOSS, 6(60), 3021. |
| **scikit-learn** | Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR, 12, 2825–2830. |
| **EDA Theory** | Tukey, J.W. (1977). *Exploratory Data Analysis.* Addison-Wesley. |
| **Logistic Regression** | Hosmer, D.W. & Lemeshow, S. (2000). *Applied Logistic Regression (2nd ed.).* Wiley. |

---

<div align="center">

---

Made with 🧠 + ☕ by **Aman Aaryan**
KIIT University · Department of Computer Science 

*If you found this project helpful, consider giving it a ⭐ on GitHub!*

</div>
