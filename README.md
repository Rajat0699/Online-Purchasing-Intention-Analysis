# Online Purchasing Intention Analysis

Predicting online shopper behavior and purchase intent from web-session data using exploratory analysis, unsupervised clustering, and supervised machine-learning models in R.

## Overview

E-commerce platforms capture a rich trail of behavioral signals every time a visitor browses a site — pages viewed, time spent, bounce and exit patterns, and the commercial value of the pages they land on. This project turns those raw session signals into a practical question for any online retailer: **which visits are likely to end in a purchase, and what distinguishes a buyer from a browser?**

Working with the UCI *Online Shoppers Purchasing Intention* dataset (12,330 sessions, 18 attributes), the project builds an end-to-end analytical pipeline: from data profiling and visualization, through customer segmentation, to predictive models that flag revenue-generating sessions. The goal is not just accuracy for its own sake, but actionable insight — identifying the levers a business can pull to reduce site abandonment and lift conversion.

## Dataset

- **Source:** UCI Machine Learning Repository — Online Shoppers Purchasing Intention Dataset
- **Size:** 12,330 visitor sessions × 18 variables
- **Target:** `Revenue` (Boolean) — whether the session ended in a completed transaction
- **Class balance:** highly imbalanced (~10,422 non-purchases vs. ~1,908 purchases)
- **Key numeric features:** `ProductRelated`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay`, and administrative/informational page counts
- **Key categorical features:** `Month`, `VisitorType`, `Weekend`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`
- **Data quality:** no missing values — a zero represents genuine absence of interaction rather than missing data.

## Approach

The analysis was carried out in **R**, and is organized into three phases.

### 1. Exploratory Data Analysis

Statistical profiling (`str`, `summary`) and visualization established the shape of the data: box plots confirmed no significant outliers, seasonal trends showed elevated activity around holiday periods and in May, and density comparisons of buyers vs. non-buyers surfaced the behavioral traits that later drove the models. `PageValues`, `BounceRates`, and time-on-page consistently separated the two groups.

### 2. Unsupervised Segmentation

- **K-Means Clustering** grouped sessions by behavioral similarity and was evaluated against actual revenue. It cleanly isolated disinterested visitors (high bounce rate, short dwell time) but struggled to recover the smaller, valuable buyer segment — a direct consequence of the class imbalance.
- **Hierarchical Clustering** refined this picture, splitting the revenue-generating cluster into meaningful sub-groups with distinct page-value profiles, useful for isolating pockets of high-potential customers.

### 3. Supervised Prediction

Three complementary models were trained to predict `Revenue`:

- **Decision Tree Classification** — tuned via cross-validation (complexity parameter 0.055, 13 splits) and pruned to **~90.8% accuracy**, with `PageValues` selected for the root split.
- **Logistic Regression** — a binomial model with backward selection identifying `PageValues`, `ExitRates`, `ProductRelated_Duration`, and several month indicators as the most significant predictors.
- **Neural Network** — a 3-hidden-node network on `PageValues`, `BounceRates`, `Weekend`, and `ExitRates`, reaching **~89% accuracy** (sensitivity 0.95, balanced accuracy 0.76).

## Key Findings

- **A small set of signals carries most of the predictive power.** `PageValues` (commercial value of visited pages), `BounceRates`/`ExitRates`, and product-page engagement dominate across every model.
- **The tree-based classifier is the standout**, delivering ~90% accuracy while remaining interpretable — an important property when the output has to justify a business decision.
- **Class imbalance is the central challenge.** Models excel at identifying non-buyers but must be tuned carefully to catch the rarer, high-value purchase sessions.
- **The insight translates to action:** conversion can be improved by raising page value, reducing bounce rate, and giving attention to how administrative-type pages are presented — and predicted high-intent visitors can be targeted with promotional offers and ads, concentrating marketing spend where it converts.

## Tech Stack

- **Language:** R
- **Techniques:** EDA & data visualization, K-Means & Hierarchical clustering, Decision Tree classification, Logistic Regression, Neural Networks, cross-validation, confusion-matrix evaluation

## Repository Contents

- `Final_Report.pdf` — the complete write-up: methodology, diagrams, model outputs, and findings.

## Collaboration

This was a **collaborative, five-member team project**, delivered as a group under a shared analytical framework. The work spanned data preparation, exploratory analysis, model development, and reporting, with contributions integrated into the single unified study documented here.

## References

1. Sakar, C.O., Polat, S.O., Katircioglu, M., et al. *Neural Computing & Applications* (2018).
2. UCI Machine Learning Repository — Online Shoppers Purchasing Intention Dataset.
