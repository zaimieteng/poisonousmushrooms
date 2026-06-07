# Mushroom Toxicity Prediction

This is a machine learning classification project to predict whether a mushroom
is edible or poisonous based on physical characteristics, using a
kaggle dataset containing 3.1 million records.

## Problem Statement
Mushroom misidentification can have fatal consequences. This project
builds a classifier to predict toxicity from physical features, and
identifies key characteristics associated with poisonous mushrooms.

## Key Findings
- Random Forest performed best: Accuracy 99.1%, AUC 0.9956, MCC 0.9826
- Poisonous mushrooms tend to have smaller caps and shorter, narrower stems
- Key warning signs: pink cap colour, attached gills, presence of rings
- Single characteristics alone are insufficient — toxicity requires
  considering multiple features simultaneously

## Models Compared
| Model | Accuracy | AUC | MCC |
|---|---|---|---|
| Random Forest | 0.9914 | 0.9956 | 0.9826 |
| Decision Tree | 0.9924 | 0.9787 | 0.9570 |
| XGBoost | 0.9764 | 0.9761 | 0.9525 |
| k-NN | 0.9753 | 0.9850 | 0.9502 |
| Logistic Regression | 0.7980 | 0.8552 | 0.5987 |
| Naive Bayes | 0.6873 | 0.8040 | 0.4271 |

## Methodology
1. Exploratory Data Analysis — handled skewed data, outliers,
   infrequent categories, and missing values
2. Feature Selection — XGBoost-based selection on 70/30 train-test split
3. Model Development — compared 4 non-ensemble (Decision tree, naive bayes, logistic regression, k-NN) and 2 ensemble models (XGBoost, random forest)

## Tools Used
Python — scikit-learn, XGBoost, pandas, matplotlib

## Data Source
Kaggle Playground Series S4E8 — combines UCI mushroom data with
synthetic data across 22 features and 3.1 million rows.
[Dataset link](https://www.kaggle.com/competitions/playground-series-s4e8)
