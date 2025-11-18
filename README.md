This project contains a clean and compact Machine Learning pipeline for the Kaggle Titanic competition.
The full solution is written in under 250 lines, including feature engineering, preprocessing, and a LightGBM / Random Forest model.

The main goal of this project is to build a strong and simple model that anyone can understand and run without errors.

The workflow includes:

extracting useful features like titles, family size, ticket prefixes, cabin decks

handling missing values correctly

scaling numeric features

one-hot encoding for all categorical features

training a LightGBM model (with a RandomForest fallback)

generating a submission file for Kaggle

Everything is written in a single script, easy to read, and easy to modify for beginners and intermediate learners.

This is my first Kaggle ML project and I wanted to keep it simple, clean, and educational
est score 0.75-0.80
Features

Works with both LightGBM and RandomForest

Good feature engineering (titles, family size, ticket prefix, cabin deck, etc.)

Clean preprocessing

Automatic handling of missing values

Saves submission file and model

Code is short, simple, and easy to read

Required Libraries

Install these before running the script:

pip install numpy pandas scikit-learn joblib lightgbm


LightGBM is optional. If it does not install, the code will auto-switch to RandomForest.

How to Run

Put train.csv and test.csv in the same folder as the script.

Run:

python titanic_strong_pipeline.py


Your output file will be saved as:

submission_formula_<timestamp>.csv


Upload this to Kaggle.

Common Mistakes to Avoid

Do not rename train.csv or test.csv

Do not remove any columns from the dataset

Install all required libraries

Make sure Python version is 3.9+

If you edit the code, do not remove the feature engineering part

If LightGBM gives errors, just let it fall back to RandomForest
