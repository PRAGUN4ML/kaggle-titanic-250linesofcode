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
