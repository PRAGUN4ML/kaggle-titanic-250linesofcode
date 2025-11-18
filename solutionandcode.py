import time
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Optional LightGBM
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

RND = 42

# -----------------------------------------------------------
#                   FEATURE HELPERS
# -----------------------------------------------------------

def extract_title(name):
    """Extract title from passenger name."""
    if not isinstance(name, str):
        return "Rare"
    m = re.search(r",\s*([^\.]+)\.", name)
    if not m:
        return "Rare"
    title = m.group(1).strip()

    mapping = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
    }
    if title in mapping:
        return mapping[title]

    common = ["Mr", "Mrs", "Miss", "Master", "Dr", "Rev"]
    return title if title in common else "Rare"


def ticket_prefix(ticket):
    """Extract alphabetical ticket prefix."""
    ticket = str(ticket)
    ticket = ticket.replace(".", "").replace("/", "")
    parts = ticket.split()
    for p in parts:
        if any(c.isalpha() for c in p):
            return p.upper()
    return "NONE"


def engineer_features(df):
    """Create engineered features."""
    df["FamilySize"] = df["SibSp"].astype(int) + df["Parch"].astype(int) + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    df["Title"] = df["Name"].apply(extract_title)

    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
    df["FarePerPerson"].replace([np.inf, -np.inf], np.nan, inplace=True)

    df["CabinKnown"] = (~df["Cabin"].isna()).astype(int)

    df["TicketPrefix"] = df["Ticket"].apply(ticket_prefix)

    df["Age_bin"] = pd.cut(
        df["Age"],
        bins=[-1, 12, 18, 25, 35, 60, 120],
        labels=["child", "teen", "young", "adult", "mid", "senior"],
    )

    df["Age_x_Pclass"] = df["Age"] * df["Pclass"]

    df["FareLog"] = np.log1p(df["Fare"])

    df["CabinDeck"] = df["Cabin"].astype(str).str[0].replace({"n": "U", "N": "U"})

    return df


# -----------------------------------------------------------
#                   MAIN PIPELINE lets go
# -----------------------------------------------------------

def main():

    # ---------------- Load data ----------------
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    test_ids = test["PassengerId"].copy()

    # ---------------- Basic fills (train-only stats) ----------------
    med_age = train["Age"].median()
    med_fare = train["Fare"].median()
    mode_emb = train["Embarked"].mode()[0]

    full = pd.concat([train, test], ignore_index=True)

    full["Age"] = full["Age"].fillna(med_age)
    full["Fare"] = full["Fare"].fillna(med_fare)
    full["Embarked"] = full["Embarked"].fillna(mode_emb)
    full["Cabin"] = full["Cabin"].fillna("U")
    full["Ticket"] = full["Ticket"].fillna("NONE")
    full["Name"] = full["Name"].fillna("Unknown")

    # ---------------- Feature Engineering ----------------
    full = engineer_features(full)

    train_feat = full.iloc[: len(train)].copy()
    test_feat = full.iloc[len(train) :].copy()

    # ---------------- Columns ----------------
    numeric_cols = [
        "Pclass", "Age", "SibSp", "Parch", "Fare",
        "FamilySize", "IsAlone", "FarePerPerson",
        "CabinKnown", "Age_x_Pclass", "FareLog"
    ]

    cat_cols = ["Title", "Age_bin", "Embarked", "CabinDeck", "TicketPrefix"]

    # ---------------- Reduce ticket prefixes ----------------
    top_prefixes = (
        train_feat["TicketPrefix"]
        .value_counts()
        .nlargest(20)
        .index
        .tolist()
    )

    train_feat["TicketPrefix"] = train_feat["TicketPrefix"].apply(
        lambda t: t if t in top_prefixes else "OTHER"
    )
    test_feat["TicketPrefix"] = test_feat["TicketPrefix"].apply(
        lambda t: t if t in top_prefixes else "OTHER"
    )

    # ---------------- Build X ----------------
    train_X = train_feat[numeric_cols + cat_cols].copy()
    test_X = test_feat[numeric_cols + cat_cols].copy()

    # ---------------- FIX: Convert all categorical cols to strings BEFORE fillna ----------------
    for c in cat_cols:
        train_X[c] = train_X[c].astype(str)
        test_X[c] = test_X[c].astype(str)

        train_X[c] = train_X[c].replace("nan", "MISSING").fillna("MISSING")
        test_X[c] = test_X[c].replace("nan", "MISSING").fillna("MISSING")

    # ---------------- One-hot encode ----------------pandas handles it for us
    train_X = pd.get_dummies(train_X, columns=cat_cols, drop_first=False)
    test_X = pd.get_dummies(test_X, columns=cat_cols, drop_first=False)

    # Align columns
    train_X, test_X = train_X.align(test_X, join="left", axis=1, fill_value=0)

    # ---------------- Scale numeric ----------------
    scaler = StandardScaler()
    train_X[numeric_cols] = scaler.fit_transform(train_X[numeric_cols])
    test_X[numeric_cols] = scaler.transform(test_X[numeric_cols])

    ts = int(time.time())
    joblib.dump(scaler, f"scaler_{ts}.joblib")
    joblib.dump(train_X.columns.tolist(), f"feature_columns_{ts}.joblib")

    # ---------------- Train/Validation Split imp imp  ----------------
    y = train["Survived"].astype(int).values

    X_tr, X_val, y_tr, y_val = train_test_split(
        train_X.values, y, test_size=0.2, random_state=RND, stratify=y
    )

    # -------------------------------------------------------
    #                   MODEL TRAINING yeah reaching it
    # -------------------------------------------------------

    if LGB_AVAILABLE:
        print("Using LightGBM...")

        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val)

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "seed": RND,
        }

        callbacks = [
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(period=200),
        ]

        model = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=2000,
            valid_sets=[val_data],
            callbacks=callbacks,
        )

        val_pred = (model.predict(X_val, num_iteration=model.best_iteration) >= 0.5).astype(int)
        test_pred = (model.predict(test_X, num_iteration=model.best_iteration) >= 0.5).astype(int)

        val_acc = accuracy_score(y_val, val_pred)
        print("Validation accuracy:", val_acc)

        model.save_model(f"lgb_{ts}.txt")

    else:
        print("LightGBM NOT installed — using RandomForest.")

        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            random_state=RND,
            n_jobs=-1,
        )

        rf.fit(X_tr, y_tr)
        val_pred = rf.predict(X_val)
        test_pred = rf.predict(test_X)

        val_acc = accuracy_score(y_val, val_pred)
        print("Validation accuracy:", val_acc)

        joblib.dump(rf, f"rf_{ts}.joblib")

    # ---------------- Save submission wow finished----------------
    sub_file = f"submission_formula_{ts}.csv"
    pd.DataFrame({"PassengerId": test_ids, "Survived": test_pred}).to_csv(sub_file, index=False)

    print("Saved:", sub_file)
    print("DONE.")


if __name__ == "__main__":
    main()


