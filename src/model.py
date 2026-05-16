import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

MODELS_DIR = Path(__file__).parent.parent / "outputs"


def train(X: pd.DataFrame, y: pd.Series, model_type: str = "gbm"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if model_type == "gbm":
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
    else:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        clf = LogisticRegression(max_iter=1000, random_state=42)

    clf.fit(X_train, y_train)
    print(classification_report(y_test, clf.predict(X_test),
                                target_names=["Derrota", "Empate", "Victoria"]))
    return clf, X_test, y_test


def save_model(model, name: str = "model_gbm.joblib"):
    joblib.dump(model, MODELS_DIR / name)


def load_model(name: str = "model_gbm.joblib"):
    return joblib.load(MODELS_DIR / name)
