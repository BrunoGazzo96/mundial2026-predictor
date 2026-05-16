import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLS

OUTPUTS = Path(__file__).parent.parent / "outputs"


def train(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "gbm",
) -> tuple:
    """
    Entrena el modelo y evalúa en test set.
    Retorna (model, X_test, y_test, scaler_or_None).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = None

    if model_type == "gbm":
        clf = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
    else:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)

    print(f"\n{'='*50}")
    print(f"Modelo: {model_type.upper()}")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"Log-loss: {log_loss(y_test, probs):.4f}")
    print(classification_report(y_test, preds, target_names=["Derrota", "Empate", "Victoria"]))

    return clf, X_test, y_test, scaler


def feature_importance(model, feature_names: list) -> pd.DataFrame:
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame()
    imp = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    return imp


def save_model(model, scaler=None, name: str = "model_gbm.joblib"):
    joblib.dump({"model": model, "scaler": scaler}, OUTPUTS / name)
    print(f"Modelo guardado: outputs/{name}")


def load_model(name: str = "model_gbm.joblib") -> tuple:
    data = joblib.load(OUTPUTS / name)
    return data["model"], data["scaler"]
