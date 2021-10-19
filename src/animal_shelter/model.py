import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import joblib

from typing import Tuple
from pathlib import Path
from sklearn.base import RegressorMixin

simple_cols = ["animal_type", "sex_upon_outcome"]


def split_x_y(train: pd.DataFrame) -> pd.DataFrame():
    x = train.drop("outcome_type", axis=1)
    y = train["outcome_type"]
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1)

    return x_train, x_val, y_train, y_val


def fit_model(
    x_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[RegressorMixin, float]:
    x_train_dummies = pd.get_dummies(x_train.loc[:, simple_cols])
    param_grid = {"C": [1e-3, 1e-2, 1e-1]}
    grid_search = GridSearchCV(
        LogisticRegression(), param_grid=param_grid, scoring="neg_log_loss"
    )

    grid_search.fit(x_train_dummies, y_train)

    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_

    return best_model, best_score


def predict_model(model: RegressorMixin, data: pd.DataFrame) -> pd.DataFrame:
    x_test = data  # no y in there
    x_pred_dummies = pd.get_dummies(x_test.loc[:, simple_cols])
    y_pred = model.predict_proba(x_pred_dummies)
    classes = model.classes_.tolist()
    proba_df = pd.DataFrame(y_pred, columns=classes)

    proba_df["id"] = data["id"]
    reordered = proba_df[["id"] + classes]

    return reordered


def save_model(model, path: Path) -> None:
    joblib.dump(model, path)


def load_model(path: Path) -> RegressorMixin:
    outcome_model = joblib.load(path)
    return outcome_model


def save_results(predicts: pd.DataFrame, path: Path) -> None:
    predicts.to_csv(path, index=False)
