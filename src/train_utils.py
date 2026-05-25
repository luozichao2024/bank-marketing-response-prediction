from typing import Dict, List
from xgboost import XGBClassifier
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """Build preprocessing pipeline for numeric and categorical features."""
    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def build_models(
    numeric_features: List[str],
    categorical_features: List[str],
    random_state: int,
) -> Dict[str, GridSearchCV]:
    """Build model pipelines and parameter grids."""
    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features
    )

    models = {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear"
            ),
            {
                "classifier__C": [0.1, 1.0, 10.0],
            },
        ),

        "Decision Tree": (
            DecisionTreeClassifier(
                random_state=random_state,
                class_weight="balanced"
            ),
            {
                "classifier__max_depth": [4, 6, 8, None],
                "classifier__min_samples_leaf": [1, 5, 10],
            },
        ),

        "Random Forest": (
            RandomForestClassifier(
                random_state=random_state,
                class_weight="balanced",
                n_jobs=-1,
            ),
            {
                "classifier__n_estimators": [200, 300],
                "classifier__max_depth": [10, 20, None],
                "classifier__min_samples_leaf": [1, 3, 5],
                "classifier__min_samples_split": [2, 5, 10],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                random_state=random_state,
                eval_metric="logloss",
                n_jobs=-1,
                scale_pos_weight=7.5,
            ),
            {
                "classifier__n_estimators": [200, 300],
                "classifier__max_depth": [3, 5, 7],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__subsample": [0.8, 1.0],
                "classifier__colsample_bytree": [0.8, 1.0],
            },
        ),
    }

    grid_searches = {}

    for model_name, (model, param_grid) in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", model),
            ]
        )

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=3,
            scoring="f1",
            n_jobs=-1,
            verbose=1,
        )

        grid_searches[model_name] = grid_search

    return grid_searches


def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model on the test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    return metrics, y_pred, y_proba


def get_feature_importance(
    model,
    numeric_features: List[str],
    categorical_features: List[str],
) -> pd.DataFrame:
    """Get feature importance from a trained tree-based pipeline."""
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    if not hasattr(classifier, "feature_importances_"):
        raise ValueError(
            "The selected model does not support feature importance."
        )

    categorical_encoder = preprocessor.named_transformers_["cat"]
    categorical_feature_names = categorical_encoder.get_feature_names_out(
        categorical_features
    )

    feature_names = list(numeric_features) + list(categorical_feature_names)

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": classifier.feature_importances_,
        }
    )

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    ).reset_index(drop=True)

    return importance_df