import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from evaluate import evaluate_model


DATASET_PATH = "../outputs/features_dataset.csv"
MODEL_DIR = "../models"


def load_dataset(csv_path=DATASET_PATH):
    df = pd.read_csv(csv_path)

    X = df.drop(columns=["label", "label_name", "file_path"])
    y = df["label"]

    return X, y, df


def train_models(X_train, y_train):
    models = {}

    # Random Forest
    rf_pipeline = Pipeline([
        ("clf", RandomForestClassifier(random_state=42))
    ])
    rf_params = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5]
    }
    rf_grid = GridSearchCV(
        rf_pipeline,
        rf_params,
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)
    models["RandomForest"] = rf_grid

    # SVM
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, random_state=42))
    ])
    svm_params = {
        "clf__C": [0.1, 1, 10],
        "clf__kernel": ["linear", "rbf"],
        "clf__gamma": ["scale", "auto"]
    }
    svm_grid = GridSearchCV(
        svm_pipeline,
        svm_params,
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1
    )
    svm_grid.fit(X_train, y_train)
    models["SVM"] = svm_grid

    # Logistic Regression
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=42))
    ])
    lr_params = {
        "clf__C": [0.1, 1, 10]
    }
    lr_grid = GridSearchCV(
        lr_pipeline,
        lr_params,
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1
    )
    lr_grid.fit(X_train, y_train)
    models["LogisticRegression"] = lr_grid

    return models


def compare_models(models, X_train, y_train, X_test, y_test):
    results = []

    for model_name, grid in models.items():
        best_model = grid.best_estimator_

        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="accuracy")
        y_pred = best_model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)

        print(f"\nModel: {model_name}")
        print(f"Best Params: {grid.best_params_}")
        print(f"CV Accuracy Mean: {cv_scores.mean():.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(classification_report(y_test, y_pred))

        results.append({
            "model": model_name,
            "best_params": grid.best_params_,
            "cv_accuracy_mean": cv_scores.mean(),
            "test_accuracy": test_acc
        })

    return results


def save_model(model, file_name):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, file_name)
    joblib.dump(model, path)
    print(f"Saved model to {path}")


def main():
    X, y, df = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = train_models(X_train, y_train)
    results = compare_models(models, X_train, y_train, X_test, y_test)

    # choose best by test accuracy
    best_model_name = max(results, key=lambda x: x["test_accuracy"])["model"]
    best_grid = models[best_model_name]
    best_model = best_grid.best_estimator_

    print(f"\nBest overall model: {best_model_name}")

    save_model(best_model, "best_model.joblib")
    save_model(models["RandomForest"].best_estimator_, "random_forest_model.joblib")
    save_model(models["SVM"].best_estimator_, "svm_model.joblib")
    save_model(models["LogisticRegression"].best_estimator_, "logistic_regression_model.joblib")

    # save feature column order
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "feature_columns.joblib"))

    # evaluate best model in detail
    evaluate_model(best_model, X_test, y_test, class_names=["Normal", "Slight Fault", "Severe Fault"])


if __name__ == "__main__":
    main()