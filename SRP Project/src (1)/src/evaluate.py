import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)


OUTPUT_DIR = "../outputs"


def evaluate_model(model, X_test, y_test, class_names):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted"
    )

    print("\n===== FINAL EVALUATION =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Save classification report
    report_text = classification_report(y_test, y_pred, target_names=class_names)
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n\n")
        f.write(report_text)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()

    # Feature importance if Random Forest
    final_clf = None
    if hasattr(model, "named_steps"):
        final_clf = model.named_steps.get("clf", None)

    if final_clf is not None and hasattr(final_clf, "feature_importances_"):
        importances = final_clf.feature_importances_
        indices = np.argsort(importances)[::-1][:15]

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), indices, rotation=45)
        plt.title("Top 15 Feature Importances (Random Forest)")
        plt.xlabel("Feature Index")
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
        plt.close()