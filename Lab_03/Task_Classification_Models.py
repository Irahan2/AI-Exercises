import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

# Make console output cleaner on Windows (suppress noisy KMeans/joblib warnings)
os.environ.setdefault("OMP_NUM_THREADS", "1")
if "LOKY_MAX_CPU_COUNT" not in os.environ:
    try:
        os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)
    except Exception:
        pass
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="KMeans is known to have a memory leak on Windows with MKL*",
    category=UserWarning,
)


def load_xy():
    
    try:
        base = Path(__file__).parent
    except NameError:
        base = Path(os.getcwd())
    df = pd.read_csv(base / "classification_set.csv", header=None)
    X = df.iloc[:, :-1].to_numpy(dtype=float)
    y = df.iloc[:, -1].to_numpy(dtype=int)
    return X, y


def majority_map(y_train, clusters_train):
    mapping = {}
    for c in np.unique(clusters_train):
        m = np.bincount(y_train[clusters_train == c]).argmax()
        mapping[c] = int(m)
    return mapping


def main():
    X, y = load_xy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # SVM (RBF)
    svm = SVC(kernel="rbf", gamma="scale", C=1.0, random_state=42)
    svm.fit(X_tr_s, y_tr)
    y_pr_svm = svm.predict(X_te_s)
    print(f"SVM accuracy: {accuracy_score(y_te, y_pr_svm):.3f}")
    cm_svm = confusion_matrix(y_te, y_pr_svm)
    print("SVM Confusion Matrix [rows=true, cols=pred]:\n", cm_svm)

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_tr, y_tr)
    y_pr_dt = dt.predict(X_te)
    print(f"DecisionTree accuracy: {accuracy_score(y_te, y_pr_dt):.3f}")
    cm_dt = confusion_matrix(y_te, y_pr_dt)
    print("DecisionTree Confusion Matrix [rows=true, cols=pred]:\n", cm_dt)

    # K-Means (sample, k=2) + majority mapping
    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    km.fit(X_tr_s)
    map_c2y = majority_map(y_tr, km.labels_)
    y_pr_km = np.vectorize(map_c2y.get)(km.predict(X_te_s))
    print(f"KMeans(mapped) accuracy: {accuracy_score(y_te, y_pr_km):.3f}")
    cm_km = confusion_matrix(y_te, y_pr_km)
    print("KMeans(mapped) Confusion Matrix [rows=true, cols=pred]:\n", cm_km)

    # Confusion matrices (3 side-by-side)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    ConfusionMatrixDisplay.from_predictions(y_te, y_pr_svm, ax=axes[0])
    axes[0].set_title("SVM")
    ConfusionMatrixDisplay.from_predictions(y_te, y_pr_dt, ax=axes[1])
    axes[1].set_title("Decision Tree")
    ConfusionMatrixDisplay.from_predictions(y_te, y_pr_km, ax=axes[2])
    axes[2].set_title("K-Means (mapped)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
