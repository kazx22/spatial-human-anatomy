import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = []


def add_result(model_name, precision, recall, f1, report_dict, runtime=None):
    RESULTS.append(
        {
            "model": model_name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "runtime": runtime,
            "report": report_dict,
        }
    )


def plot_model_performance():
    models = [r["model"] for r in RESULTS]
    precisions = [r["precision"] for r in RESULTS]
    recalls = [r["recall"] for r in RESULTS]
    f1_scores = [r["f1"] for r in RESULTS]

    x = np.arange(len(models))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precisions, width, label="Precision")
    plt.bar(x, recalls, width, label="Recall")
    plt.bar(x + width, f1_scores, width, label="F1-score")

    plt.xticks(x, models)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "figure/model_performance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_runtime_comparison():
    models = [r["model"] for r in RESULTS]
    runtimes = [r["runtime"] for r in RESULTS]

    plt.figure(figsize=(8, 5))
    plt.bar(models, runtimes)
    plt.ylabel("Average Time per Note (seconds)")
    plt.title("Runtime Comparison")
    plt.tight_layout()
    plt.savefig(
        "figure/runtime_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def print_per_label_table():
    rows = []

    for r in RESULTS:
        for label in ["DISEASE", "CHEMICAL"]:
            if label in r["report"]:
                rows.append(
                    {
                        "Model": r["model"],
                        "Label": label,
                        "Precision": r["report"][label]["precision"],
                        "Recall": r["report"][label]["recall"],
                        "F1-score": r["report"][label]["f1-score"],
                        "Support": r["report"][label]["support"],
                    }
                )

    df = pd.DataFrame(rows)
    print("\nPer-label Performance")
    print(df.to_string(index=False))


def plot_per_label_f1():
    models = [r["model"] for r in RESULTS]
    x = np.arange(len(models))
    width = 0.35

    disease_f1 = []
    chemical_f1 = []

    for r in RESULTS:
        disease_f1.append(r["report"].get("DISEASE", {}).get("f1-score", 0))
        chemical_f1.append(r["report"].get("CHEMICAL", {}).get("f1-score", 0))

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, disease_f1, width, label="DISEASE")
    plt.bar(x + width / 2, chemical_f1, width, label="CHEMICAL")

    plt.xticks(x, models)
    plt.ylim(0, 1)
    plt.ylabel("F1-score")
    plt.title("Per-label Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figure/per_label_f1.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_all():
    plot_model_performance()
    plot_runtime_comparison()
    print_per_label_table()
    plot_per_label_f1()
