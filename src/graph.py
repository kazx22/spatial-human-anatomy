import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


MODEL_RESULTS = []
HUMAN_COMPARISON_RESULTS = []


def add_model_result(
    model_name,
    precision,
    recall,
    f1,
    report_dict,
    runtime=None,
    y_true=None,
    y_pred=None,
):
    MODEL_RESULTS.append(
        {
            "model": model_name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "runtime": runtime,
            "report": report_dict,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )


def add_human_comparison_result(
    model_name,
    precision,
    recall,
    f1,
    report_dict,
    runtime=None,
    y_true=None,
    y_pred=None,
):
    HUMAN_COMPARISON_RESULTS.append(
        {
            "model": model_name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "runtime": runtime,
            "report": report_dict,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )


def ensure_figure_dir():
    os.makedirs("figure", exist_ok=True)


def plot_model_performance():
    if not MODEL_RESULTS:
        print("No model results to plot.")
        return

    models = [r["model"] for r in MODEL_RESULTS]
    precisions = [r["precision"] for r in MODEL_RESULTS]
    recalls = [r["recall"] for r in MODEL_RESULTS]
    f1_scores = [r["f1"] for r in MODEL_RESULTS]

    x = np.arange(len(models))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precisions, width, label="Precision")
    plt.bar(x, recalls, width, label="Recall")
    plt.bar(x + width, f1_scores, width, label="F1-score")

    plt.xticks(x, models, rotation=15)
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figure/model_performance_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_human_comparison_performance():
    if not HUMAN_COMPARISON_RESULTS:
        print("No human comparison results to plot.")
        return

    models = [r["model"] for r in HUMAN_COMPARISON_RESULTS]
    precisions = [r["precision"] for r in HUMAN_COMPARISON_RESULTS]
    recalls = [r["recall"] for r in HUMAN_COMPARISON_RESULTS]
    f1_scores = [r["f1"] for r in HUMAN_COMPARISON_RESULTS]

    x = np.arange(len(models))
    width = 0.25

    plt.figure(figsize=(8, 6))
    plt.bar(x - width, precisions, width, label="Precision")
    plt.bar(x, recalls, width, label="Recall")
    plt.bar(x + width, f1_scores, width, label="F1-score")

    plt.xticks(x, models, rotation=10)
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("Candidate Gold vs Human Gold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "figure/candidate_vs_human_performance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()


def plot_model_runtime():
    runtime_results = [r for r in MODEL_RESULTS if r["runtime"] is not None]

    if not runtime_results:
        print("No runtime data found for model results.")
        return

    models = [r["model"] for r in runtime_results]
    runtimes = [r["runtime"] for r in runtime_results]

    x = np.arange(len(models))

    plt.figure(figsize=(8, 6))
    plt.bar(x, runtimes)
    plt.xticks(x, models, rotation=15)
    plt.ylabel("Average Runtime per Note (seconds)")
    plt.title("Model Runtime Comparison")
    plt.tight_layout()
    plt.savefig("figure/model_runtime_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_model_confusion_matrices():
    labels = ["O", "B-DISEASE", "I-DISEASE", "B-CHEMICAL", "I-CHEMICAL"]

    if not MODEL_RESULTS:
        print("No model results for confusion matrices.")
        return

    for r in MODEL_RESULTS:
        if r["y_true"] is None or r["y_pred"] is None:
            print(f"No y_true / y_pred found for {r['model']}")
            continue

        cm = confusion_matrix(r["y_true"], r["y_pred"], labels=labels)

        fig, ax = plt.subplots(figsize=(7, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, values_format="d", cmap="Blues")

        plt.title(f"Confusion Matrix - {r['model']}")
        plt.tight_layout()
        plt.savefig(
            f"figure/confusion_matrix_{r['model'].lower().replace(' ', '_')}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()
        plt.close(fig)


def plot_human_comparison_confusion_matrices():
    labels = ["O", "B-DISEASE", "I-DISEASE", "B-CHEMICAL", "I-CHEMICAL"]

    if not HUMAN_COMPARISON_RESULTS:
        print("No human comparison results for confusion matrices.")
        return

    for r in HUMAN_COMPARISON_RESULTS:
        if r["y_true"] is None or r["y_pred"] is None:
            print(f"No y_true / y_pred found for {r['model']}")
            continue

        cm = confusion_matrix(r["y_true"], r["y_pred"], labels=labels)

        fig, ax = plt.subplots(figsize=(7, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, values_format="d", cmap="Blues")

        plt.title(f"Confusion Matrix - {r['model']}")
        plt.tight_layout()
        plt.savefig(
            "figure/confusion_matrix_candidate_vs_human.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()
        plt.close(fig)


def plot_per_label_performance():
    if not MODEL_RESULTS:
        print("No model results to plot for per-label performance.")
        return

    rows = []

    for r in MODEL_RESULTS:
        model_name = r["model"]
        report = r["report"]

        for entity_label in ["DISEASE", "CHEMICAL"]:
            if entity_label not in report:
                continue

            rows.append(
                [
                    model_name,
                    entity_label,
                    f"{report[entity_label]['precision']:.4f}",
                    f"{report[entity_label]['recall']:.4f}",
                    f"{report[entity_label]['f1-score']:.4f}",
                    str(int(report[entity_label]["support"])),
                ]
            )

    if not rows:
        print("No per-label rows could be created.")
        return

    col_labels = ["Model", "Label", "Precision", "Recall", "F1-score", "Support"]

    fig_height = max(3, 0.6 * len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        bbox=[0, 0, 1, 1],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    plt.title("Per-label Performance", pad=12)
    plt.savefig("figure/per_label_performance.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_all():
    ensure_figure_dir()
    plot_model_performance()
    plot_human_comparison_performance()
    plot_model_runtime()
    plot_per_label_performance()
    plot_model_confusion_matrices()
    plot_human_comparison_confusion_matrices()
