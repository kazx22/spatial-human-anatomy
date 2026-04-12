import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


MODEL_RESULTS_HUMAN = []
MODEL_RESULTS_CANDIDATE = []
HUMAN_COMPARISON_RESULTS = []


def add_model_result_human(
    model_name,
    precision,
    recall,
    f1,
    report_dict,
    runtime=None,
    y_true=None,
    y_pred=None,
):
    MODEL_RESULTS_HUMAN.append(
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


def add_model_result_candidate(
    model_name,
    precision,
    recall,
    f1,
    report_dict,
    runtime=None,
    y_true=None,
    y_pred=None,
):
    MODEL_RESULTS_CANDIDATE.append(
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


def plot_performance(results, title, output_file):
    if not results:
        print(f"No results to plot for {title}.")
        return

    models = [r["model"] for r in results]
    precisions = [r["precision"] for r in results]
    recalls = [r["recall"] for r in results]
    f1_scores = [r["f1"] for r in results]

    x = np.arange(len(models))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precisions, width, label="Precision")
    plt.bar(x, recalls, width, label="Recall")
    plt.bar(x + width, f1_scores, width, label="F1-score")

    plt.xticks(x, models, rotation=15)
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_runtime(results, title, output_file):
    runtime_results = [r for r in results if r["runtime"] is not None]

    if not runtime_results:
        print(f"No runtime data found for {title}.")
        return

    models = [r["model"] for r in runtime_results]
    runtimes = [r["runtime"] for r in runtime_results]

    x = np.arange(len(models))

    plt.figure(figsize=(8, 6))
    plt.bar(x, runtimes)
    plt.xticks(x, models, rotation=15)
    plt.ylabel("Average Runtime per Note (seconds)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_confusion_matrices(results, prefix, title_prefix):
    labels = ["O", "B-DISEASE", "I-DISEASE", "B-CHEMICAL", "I-CHEMICAL"]

    if not results:
        print(f"No results for confusion matrices: {title_prefix}")
        return

    for r in results:
        if r["y_true"] is None or r["y_pred"] is None:
            print(f"No y_true / y_pred found for {r['model']}")
            continue

        cm = confusion_matrix(r["y_true"], r["y_pred"], labels=labels)

        fig, ax = plt.subplots(figsize=(7, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, values_format="d", cmap="Blues")

        plt.title(f"{title_prefix} - {r['model']}")
        plt.tight_layout()
        safe_name = r["model"].lower().replace(" ", "_")
        plt.savefig(
            f"figure/{prefix}_{safe_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()
        plt.close(fig)


def plot_per_label_table(results, title, output_file):
    if not results:
        print(f"No results to plot for {title}.")
        return

    rows = []

    for r in results:
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
        print(f"No per-label rows could be created for {title}.")
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

    plt.title(title, pad=12)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_human_comparison_performance():
    plot_performance(
        HUMAN_COMPARISON_RESULTS,
        "Candidate Gold vs Human Gold",
        "figure/candidate_vs_human_performance.png",
    )


def plot_model_performance_human():
    plot_performance(
        MODEL_RESULTS_HUMAN,
        "Model Performance vs Human Gold",
        "figure/model_performance_human_gold.png",
    )


def plot_model_performance_candidate():
    plot_performance(
        MODEL_RESULTS_CANDIDATE,
        "Model Performance vs Candidate Gold",
        "figure/model_performance_candidate_gold.png",
    )


def plot_model_runtime_human():
    plot_runtime(
        MODEL_RESULTS_HUMAN,
        "Model Runtime vs Human Gold",
        "figure/model_runtime_human_gold.png",
    )


def plot_model_runtime_candidate():
    plot_runtime(
        MODEL_RESULTS_CANDIDATE,
        "Model Runtime vs Candidate Gold",
        "figure/model_runtime_candidate_gold.png",
    )


def plot_per_label_performance_human():
    plot_per_label_table(
        MODEL_RESULTS_HUMAN,
        "Per-label Performance vs Human Gold",
        "figure/per_label_performance_human_gold.png",
    )


def plot_per_label_performance_candidate():
    plot_per_label_table(
        MODEL_RESULTS_CANDIDATE,
        "Per-label Performance vs Candidate Gold",
        "figure/per_label_performance_candidate_gold.png",
    )


def plot_model_confusion_matrices_human():
    plot_confusion_matrices(
        MODEL_RESULTS_HUMAN,
        "confusion_matrix_human",
        "Confusion Matrix vs Human Gold",
    )


def plot_model_confusion_matrices_candidate():
    plot_confusion_matrices(
        MODEL_RESULTS_CANDIDATE,
        "confusion_matrix_candidate",
        "Confusion Matrix vs Candidate Gold",
    )


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


def plot_all():
    ensure_figure_dir()

    plot_human_comparison_performance()
    plot_human_comparison_confusion_matrices()

    plot_model_performance_human()
    plot_model_runtime_human()
    plot_per_label_performance_human()
    plot_model_confusion_matrices_human()

    plot_model_performance_candidate()
    plot_model_runtime_candidate()
    plot_per_label_performance_candidate()
    plot_model_confusion_matrices_candidate()
