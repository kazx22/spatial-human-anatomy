# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# MODEL_RESULTS_HUMAN = []
# MODEL_RESULTS_CANDIDATE = []
# HUMAN_COMPARISON_RESULTS = []


# def add_model_result_human(
#     model_name,
#     precision,
#     recall,
#     f1,
#     report_dict,
#     runtime=None,
#     y_true=None,
#     y_pred=None,
# ):
#     MODEL_RESULTS_HUMAN.append(
#         {
#             "model": model_name,
#             "precision": precision,
#             "recall": recall,
#             "f1": f1,
#             "runtime": runtime,
#             "report": report_dict,
#             "y_true": y_true,
#             "y_pred": y_pred,
#         }
#     )


# def add_model_result_candidate(
#     model_name,
#     precision,
#     recall,
#     f1,
#     report_dict,
#     runtime=None,
#     y_true=None,
#     y_pred=None,
# ):
#     MODEL_RESULTS_CANDIDATE.append(
#         {
#             "model": model_name,
#             "precision": precision,
#             "recall": recall,
#             "f1": f1,
#             "runtime": runtime,
#             "report": report_dict,
#             "y_true": y_true,
#             "y_pred": y_pred,
#         }
#     )


# def add_human_comparison_result(
#     model_name,
#     precision,
#     recall,
#     f1,
#     report_dict,
#     runtime=None,
#     y_true=None,
#     y_pred=None,
# ):
#     HUMAN_COMPARISON_RESULTS.append(
#         {
#             "model": model_name,
#             "precision": precision,
#             "recall": recall,
#             "f1": f1,
#             "runtime": runtime,
#             "report": report_dict,
#             "y_true": y_true,
#             "y_pred": y_pred,
#         }
#     )


# def ensure_figure_dir():
#     os.makedirs("figure", exist_ok=True)


# def plot_performance(results, title, output_file):
#     if not results:
#         print(f"No results to plot for {title}.")
#         return

#     models = [r["model"] for r in results]
#     precisions = [r["precision"] for r in results]
#     recalls = [r["recall"] for r in results]
#     f1_scores = [r["f1"] for r in results]

#     x = np.arange(len(models))
#     width = 0.25

#     plt.figure(figsize=(10, 6))
#     plt.bar(x - width, precisions, width, label="Precision")
#     plt.bar(x, recalls, width, label="Recall")
#     plt.bar(x + width, f1_scores, width, label="F1-score")

#     plt.xticks(x, models, rotation=15)
#     plt.ylabel("Score")
#     plt.ylim(0, 1)
#     plt.title(title)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(output_file, dpi=300, bbox_inches="tight")
#     plt.show()
#     plt.close()


# def plot_runtime(results, title, output_file):
#     runtime_results = [r for r in results if r["runtime"] is not None]

#     if not runtime_results:
#         print(f"No runtime data found for {title}.")
#         return

#     models = [r["model"] for r in runtime_results]
#     runtimes = [r["runtime"] for r in runtime_results]

#     x = np.arange(len(models))

#     plt.figure(figsize=(8, 6))
#     plt.bar(x, runtimes)
#     plt.xticks(x, models, rotation=15)
#     plt.ylabel("Average Runtime per Note (seconds)")
#     plt.title(title)
#     plt.tight_layout()
#     plt.savefig(output_file, dpi=300, bbox_inches="tight")
#     plt.show()
#     plt.close()


# def plot_confusion_matrices(results, prefix, title_prefix):
#     labels = ["O", "B-DISEASE", "I-DISEASE", "B-CHEMICAL", "I-CHEMICAL"]

#     if not results:
#         print(f"No results for confusion matrices: {title_prefix}")
#         return

#     for r in results:
#         if r["y_true"] is None or r["y_pred"] is None:
#             print(f"No y_true / y_pred found for {r['model']}")
#             continue

#         cm = confusion_matrix(r["y_true"], r["y_pred"], labels=labels)

#         fig, ax = plt.subplots(figsize=(7, 6))
#         disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
#         disp.plot(ax=ax, values_format="d", cmap="Blues")

#         plt.title(f"{title_prefix} - {r['model']}")
#         plt.tight_layout()
#         safe_name = r["model"].lower().replace(" ", "_")
#         plt.savefig(
#             f"figure/{prefix}_{safe_name}.png",
#             dpi=300,
#             bbox_inches="tight",
#         )
#         plt.show()
#         plt.close(fig)


# def plot_per_label_table(results, title, output_file):
#     if not results:
#         print(f"No results to plot for {title}.")
#         return

#     rows = []

#     for r in results:
#         model_name = r["model"]
#         report = r["report"]

#         for entity_label in ["DISEASE", "CHEMICAL"]:
#             if entity_label not in report:
#                 continue

#             rows.append(
#                 [
#                     model_name,
#                     entity_label,
#                     f"{report[entity_label]['precision']:.4f}",
#                     f"{report[entity_label]['recall']:.4f}",
#                     f"{report[entity_label]['f1-score']:.4f}",
#                     str(int(report[entity_label]["support"])),
#                 ]
#             )

#     if not rows:
#         print(f"No per-label rows could be created for {title}.")
#         return

#     col_labels = ["Model", "Label", "Precision", "Recall", "F1-score", "Support"]

#     fig_height = max(3, 0.6 * len(rows) + 1.5)
#     fig, ax = plt.subplots(figsize=(10, fig_height))
#     ax.axis("off")

#     table = ax.table(
#         cellText=rows,
#         colLabels=col_labels,
#         loc="center",
#         cellLoc="center",
#         bbox=[0, 0, 1, 1],
#     )

#     table.auto_set_font_size(False)
#     table.set_fontsize(10)
#     table.scale(1, 1.4)

#     plt.title(title, pad=12)
#     plt.savefig(output_file, dpi=300, bbox_inches="tight")
#     plt.show()
#     plt.close(fig)


# def plot_human_comparison_performance():
#     plot_performance(
#         HUMAN_COMPARISON_RESULTS,
#         "Candidate Gold vs Human Gold",
#         "figure/candidate_vs_human_performance.png",
#     )


# def plot_model_performance_human():
#     plot_performance(
#         MODEL_RESULTS_HUMAN,
#         "Model Performance vs Human Gold",
#         "figure/model_performance_human_gold.png",
#     )


# def plot_model_performance_candidate():
#     plot_performance(
#         MODEL_RESULTS_CANDIDATE,
#         "Model Performance vs Candidate Gold",
#         "figure/model_performance_candidate_gold.png",
#     )


# def plot_model_runtime_human():
#     plot_runtime(
#         MODEL_RESULTS_HUMAN,
#         "Model Runtime",
#         "figure/model_runtime_human_gold.png",
#     )


# def plot_model_runtime_candidate():
#     plot_runtime(
#         MODEL_RESULTS_CANDIDATE,
#         "Model Runtime vs Candidate Gold",
#         "figure/model_runtime_candidate_gold.png",
#     )


# def plot_per_label_performance_human():
#     plot_per_label_table(
#         MODEL_RESULTS_HUMAN,
#         "Per-label Performance vs Human Gold",
#         "figure/per_label_performance_human_gold.png",
#     )


# def plot_per_label_performance_candidate():
#     plot_per_label_table(
#         MODEL_RESULTS_CANDIDATE,
#         "Per-label Performance vs Candidate Gold",
#         "figure/per_label_performance_candidate_gold.png",
#     )


# def plot_model_confusion_matrices_human():
#     plot_confusion_matrices(
#         MODEL_RESULTS_HUMAN,
#         "confusion_matrix_human",
#         "Confusion Matrix vs Human Gold",
#     )


# def plot_model_confusion_matrices_candidate():
#     plot_confusion_matrices(
#         MODEL_RESULTS_CANDIDATE,
#         "confusion_matrix_candidate",
#         "Confusion Matrix vs Candidate Gold",
#     )


# def plot_human_comparison_confusion_matrices():
#     labels = ["O", "B-DISEASE", "I-DISEASE", "B-CHEMICAL", "I-CHEMICAL"]

#     if not HUMAN_COMPARISON_RESULTS:
#         print("No human comparison results for confusion matrices.")
#         return

#     for r in HUMAN_COMPARISON_RESULTS:
#         if r["y_true"] is None or r["y_pred"] is None:
#             print(f"No y_true / y_pred found for {r['model']}")
#             continue

#         cm = confusion_matrix(r["y_true"], r["y_pred"], labels=labels)

#         fig, ax = plt.subplots(figsize=(7, 6))
#         disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
#         disp.plot(ax=ax, values_format="d", cmap="Blues")

#         plt.title(f"Confusion Matrix - {r['model']}")
#         plt.tight_layout()
#         plt.savefig(
#             "figure/confusion_matrix_candidate_vs_human.png",
#             dpi=300,
#             bbox_inches="tight",
#         )
#         plt.show()
#         plt.close(fig)


# def plot_all():
#     ensure_figure_dir()

#     plot_human_comparison_performance()
#     plot_human_comparison_confusion_matrices()

#     plot_model_performance_human()
#     plot_model_runtime_human()
#     plot_per_label_performance_human()
#     plot_model_confusion_matrices_human()

#     plot_model_performance_candidate()
#     plot_model_runtime_candidate()
#     plot_per_label_performance_candidate()
#     plot_model_confusion_matrices_candidate()


import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Publication settings
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# Colour palette — accessible, print-safe
COLOURS = {
    "precision": "#2166ac",
    "recall": "#d6604d",
    "f1": "#4dac26",
}

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


# --- Figure 1: Base models vs human gold ---


def plot_models_vs_human_gold():
    # Only base models, no filtered variants
    base_models = ["SciSpacy", "BioBERT", "PubMedBERT", "ClinicalBERT", "BioELECTRA"]
    results = [r for r in MODEL_RESULTS_HUMAN if r["model"] in base_models]

    if not results:
        print("No base model results for Figure 1.")
        return

    # Preserve order
    results = sorted(results, key=lambda r: base_models.index(r["model"]))

    models = [r["model"] for r in results]
    precisions = [r["precision"] for r in results]
    recalls = [r["recall"] for r in results]
    f1_scores = [r["f1"] for r in results]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))

    bars_p = ax.bar(
        x - width,
        precisions,
        width,
        label="Precision",
        color=COLOURS["precision"],
        edgecolor="white",
        linewidth=0.5,
    )
    bars_r = ax.bar(
        x,
        recalls,
        width,
        label="Recall",
        color=COLOURS["recall"],
        edgecolor="white",
        linewidth=0.5,
    )
    bars_f = ax.bar(
        x + width,
        f1_scores,
        width,
        label="F1-score",
        color=COLOURS["f1"],
        edgecolor="white",
        linewidth=0.5,
    )

    # Value labels on bars
    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title("Model Performance Against Human Gold Annotations (BC5CDR)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig("figure/fig1_models_vs_human_gold.png")
    plt.close()
    print("Saved: figure/fig1_models_vs_human_gold.png")


# --- Figure 2: Weighted vs majority vs human gold ---


def plot_candidate_gold_comparison():
    if not HUMAN_COMPARISON_RESULTS:
        print("No human comparison results for Figure 2.")
        return

    # Expect two entries: majority and weighted
    results = HUMAN_COMPARISON_RESULTS

    labels = [r["model"] for r in results]
    precisions = [r["precision"] for r in results]
    recalls = [r["recall"] for r in results]
    f1_scores = [r["f1"] for r in results]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))

    bars_p = ax.bar(
        x - width,
        precisions,
        width,
        label="Precision",
        color=COLOURS["precision"],
        edgecolor="white",
        linewidth=0.5,
    )
    bars_r = ax.bar(
        x,
        recalls,
        width,
        label="Recall",
        color=COLOURS["recall"],
        edgecolor="white",
        linewidth=0.5,
    )
    bars_f = ax.bar(
        x + width,
        f1_scores,
        width,
        label="F1-score",
        color=COLOURS["f1"],
        edgecolor="white",
        linewidth=0.5,
    )

    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Majority Candidate Gold", "Weighted Candidate Gold"], rotation=0
    )
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title("Pseudo-Gold Annotation Quality Against Human Gold (BC5CDR)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig("figure/fig2_candidate_gold_comparison.png")
    plt.close()
    print("Saved: figure/fig2_candidate_gold_comparison.png")


# --- Figure 3: Per-label performance (DISEASE vs CHEMICAL) ---


def plot_per_label_performance():
    base_models = ["SciSpacy", "BioBERT", "PubMedBERT", "ClinicalBERT", "BioELECTRA"]
    results = [r for r in MODEL_RESULTS_HUMAN if r["model"] in base_models]

    if not results:
        print("No results for Figure 3.")
        return

    results = sorted(results, key=lambda r: base_models.index(r["model"]))

    models = [r["model"] for r in results]
    disease_f1 = [r["report"].get("DISEASE", {}).get("f1-score", 0) for r in results]
    chemical_f1 = [r["report"].get("CHEMICAL", {}).get("f1-score", 0) for r in results]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    bars_d = ax.bar(
        x - width / 2,
        disease_f1,
        width,
        label="DISEASE",
        color="#b2182b",
        edgecolor="white",
        linewidth=0.5,
    )
    bars_c = ax.bar(
        x + width / 2,
        chemical_f1,
        width,
        label="CHEMICAL",
        color="#2166ac",
        edgecolor="white",
        linewidth=0.5,
    )

    for bars in [bars_d, bars_c]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0)
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.08)
    ax.set_title("Per-Label F1-score Against Human Gold (BC5CDR)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig("figure/fig3_per_label_f1.png")
    plt.close()
    print("Saved: figure/fig3_per_label_f1.png")


# --- Figure 4: SciSpacy confusion matrix vs human gold ---


def plot_scispacy_confusion_matrix():
    results = [r for r in MODEL_RESULTS_HUMAN if r["model"] == "SciSpacy"]

    if not results or results[0]["y_true"] is None:
        print("No SciSpacy data for Figure 4.")
        return

    r = results[0]
    labels = ["O", "B-DISEASE", "I-DISEASE", "B-CHEMICAL", "I-CHEMICAL"]
    cm = confusion_matrix(r["y_true"], r["y_pred"], labels=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, values_format="d", cmap="Blues", colorbar=False)

    ax.set_title("SciSpacy — Confusion Matrix vs Human Gold (BC5CDR)")
    plt.tight_layout()
    plt.savefig("figure/fig4_scispacy_confusion_matrix.png")
    plt.close()
    print("Saved: figure/fig4_scispacy_confusion_matrix.png")


# --- Figure 5: Weighted candidate gold confusion matrix vs human gold ---


def plot_weighted_gold_confusion_matrix():
    results = [r for r in HUMAN_COMPARISON_RESULTS if "Weighted" in r["model"]]

    if not results or results[0]["y_true"] is None:
        print("No weighted candidate gold data for Figure 5.")
        return

    r = results[0]
    labels = ["O", "B-DISEASE", "I-DISEASE", "B-CHEMICAL", "I-CHEMICAL"]
    cm = confusion_matrix(r["y_true"], r["y_pred"], labels=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, values_format="d", cmap="Blues", colorbar=False)

    ax.set_title("Weighted Candidate Gold — Confusion Matrix vs Human Gold (BC5CDR)")
    plt.tight_layout()
    plt.savefig("figure/fig5_weighted_gold_confusion_matrix.png")
    plt.close()
    print("Saved: figure/fig5_weighted_gold_confusion_matrix.png")


def plot_all():
    ensure_figure_dir()

    plot_models_vs_human_gold()
    plot_candidate_gold_comparison()
    plot_per_label_performance()
    plot_scispacy_confusion_matrix()
    plot_weighted_gold_confusion_matrix()
