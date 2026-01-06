import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ======================
# MODELLI
# ======================
models = ["BoostedDT", "RandomForest", "LogReg", "XGBoost"]


# ======================
# DATI AUC [ER, PR, HER2]
# ======================

auc_ambl_radiomic = {
    "BoostedDT":    [0.500, 0.500, 0.500],
    "RandomForest": [0.583, 0.733, 0.714],
    "LogReg":       [0.500, 0.500, 0.500],
    "XGBoost":      [0.887, 0.914, 0.500],
}

auc_ambl_lesions = {
    "BoostedDT":    [0.500, 0.500, 0.500],
    "RandomForest": [0.667, 0.800, 0.756],
    "LogReg":       [0.500, 0.500, 0.500],
    "XGBoost":      [0.888, 0.946, 0.364],
}

auc_duke_radiomic = {
    "BoostedDT":     [0.532, 0.500, 0.437],
    "RandomForest":  [0.515, 0.485, 0.463],
    "LogReg":        [0.519, 0.505, 0.544],
    "XGBoost":       [0.528, 0.506, 0.468],
}

auc_duke_lesions = {
    "BoostedDT":     [0.540, 0.507, 0.409],
    "RandomForest":  [0.530, 0.539, 0.454],
    "LogReg":        [0.528, 0.513, 0.532],
    "XGBoost":       [0.549, 0.512, 0.431],
}

datasets = [
    ("AMBL – Radiomiche", auc_ambl_radiomic, "white"),
    ("AMBL – Lesioni", auc_ambl_lesions, "#d9d9d9"),
    ("DUKE – Radiomiche", auc_duke_radiomic, "#bdbdbd"),
    ("DUKE – Lesioni", auc_duke_lesions, "#969696"),
]

# ======================
# DOT PLOT
# ======================
colors = {
    "AMBL_rad":  "#1f77b4",  # blu
    "AMBL_les":  "#aec7e8",  # azzurro
    "DUKE_rad":  "#ff7f0e",  # arancione
    "DUKE_les":  "#2ca02c",  # verde
}

markersize = 60
alpha = 0.85

for i, model in enumerate(models):
    x_pos = i + 1

    # AMBL – Radiomiche
    plt.scatter(
        [x_pos - 0.18]*3, auc_ambl_radiomic[model],
        color=colors["AMBL_rad"],
        edgecolor="black",
        s=markersize,
        alpha=alpha,
        zorder=3
    )

    # AMBL – Lesioni
    plt.scatter(
        [x_pos - 0.06]*3, auc_ambl_lesions[model],
        color=colors["AMBL_les"],
        edgecolor="black",
        s=markersize,
        alpha=alpha,
        zorder=3
    )

    # DUKE – Radiomiche
    plt.scatter(
        [x_pos + 0.06]*3, auc_duke_radiomic[model],
        color=colors["DUKE_rad"],
        edgecolor="black",
        s=markersize,
        alpha=alpha,
        zorder=3
    )

    # DUKE – Lesioni
    plt.scatter(
        [x_pos + 0.18]*3, auc_duke_lesions[model],
        color=colors["DUKE_les"],
        edgecolor="black",
        s=markersize,
        alpha=alpha,
        zorder=3
    )

# ======================
# STILE
# ======================

plt.ylabel("AUC-ROC")
plt.ylim(0.30, 1.00)
plt.title("Confronto tra modelli – AUC-ROC per biomarcatore (ER, PR, HER2)")

legend_handles = [
    mpatches.Patch(facecolor=colors["AMBL_rad"], edgecolor="black", label="AMBL – Radiomiche"),
    mpatches.Patch(facecolor=colors["AMBL_les"], edgecolor="black", label="AMBL – Lesioni"),
    mpatches.Patch(facecolor=colors["DUKE_rad"], edgecolor="black", label="DUKE – Radiomiche"),
    mpatches.Patch(facecolor=colors["DUKE_les"], edgecolor="black", label="DUKE – Lesioni"),
]

plt.legend(
    handles=legend_handles,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False
)


plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()
