import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Ordine modelli (come richiesto)
models = ["BoostedDT", "RandomForest", "LogReg", "XGBoost"]

# AUC per biomarcatore [ER, PR, HER2]
# Dataset 1: duke_lesions_radiomic
auc_radiomic = {
    "BoostedDT":     [0.532, 0.500, 0.437],
    "RandomForest":  [0.515, 0.485, 0.463],
    "LogReg":        [0.519, 0.505, 0.544],
    "XGBoost":       [0.528, 0.506, 0.468],
}

# Dataset 2: duke_lesions
auc_duke = {
    "BoostedDT":     [0.540, 0.507, 0.409],
    "RandomForest":  [0.530, 0.539, 0.454],
    "LogReg":        [0.528, 0.513, 0.532],
    "XGBoost":       [0.549, 0.512, 0.431],
}

data_radiomic = [auc_radiomic[m] for m in models]
data_duke = [auc_duke[m] for m in models]

# Posizioni: due boxplot per modello (affiancati)
x = np.arange(1, len(models) + 1)
offset = 0.18

plt.figure(figsize=(9, 5))

bp1 = plt.boxplot(
    data_radiomic,
    positions=x - offset,
    widths=0.30,
    patch_artist=True,
    showfliers=True
)

bp2 = plt.boxplot(
    data_duke,
    positions=x + offset,
    widths=0.30,
    patch_artist=True,
    showfliers=True
)

# Colora i box (senza impazzire con lo stile)
for b in bp1["boxes"]:
    b.set_facecolor("white")
for b in bp2["boxes"]:
    b.set_facecolor("lightgray")

plt.xticks(x, models, rotation=0)
plt.ylabel("AUC-ROC")
plt.title("Confronto tra modelli – Distribuzione AUC (ER, PR, HER2)")

# Limiti asse Y: puoi modificarli se vuoi zoomare
plt.ylim(0.38, 0.60)

# Legenda
legend_handles = [
    mpatches.Patch(facecolor="white", edgecolor="black", label="duke lesions radiomic"),
    mpatches.Patch(facecolor="lightgray", edgecolor="black", label="duke lesions"),
]
plt.legend(handles=legend_handles, loc="lower right")

plt.tight_layout()
plt.savefig("report/boxplot/boxplot_auc_duke_radiomic_vs_duke.png", dpi=300)
plt.show()
