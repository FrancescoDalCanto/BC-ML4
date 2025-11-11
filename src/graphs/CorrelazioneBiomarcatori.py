import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CLEANED_FILE = Path('dataset/cleaned/medsam_dynamic.csv')
OUTPUT_DIR = Path('report/report_biomarcatori/correlazione_matrice')


def correlazione_biomarcatori():
    df = pd.read_csv(CLEANED_FILE)

    # Seleziono solo i biomarcatori che mi interessano
    biomarkers = ['GRADE', 'ER [SII]', 'PR [SII]', 'HER2 [SII]', 'isTN', 'KI67 [%]']


    # Seleziono solo le colonne che mi interessano
    df = df[biomarkers].copy()


    # ==============
    # Tipo: Spearman
    # ==============
    # Defiinisco la matrice di correlazione
    matrice_correlazione = df.corr(method='spearman')

    # Mostro la heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrice_correlazione, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                vmin=-1, vmax=1,
                square=True,
                linewidths=0.5)
    plt.title('Matrice di Correlazione dei Biomarker')
    plt.tight_layout()

    output_path = OUTPUT_DIR / 'matrice_correlazione_Sperman.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')



    # ==============
    # Tipo: Pearson
    # ==============
    # Defiinisco la matrice di correlazione
    matrice_correlazione = df.corr(method='pearson')

    # Mostro la heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrice_correlazione, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                vmin=-1, vmax=1,
                square=True,
                linewidths=0.5)
    plt.title('Matrice di Correlazione dei Biomarker')
    plt.tight_layout()

    output_path = OUTPUT_DIR / 'matrice_correlazione_Person.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')




     # ==============
    # Tipo: Kendall
    # ==============
    # Defiinisco la matrice di correlazione
    matrice_correlazione = df.corr(method='kendall')

    # Mostro la heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrice_correlazione, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                vmin=-1, vmax=1,
                square=True,
                linewidths=0.5)
    plt.title('Matrice di Correlazione dei Biomarker')
    plt.tight_layout()

    output_path = OUTPUT_DIR / 'matrice_correlazione_Kendall.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')



    plt.close()


if __name__ == "__main__":
    correlazione_biomarcatori()