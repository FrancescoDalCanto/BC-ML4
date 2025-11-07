"""
    Questo script mi genera un singolo grafico per ogni biomarcatore
    solo per i dati AFTER pulizia
        - GRADE
        - ER
        - PR
        - HER2
        - isTN
        - ki-67
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, init
import numpy as np

init(autoreset=True)

CLEANED_PATH = Path('/Users/francesco/Tesi/BC-ML4/dataset/cleaned')
OUTPUT_PATH = Path('/Users/francesco/Tesi/BC-ML4/report/report_biomarcatori')

DATASET_FILES = [
    'medsam_dynamic.csv',
    'original_dynamic.csv',
    'preprocessed_dynamic.csv',
    't2_medsam_masks.csv',
    't2_original_masks.csv',
    't2_preprocessed_masks.csv'
]

# ==========
# Plot GRADE
# ==========
def plot_grade(df, output_dir):
    # Definisco la colonna target
    data = df['GRADE']

    # GRADE valido deve essere tra 1 e 3
    utilizzabili = ((data >= 1) & (data <= 3)).sum()  
    invalidi = ((data == 0) | (data < 1) | (data > 3)).sum()
    mancanti = data.isna().sum()  # Conta i NaN
    totale = len(df)

    # Crea figura
    fig, ax = plt.subplots(figsize=(10, 6))

    categorie = ['UTILIZZABILI\n(1-3)', 'INVALIDI\n(0 o out-of-range)', 'MANCANTI\n(NaN)']
    valori = [utilizzabili, invalidi, mancanti]
    colori = ['#2ecc71', '#f39c12', '#e74c3c']

    # Bar chart
    bars = ax.bar(categorie, valori, color=colori, alpha=0.8, 
                edgecolor='black', linewidth=2, width=0.6)

    # Etichette
    ax.set_ylabel('Numero Campioni', fontsize=12, fontweight='bold')
    ax.set_title(f'GRADE - Disponibilità per Classificatore\n(Solo valori 1-3 sono utilizzabili)', 
                fontsize=14, fontweight='bold', pad=20)

    # Grid
    ax.grid(alpha=0.3, axis='y')

    # Modifico le barre 
    for i, (bar, val) in enumerate(zip(bars, valori)):
        height = bar.get_height()
        pct = (val / totale * 100)
        ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val}\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Linea orizzontale a 50%
    ax.axhline(y=totale*0.5, color='gray', linestyle='--', linewidth=1.5, 
            alpha=0.5, label='50% del totale')
    ax.legend(loc='upper right', fontsize=10)

    # Set limite Y
    ax.set_ylim(0, totale * 1.1)

    plt.tight_layout()

    # Salva il grafico
    output_path = output_dir / 'GRADE_disponibilita_classificatore.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return True



# =======
# Plot ER
# =======
def plot_er(df, output_dir):
    # Definisco la colonna target
    data = df['ER [SII]']

    # Calcola i valori
    disponibili = data.notna().sum()
    mancanti = data.isna().sum()
    totale = len(df)

    
    # Crea figura
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Dati per il bar chart
    categorie = ['UTILIZZABILI\n(Per Classificatore)', 'INUTILIZZABILI\n(Mancanti)']
    valori = [disponibili, mancanti]
    colori = ['#2ecc71', '#e74c3c']
    
    # Bar chart
    bars = ax.bar(categorie, valori, color=colori, alpha=0.8, 
                  edgecolor='black', linewidth=2, width=0.6)
    
    # Etichette
    ax.set_ylabel('Numero Campioni', fontsize=12, fontweight='bold')
    ax.set_title(f'ER [SII]', fontsize=14, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(alpha=0.3, axis='y')
    
    # Etichette sulle barre
    for i, (bar, val) in enumerate(zip(bars, valori)):
        height = bar.get_height()
        pct = (val / totale * 100)
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val}\n({pct:.1f}%)',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Linea orizzontale a 50%
    ax.axhline(y=totale*0.5, color='gray', linestyle='--', linewidth=1.5, 
              alpha=0.5, label='50% del totale')
    ax.legend(loc='upper right', fontsize=10)
    
    # Set limite Y
    ax.set_ylim(0, totale * 1.1)
    
    plt.tight_layout()
    
    # Salva il grafico
    output_path = output_dir / 'ER_disponibilita_classificatore.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return True


# =======
# Plot PR
# =======
def plot_pr(df, output_dir):
# Definisco la colonna target
    data = df['PR [SII]']

    # Calcola i valori
    disponibili = data.notna().sum()
    mancanti = data.isna().sum()
    totale = len(df)

    
    # Crea figura
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Dati per il bar chart
    categorie = ['UTILIZZABILI\n(Per Classificatore)', 'INUTILIZZABILI\n(Mancanti)']
    valori = [disponibili, mancanti]
    colori = ['#2ecc71', '#e74c3c']
    
    # Bar chart
    bars = ax.bar(categorie, valori, color=colori, alpha=0.8, 
                  edgecolor='black', linewidth=2, width=0.6)
    
    # Etichette
    ax.set_ylabel('Numero Campioni', fontsize=12, fontweight='bold')
    ax.set_title(f'PR [SII]', fontsize=14, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(alpha=0.3, axis='y')
    
    # Etichette sulle barre
    for i, (bar, val) in enumerate(zip(bars, valori)):
        height = bar.get_height()
        pct = (val / totale * 100)
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val}\n({pct:.1f}%)',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Linea orizzontale a 50%
    ax.axhline(y=totale*0.5, color='gray', linestyle='--', linewidth=1.5, 
              alpha=0.5, label='50% del totale')
    ax.legend(loc='upper right', fontsize=10)
    
    # Set limite Y
    ax.set_ylim(0, totale * 1.1)
    
    plt.tight_layout()
    
    # Salva il grafico
    output_path = output_dir / 'PR_disponibilita_classificatore.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return True

# =========
# Plot HER2
# =========
def plot_her2(df, output_dir):
    # Definisco la colonna target
    data = df['HER2 [SII]']

    # Calcola i valori
    disponibili = data.notna().sum()
    mancanti = data.isna().sum()
    totale = len(df)

    
    # Crea figura
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Dati per il bar chart
    categorie = ['UTILIZZABILI\n(Per Classificatore)', 'INUTILIZZABILI\n(Mancanti)']
    valori = [disponibili, mancanti]
    colori = ['#2ecc71', '#e74c3c']
    
    # Bar chart
    bars = ax.bar(categorie, valori, color=colori, alpha=0.8, 
                  edgecolor='black', linewidth=2, width=0.6)
    
    # Etichette
    ax.set_ylabel('Numero Campioni', fontsize=12, fontweight='bold')
    ax.set_title(f'HER2 [SII]', fontsize=14, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(alpha=0.3, axis='y')
    
    # Etichette sulle barre
    for i, (bar, val) in enumerate(zip(bars, valori)):
        height = bar.get_height()
        pct = (val / totale * 100)
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val}\n({pct:.1f}%)',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Linea orizzontale a 50%
    ax.axhline(y=totale*0.5, color='gray', linestyle='--', linewidth=1.5, 
              alpha=0.5, label='50% del totale')
    ax.legend(loc='upper right', fontsize=10)
    
    # Set limite Y
    ax.set_ylim(0, totale * 1.1)
    
    plt.tight_layout()
    
    # Salva il grafico
    output_path = output_dir / 'HER2_disponibilita_classificatore.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return True

# =========
# Plot isTN
# =========
def plot_istn(df, output_dir):
    # Definisco la colonna target
    data = df['isTN']

    # Calcola i valori
    disponibili = data.notna().sum()
    mancanti = data.isna().sum()
    totale = len(df)

    
    # Crea figura
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Dati per il bar chart
    categorie = ['UTILIZZABILI\n(Per Classificatore)', 'INUTILIZZABILI\n(Mancanti)']
    valori = [disponibili, mancanti]
    colori = ['#2ecc71', '#e74c3c']
    
    # Bar chart
    bars = ax.bar(categorie, valori, color=colori, alpha=0.8, 
                  edgecolor='black', linewidth=2, width=0.6)
    
    # Etichette
    ax.set_ylabel('Numero Campioni', fontsize=12, fontweight='bold')
    ax.set_title(f'isTN', fontsize=14, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(alpha=0.3, axis='y')
    
    # Etichette sulle barre
    for i, (bar, val) in enumerate(zip(bars, valori)):
        height = bar.get_height()
        pct = (val / totale * 100)
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val}\n({pct:.1f}%)',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Linea orizzontale a 50%
    ax.axhline(y=totale*0.5, color='gray', linestyle='--', linewidth=1.5, 
              alpha=0.5, label='50% del totale')
    ax.legend(loc='upper right', fontsize=10)
    
    # Set limite Y
    ax.set_ylim(0, totale * 1.1)
    
    plt.tight_layout()
    
    # Salva il grafico
    output_path = output_dir / 'isTN_disponibilita_classificatore.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return True

# ==========
# Plot KI67
# ==========
def plot_ki67(df, output_dir):

    # Definisco la colonna target
    data = df['KI67 [%]']

    # Calcola i valori
    disponibili = data.notna().sum()
    mancanti = data.isna().sum()
    totale = len(df)

    
    # Crea figura
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Dati per il bar chart
    categorie = ['UTILIZZABILI\n(Per Classificatore)', 'INUTILIZZABILI\n(Mancanti)']
    valori = [disponibili, mancanti]
    colori = ['#2ecc71', '#e74c3c']
    
    # Bar chart
    bars = ax.bar(categorie, valori, color=colori, alpha=0.8, 
                  edgecolor='black', linewidth=2, width=0.6)
    
    # Etichette
    ax.set_ylabel('Numero Campioni', fontsize=12, fontweight='bold')
    ax.set_title(f'KI67 [%]', fontsize=14, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(alpha=0.3, axis='y')
    
    # Etichette sulle barre
    for i, (bar, val) in enumerate(zip(bars, valori)):
        height = bar.get_height()
        pct = (val / totale * 100)
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val}\n({pct:.1f}%)',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Linea orizzontale a 50%
    ax.axhline(y=totale*0.5, color='gray', linestyle='--', linewidth=1.5, 
              alpha=0.5, label='50% del totale')
    ax.legend(loc='upper right', fontsize=10)
    
    # Set limite Y
    ax.set_ylim(0, totale * 1.1)
    
    plt.tight_layout()
    
    # Salva il grafico
    output_path = output_dir / 'ki67_disponibilita_classificatore.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return True

def main():
    try:
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        
        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = 10
        
        for file_name in DATASET_FILES:
            
            try:
                file_path = CLEANED_PATH / file_name
                df = pd.read_csv(file_path)
                
                # Crea tutti i grafici
                plot_grade(df, OUTPUT_PATH)
                plot_er(df, OUTPUT_PATH)
                plot_pr(df, OUTPUT_PATH)
                plot_her2(df, OUTPUT_PATH)
                plot_istn(df, OUTPUT_PATH)
                plot_ki67(df, OUTPUT_PATH)
            
            except FileNotFoundError:
                print(Fore.RED + f"File non trovato: {file_path}")
            except Exception as e:
                print(Fore.RED + f"Errore in : {e}")
    except Exception as e:
        print(Fore.RED + f"\nERRORE GENERALE: {e}")
        import traceback
        print(Fore.RED + traceback.format_exc())


if __name__ == "__main__":
    main()
