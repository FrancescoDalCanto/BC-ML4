from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, init

init(autoreset=True)

# Percorsi
ORIGINAL_FILE = Path('dataset/original/medsam_dynamic.csv')
CLEANED_FILE = Path('dataset/cleaned/medsam_dynamic.csv')
OUTPUT_DIR_AFTER = Path('report/report_biomarcatori/after')
OUTPUT_DIR_BEFORE= Path('report/report_biomarcatori/before')

# Configurazione biomarcatori
BIOMARCATORI = {
    'ER [SII]': {'range': (0, 3)},
    'PR [SII]': {'range': (0, 3)},
    'HER2 [SII]': {'range': (0, 3)},
    'GRADE': {'range': (1, 3)},  # 0 INVALIDO
    'isTN': {'range': (0, 1)},
    'KI67 [%]': {'range': (0, 100)}
}

# ========================
# ANALIZZA E PLOTTA BEFORE
# ========================
def plot_before_biomarkers(original_file, output_dir):    
    df_before = pd.read_csv(original_file)
    
    print(Fore.BLUE + "\n" + "="*70)
    print(Fore.BLUE + "GENERAZIONE GRAFICI BEFORE (DATASET SPORCO)")
    print(Fore.BLUE + "="*70 + "\n")
    
    for bio, config in BIOMARCATORI.items():
        data_before = pd.to_numeric(df_before[bio], errors='coerce')
        min_val, max_val = config['range']
        
        # Calcoli
        if bio == 'GRADE':
            validi = ((data_before >= min_val) & (data_before <= max_val)).sum()
            invalidi = ((data_before == 0) | (data_before < min_val) | (data_before > max_val)).sum()
            mancanti = data_before.isna().sum()
        else:
            validi = data_before.notna().sum()
            invalidi = 0
            mancanti = data_before.isna().sum()
        
        # Crea figura
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if bio == 'GRADE':
            categorie = ['VALIDI\n(1-3)', 'INVALIDI\n(0 o out-of-range)', 'MANCANTI']
            valori = [validi, invalidi, mancanti]
            colori = ['#f39c12', '#e74c3c', '#95a5a6']
        else:
            categorie = ['VALIDI', 'MANCANTI']
            valori = [validi, mancanti]
            colori = ['#f39c12', '#e74c3c']
        
        bars = ax.bar(categorie, valori, color=colori, alpha=0.8, edgecolor='black', linewidth=2, width=0.6)
        
        ax.set_ylabel('Numero Campioni', fontsize=12, fontweight='bold')
        ax.set_title(f'{bio} - BEFORE (Dataset Sporco)', fontsize=14, fontweight='bold', pad=20)
        ax.grid(alpha=0.3, axis='y')
        
        # Etichette
        for bar, val in zip(bars, valori):
            height = bar.get_height()
            pct = (val / len(df_before) * 100)
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.axhline(y=len(df_before)*0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.4, label='50% del totale')
        ax.legend(loc='upper right', fontsize=10)
        ax.set_ylim(0, len(df_before) * 1.15)
        
        plt.tight_layout()
        
        # Salva
        filename = f'{bio.replace(" ", "_").replace("[", "").replace("]", "")}_BEFORE.png'
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(Fore.GREEN + f"Grafico creato per: {filename}")


# =======================
# ANALIZZA E PLOTTA AFTER
# =======================
def plot_after_biomarkers(cleaned_file, output_dir):
    df_after = pd.read_csv(cleaned_file)
    
    print(Fore.BLUE + "\n" + "="*70)
    print(Fore.BLUE + "GENERAZIONE GRAFICI AFTER (DATASET STANDARDIZZATO)")
    print(Fore.BLUE + "="*70 + "\n")
    
    for bio, config in BIOMARCATORI.items():
        data_after = pd.to_numeric(df_after[bio], errors='coerce')
        min_val, max_val = config['range']
        
        # Calcoli
        if bio == 'GRADE':
            validi = ((data_after >= min_val) & (data_after <= max_val)).sum()
            invalidi = ((data_after == 0) | (data_after < min_val) | (data_after > max_val)).sum()
            mancanti = data_after.isna().sum()
        else:
            validi = data_after.notna().sum()
            invalidi = 0
            mancanti = data_after.isna().sum()
        
        # Crea figura
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if bio == 'GRADE':
            categorie = ['VALIDI\n(1-3)', 'INVALIDI\n(0 o out-of-range)', 'MANCANTI']
            valori = [validi, invalidi, mancanti]
            colori = ['#2ecc71', '#f39c12', '#e74c3c']
        else:
            categorie = ['VALIDI', 'MANCANTI']
            valori = [validi, mancanti]
            colori = ['#2ecc71', '#e74c3c']
        
        bars = ax.bar(categorie, valori, color=colori, alpha=0.8, edgecolor='black', linewidth=2, width=0.6)
        
        ax.set_ylabel('Numero Campioni', fontsize=12, fontweight='bold')
        ax.set_title(f'{bio} - AFTER (Dataset Standardizzato)', fontsize=14, fontweight='bold', pad=20)
        ax.grid(alpha=0.3, axis='y')
        
        # Etichette
        for bar, val in zip(bars, valori):
            height = bar.get_height()
            pct = (val / len(df_after) * 100)
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.axhline(y=len(df_after)*0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.4, label='50% del totale')
        ax.legend(loc='upper right', fontsize=10)
        ax.set_ylim(0, len(df_after) * 1.15)
        
        plt.tight_layout()
        
        # Salva
        filename = f'{bio.replace(" ", "_").replace("[", "").replace("]", "")}_AFTER.png'
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(Fore.GREEN + f"Grafico Creato per: {filename}")


if __name__ == "__main__":
    
    # Per generare grafici BEFORE
    plot_before_biomarkers(ORIGINAL_FILE, OUTPUT_DIR_BEFORE)
    
    # Per generare grafici AFTER
    plot_after_biomarkers(CLEANED_FILE, OUTPUT_DIR_AFTER)
