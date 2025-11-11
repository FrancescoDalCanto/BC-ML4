from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, init



init(autoreset=True)



# Percorsi
CLEANED_FILE = Path('dataset/cleaned/medsam_dynamic.csv')
OUTPUT_DIR = Path('report/report_biomarcatori/istogrammi')




# Configurazione biomarcatori
BIOMARCATORI_CONFIG = {
    'ER [SII]': {
        'labels': {0: 'Negativo', 1: 'Positivo'},
        'colors': ['#e74c3c', '#2ecc71']
    },
    'PR [SII]': {
        'labels': {0: 'Negativo', 1: 'Positivo'},
        'colors': ['#e74c3c', '#2ecc71']
    },
    'HER2 [SII]': {
        'labels': {0: 'Negativo', 1: 'Positivo'},
        'colors': ['#e74c3c', '#2ecc71']
    },
    'GRADE': {
        'labels': {1: 'GRADE 1', 1.5: 'GRADE 1.5 (1 to 2)', 2: 'GRADE 2', 2.5:'GRADE 2.5 (2 to 3)',3: 'GRADE 3'}, 
        'colors': ['#2ecc71', "#95a0e1", '#f39c12', "#9d3ce7", '#e74c3c']  
    },
    'isTN': {
        'labels': {0: 'Non Triplo-Negativo', 1: 'Triplo-Negativo'},
        'colors': ['#3498db', '#e74c3c']
    },
    'KI67 [%]': {
        'bins': [0, 14, 30, 100],
        'labels': ['Basso (<14%)', 'Medio (14-30%)', 'Alto (>30%)'],
        'colors': ['#2ecc71', '#f39c12', '#e74c3c']
    }
}





def plot_biomarkers(cleaned_file, output_dir):


    # Vado a leggere il csv pulito
    df_after = pd.read_csv(cleaned_file)


    print(Fore.YELLOW + f"\nLesioni maligne dopo pulizia: {len(df_after)}")
    print(Fore.BLUE + "\n" + "="*70)
    print(Fore.BLUE + "GENERAZIONE GRAFICI DISTRIBUZIONE (SOLO LESIONI MALIGNE)")
    print(Fore.BLUE + "="*70 + "\n")


    for bio, config in BIOMARCATORI_CONFIG.items():
        data = pd.to_numeric(df_after[bio], errors='coerce')
        
        # Rimuovi valori mancanti
        data = data.dropna()
    
        # Crea figura
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Gestione per Ki-67
        if bio == 'KI67 [%]':
            # Discretizza in bins
            data_binned = pd.cut(data, bins=config['bins'], labels=config['labels'], include_lowest=True)
            value_counts = data_binned.value_counts().sort_index()
            categorie = config['labels']
            valori = [value_counts.get(cat, 0) for cat in categorie]
            colori = config['colors']
        else:
            if bio in ['ER [SII]', 'PR [SII]', 'HER2 [SII]']:
                # Converti tutto ≥1 in Positivo
                negativo = (data == 0.0).sum()
                positivo = (data > 0.0).sum()
                categorie = ['Negativo', 'Positivo']
                valori = [negativo, positivo]
                colori = config['colors']
            else:
                # Per GRADE e isTN uso la logica normale
                value_counts = data.value_counts().sort_index()
                # Ottieni tutti i valori presenti nei dati che hanno una label definita
                categorie = [config['labels'][val] for val in sorted(config['labels'].keys()) if val in value_counts.index]
                valori = [value_counts[val] for val in sorted(config['labels'].keys()) if val in value_counts.index]
                colori = [config['colors'][i] for i, val in enumerate(sorted(config['labels'].keys())) if val in value_counts.index]
        
        # Plot
        bars = ax.bar(categorie, valori, color=colori, alpha=0.8, edgecolor='black', linewidth=2, width=0.6)
        
        ax.set_ylabel('Numero Pazienti', fontsize=12, fontweight='bold')
        ax.set_title(f'{bio} - Distribuzione delle Lesioni Maligne', fontsize=14, fontweight='bold', pad=20)
        ax.grid(alpha=0.3, axis='y')
        
        # Aggiungo spazio per non far tagliare la riga del bordo col numero
        y_max = max(valori) * 1.15 
        ax.set_ylim(0, y_max)


        # Etichette sopra le barre
        for bar, val in zip(bars, valori):
            height = bar.get_height()
            pct = (val / len(data) * 100) if len(data) > 0 else 0
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        # Salva
        filename = f'{bio.replace(" ", "_").replace("[", "").replace("]", "")}_distribuzione.png'
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(Fore.GREEN + f" Grafico creato: {filename}")
        print(Fore.CYAN + f"  Distribuzione: {dict(zip(categorie, valori))}")




if __name__ == "__main__":
    plot_biomarkers(CLEANED_FILE, OUTPUT_DIR)
