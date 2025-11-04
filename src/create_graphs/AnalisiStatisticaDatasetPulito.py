import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, init

init(autoreset=True)


# Percorso della CARTELLA dataset 
DATASET_PATH = Path('/Users/francesco/Tesi/BC-ML4/dataset/cleaned')

# Cartella per i risultati
OUTPUT_ANALISI = Path('/Users/francesco/Tesi/BC-ML4/Graphs/analisi_statistica')

# Lista dei CSV da analizzare
FILENAME = [
    'medsam_dynamic.csv',
    'original_dynamic.csv',
    'preprocessed_dynamic.csv',
    't2_medsam_masks.csv',
    't2_original_masks.csv',
    't2_preprocessed_masks.csv'
]


# TODO: Analizzare il codice e capire cosa fa

# ********************************
#   Analisi statistica dei dataset
# ********************************
def analisi_statistica(dataset, nome_file):
    print(Fore.CYAN + f"\n{'='*80}")
    print(Fore.CYAN + f"ANALISI: {nome_file}")
    print(Fore.CYAN + f"{'='*80}\n")

    # 1. Informazioni generali
    print(Fore.YELLOW + "1. INFORMAZIONI GENERALI")
    print(f"   Righe: {len(dataset)}")
    print(f"   Colonne: {len(dataset.columns)}")
    print(f"   Memoria: {dataset.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")

    # 2. Tipi di dati
    print(Fore.YELLOW + "2. TIPI DI DATI E VALORI MANCANTI")
    print(dataset.info())
    print()

    # 3. Statistiche descrittive
    print(Fore.YELLOW + "3. STATISTICHE DESCRITTIVE")
    print(dataset.describe().T)
    print()
    
    # 4. Valori mancanti
    print(Fore.YELLOW + "4. ANALISI VALORI MANCANTI")
    missing = dataset.isnull().sum()
    missing_percent = (missing / len(dataset)) * 100
    missing_df = pd.DataFrame({
        'Colonna': missing.index,
        'Mancanti': missing.values,
        'Percentuale': missing_percent.values
    })
    missing_df = missing_df[missing_df['Mancanti'] > 0].sort_values('Percentuale', ascending=False)
    print(missing_df)
    print()

    # 5. Analisi per classe
    if 'tumor/benign' in dataset.columns:
        print(Fore.YELLOW + "5. ANALISI PER CLASSE (tumor/benign)")
        print(dataset.groupby('tumor/benign').describe().T)
        print()


# ********************************
#   Creazione del grafico
# ********************************
def graphs(dataset, nome_file, output_path):
    """
    Grafico sui valori mancanti
    """
    
    # Calcola i valori mancanti
    missing = dataset.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    # Crea figura
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if len(missing) > 0:
        # Grafico a barre orizzontali
        ax.barh(range(len(missing)), missing.values, color='coral')
        ax.set_yticks(range(len(missing)))
        ax.set_yticklabels(missing.index, fontsize=10)
        ax.set_xlabel('Numero di valori mancanti', fontsize=12, fontweight='bold')
        ax.set_title(f'Valori Mancanti - {nome_file}\nTotale record: {len(dataset)}', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3)
        
        # Trova il valore massimo per posizionare i testi
        max_val = missing.values.max()
        
        # Aggiungi i numeri e percentuali sulle barre
        for i, (col, val) in enumerate(missing.items()):
            pct = (val / len(dataset)) * 100
            # Numero a destra della barra
            ax.text(val + (max_val * 0.02), i, str(val), va='center', fontweight='bold')
            # Percentuale più spostata a destra
            ax.text(val + (max_val * 0.08), i, f'({pct:.1f}%)', va='center', fontsize=9, color='gray')
        
        # Aumenta il limite dell'asse x per dare spazio ai testi
        ax.set_xlim(0, max_val * 1.2)
        
    else:
        # Nessun valore mancante
        ax.text(0.5, 0.5, '✓ Nessun valore mancante!\nDataset completamente pulito', 
                ha='center', va='center', fontsize=20, color='green', fontweight='bold')
        ax.axis('off')
    
    fig.tight_layout()
    fig.savefig(
        output_path / f'{nome_file.replace(".csv", "")}_missing_values.png', 
        dpi=300, 
        bbox_inches='tight',
        facecolor='white'
    )
    plt.close(fig)



# ************************
#   Lettura di tutti i CSV
# ************************
def process_all_file():
    # Crea la cartella se non esiste
    OUTPUT_ANALISI.mkdir(parents=True, exist_ok=True)
    
    try:
        for nome_file in FILENAME:
            try:
                # Percorso del file
                CLEANED_PATH_DATASET = DATASET_PATH / nome_file
                
                print(Fore.BLUE + f"Sto analizzando: {nome_file}\n")
                
                # Carico il file 
                dataset = pd.read_csv(CLEANED_PATH_DATASET)
                
                # Analizzo i dati
                analisi_statistica(dataset, nome_file)
                
                # Creo il grafico
                graphs(dataset, nome_file, OUTPUT_ANALISI)
                
                print(Fore.LIGHTGREEN_EX + f"{nome_file} analizzato con successo")
                print(Fore.LIGHTGREEN_EX + f"Grafico salvato in: {OUTPUT_ANALISI}\n")
                
            except Exception as e:
                print(Fore.RED + f"Errore in {nome_file}: {str(e)}\n")
    
    except Exception as e:
        print(Fore.RED + f"ERRORE generale: {e}")


# ********************************************
#   Avvio del programma
# ********************************************
if __name__ == "__main__":
    process_all_file()
