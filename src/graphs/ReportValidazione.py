from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, init
import numpy as np

# Resetto il colore dopo ogni print
init(autoreset=True)

# ====================
# CONFIGURAZIONE PATHS
# ====================

BEFORE_PATH = Path('/Users/francesco/Tesi/BC-ML4/dataset/original')

AFTER_PATH = Path('/Users/francesco/Tesi/BC-ML4/dataset/cleaned')

OUTPUT_DIR = Path('/Users/francesco/Tesi/BC-ML4/report/before_vs_after_cleaning')

# DATASET_FILES contiene i nomi dei 6 file CSV che vogliamo analizzare
DATASET_FILES = [
    'medsam_dynamic.csv',
    'original_dynamic.csv',
    'preprocessed_dynamic.csv',
    't2_medsam_masks.csv',
    't2_original_masks.csv',
    't2_preprocessed_masks.csv'
]



# ====================
# FUNZIONI DI ANALISI
# ====================


def analyze_dataset(dataset, nome_file):
    """
        Questa funzione analizza un singolo dataset ed estraggo informazioni chiave.
        Conto le classi (benigne vs maligne), i missing values, e il numero di features reali
        escludendo i metadati che non sono predittive.
    """
    try:
        # Estraggo la colonna target dal dataset
        target = dataset['tumor/benign']
        
        # Conto quanti campioni appartengono a ciascuna classe
        numero_benigne = (target == 0.0).sum()
        numero_maligne = (target == 1.0).sum()
        numero_mancanti = target.isna().sum()  # Valori mancanti nel target
        
        # Definisco le colonne che sono metadati e non features predittive
        # Escludo queste dal conteggio delle features perché descrivono il campione
        metadata_colonne = ['Patient ID', 'lesion idx', 'tumor/benign', 'isTN', 
                          'Breast', 'z_offset', 'y_offset', 'x_offset', 
                          'Pixel Spacing', 'Slice Thickness', 'Unnamed: 0',
                          'GRADE', 'ER [%]', 'PR [%]', 'HER2 [%]']
        
        # Creo una lista di feature eliminando i metadati dal totale delle colonne
        feature_columns = [col for col in dataset.columns if col not in metadata_colonne]
        numero_features = len(feature_columns)
        
        # Calcolo il totale dei missing values in tutto il dataset
        total_missing = dataset.isna().sum().sum()
        # Converto il conteggio assoluto in percentuale rispetto al totale delle celle
        missing_percentage = (total_missing / (dataset.shape[0] * dataset.shape[1])) * 100
        
        # Creo un dizionario con tutte le informazioni estratte
        info = {
            'Dataset': nome_file.replace('.csv', ''),
            'Righe': len(dataset),
            'Colonne_Totali': len(dataset.columns),
            'Features': numero_features,
            'Benign': numero_benigne,
            'Tumor': numero_maligne,
            'Missing_Target': numero_mancanti,
            'Missing_Total': total_missing,
            'Missing_Percentage': missing_percentage,
            # Determino se il dataset è bilanciato confrontando il numero di campioni per classe
            'Bilanciamento': 'Bilanciato' if numero_benigne == numero_maligne else 'Sbilanciato'
        }
        
        return info
    
    except Exception as e:
        print(Fore.RED + f"Errore nell'analisi di {nome_file}: {str(e)}")
        return None



def load_datasets(file_list, before_path, after_path):
    before_info = []
    after_info = []
    
    print(Fore.YELLOW + "\n" + "="*70)
    print(Fore.YELLOW + "  CARICAMENTO E ANALISI DATASET")
    print(Fore.YELLOW + "="*70 + "\n")
    
    for nome_file in file_list:
        try:
            # Leggo il file originale 
            before_file = before_path / nome_file
            if before_file.exists():
                df_before = pd.read_csv(before_file)
                info_before = analyze_dataset(df_before, nome_file)
                if info_before:
                    before_info.append(info_before)
                    print(Fore.CYAN + f"BEFORE: {nome_file} - {df_before.shape}")
            else:
                print(Fore.RED + f"File BEFORE non trovato: {before_file}")
            
            # Leggo il file pulito
            after_file = after_path / nome_file
            if after_file.exists():
                df_after = pd.read_csv(after_file)
                info_after = analyze_dataset(df_after, nome_file)
                if info_after:
                    after_info.append(info_after)
                    print(Fore.GREEN + f"AFTER:  {nome_file} - {df_after.shape}")
            else:
                print(Fore.RED + f"File AFTER non trovato: {after_file}")
                
            print()
            
        except FileNotFoundError:
            print(Fore.RED + f"File non trovato: {nome_file}\n")
        except Exception as e:
            print(Fore.RED + f"Errore in {nome_file}: {str(e)}\n")
    
    return before_info, after_info



# ================================
# FUNZIONI PER GRAFICI INDIVIDUALI
# ================================
def plot_2_numero_features(df_before, df_after, output_dir):
    """
        Grafico 2: Confronto numero di features
        
        Visualizzo quante feature predittive rimangono dopo la pulizia.
    """
    plt.figure(figsize=(12, 7))
    
    x = np.arange(len(df_before))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, df_before['Features'], width, 
                   label='BEFORE', color='#e67e22', alpha=0.8)
    bars2 = plt.bar(x + width/2, df_after['Features'], width, 
                   label='AFTER', color='#3498db', alpha=0.8)
    
    plt.xlabel('Dataset', fontweight='bold', fontsize=12)
    plt.ylabel('Numero di Features', fontweight='bold', fontsize=12)
    plt.title('Confronto: Complessità Dataset (Features)', 
             fontweight='bold', fontsize=15, pad=20)
    plt.xticks(x, df_after['Dataset'], rotation=45, ha='right', fontsize=10)
    plt.legend(loc='upper right', frameon=True, fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'grafico_2_numero_features.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(Fore.GREEN + f"Salvato: {output_path.name}")



def plot_3_bilanciamento_before(df_before, output_dir):
    """
        Grafico 3: Bilanciamento classi BEFORE
        
        Creo un istogramma che mostra la distribuzione delle due classi
        (benign vs tumor) nei dataset originali. 
    """
    plt.figure(figsize=(12, 7))
    
    x = np.arange(len(df_before))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, df_before['Benign'], width, 
                   label='Benign', color='#3498db', alpha=0.8)
    bars2 = plt.bar(x + width/2, df_before['Tumor'], width, 
                   label='Tumor', color='#e74c3c', alpha=0.8)
    
    plt.xlabel('Dataset', fontweight='bold', fontsize=12)
    plt.ylabel('Numero Campioni', fontweight='bold', fontsize=12)
    plt.title('BEFORE Cleaning: Distribuzione Classi (Tumor vs Benign)', 
             fontweight='bold', fontsize=15, pad=20)
    plt.xticks(x, df_before['Dataset'], rotation=45, ha='right', fontsize=10)
    plt.legend(loc='upper right', frameon=True, fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    
    # Aggiungo un'etichetta "Bilanciato" se le due classi hanno lo stesso numero di campioni
    for i, row in df_before.iterrows():
        if row['Benign'] == row['Tumor']:
            plt.text(i, max(row['Benign'], row['Tumor']) + 5, 
                    'Bilanciato', ha='center', fontsize=10, 
                    color='green', fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'grafico_3_bilanciamento_classi_before.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(Fore.GREEN + f"Salvato: {output_path.name}")



def plot_4_bilanciamento_after(df_after, output_dir):
    """
        Grafico 4: Bilanciamento classi AFTER
        
        Creo lo stesso tipo di grafico di plot_3, ma per i dataset puliti.
        Questo mi permette di verificare se il processo di cleaning ha migliorato
        il bilanciamento delle classi.
    """
    plt.figure(figsize=(12, 7))
    
    x = np.arange(len(df_after))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, df_after['Benign'], width, 
                   label='Benign', color='#3498db', alpha=0.8)
    bars2 = plt.bar(x + width/2, df_after['Tumor'], width, 
                   label='Tumor', color='#e74c3c', alpha=0.8)
    
    plt.xlabel('Dataset', fontweight='bold', fontsize=12)
    plt.ylabel('Numero Campioni', fontweight='bold', fontsize=12)
    plt.title('AFTER Cleaning: Distribuzione Classi (Tumor vs Benign)', 
             fontweight='bold', fontsize=15, pad=20)
    plt.xticks(x, df_after['Dataset'], rotation=45, ha='right', fontsize=10)
    plt.legend(loc='upper right', frameon=True, fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    
    for i, row in df_after.iterrows():
        if row['Benign'] == row['Tumor']:
            plt.text(i, max(row['Benign'], row['Tumor']) + 5, 
                    'Bilanciato', ha='center', fontsize=10, 
                    color='green', fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'grafico_4_bilanciamento_classi_after.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(Fore.GREEN + f"Salvato: {output_path.name}")



def plot_5_missing_values(df_before, df_after, output_dir):
    """
        Grafico 5: Missing valori BEFORE vs AFTER
        
        Creo un grafico che confronta la percentuale di missing values
        prima e dopo la pulizia. Questa è una delle metriche più importanti
        poiché il cleaning dovrebbe ridurre significativamente i dati mancanti.
    """
    plt.figure(figsize=(12, 7))
    
    x = np.arange(len(df_before))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, df_before['Missing_Percentage'], width, 
                   label='BEFORE', color='#e74c3c', alpha=0.8)
    bars2 = plt.bar(x + width/2, df_after['Missing_Percentage'], width, 
                   label='AFTER', color='#2ecc71', alpha=0.8)
    
    plt.xlabel('Dataset', fontweight='bold', fontsize=12)
    plt.ylabel('Missing Values (%)', fontweight='bold', fontsize=12)
    plt.title('Confronto: Percentuale Missing Values', 
             fontweight='bold', fontsize=15, pad=20)
    plt.xticks(x, df_before['Dataset'], rotation=45, ha='right', fontsize=10)
    plt.legend(loc='upper right', frameon=True, fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'grafico_5_missing_values.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(Fore.GREEN + f"Salvato: {output_path.name}")


# ============================================
# FUNZIONE PRINCIPALE: GENERA TUTTI I GRAFICI
# ============================================


def generate_all_plots(before_info, after_info, output_dir):
    """
        Chiamo sequenzialmente tutte le 6 funzioni di plotting.
        Converto prima i dati in DataFrame per facilitare l'accesso ai dati,
        poi configuro lo stile generale di seaborn e matplotlib.
        Infine genero i grafici e stampo un sommario dei cambiamenti.
    """
    try:
        # Converto le liste di dizionari in DataFrame pandas per un accesso più semplice
        df_before = pd.DataFrame(before_info)
        df_after = pd.DataFrame(after_info)
        
        # Configuro lo stile visuale globale per tutti i grafici
        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = 10

        
        print(Fore.YELLOW + "\n" + "="*70)
        print(Fore.YELLOW + "  GENERAZIONE GRAFICI PNG SEPARATI")
        print(Fore.YELLOW + "="*70 + "\n")
        
        # Chiamo tutte e 6 le funzioni di plotting nell'ordine stabilito
        plot_2_numero_features(df_before, df_after, output_dir)
        plot_3_bilanciamento_before(df_before, output_dir)
        plot_4_bilanciamento_after(df_after, output_dir)
        plot_5_missing_values(df_before, df_after, output_dir)
        
        print(Fore.GREEN + "\n" + "="*70)
        print(Fore.GREEN + "  TUTTI I GRAFICI GENERATI CON SUCCESSO!")
        print(Fore.GREEN + "="*70)
        print(Fore.CYAN + f"\nCartella output: {output_dir}")
        print(Fore.CYAN + f"Grafici generati: 6 PNG separati")
        print(Fore.CYAN + f"Risoluzione: 300 DPI per ogni grafico")
        
        # Stampo un sommario dettagliato dei cambiamenti tra BEFORE e AFTER
        print(Fore.YELLOW + "\n" + "="*70)
        print(Fore.YELLOW + "  SOMMARIO DEI CAMBIAMENTI")
        print(Fore.YELLOW + "="*70)
        
        # Itero su ogni dataset e mostro le differenze tra BEFORE e AFTER
        for i in range(len(df_before)):
            before = df_before.iloc[i]
            after = df_after.iloc[i]
            print(Fore.WHITE + f"\n{before['Dataset']}:")
            # Uso la notazione +/- per mostrare se il valore è aumentato o diminuito
            print(Fore.CYAN + f"  Righe:      {before['Righe']} → {after['Righe']} "
                  f"({after['Righe'] - before['Righe']:+d})")
            print(Fore.CYAN + f"  Features:   {before['Features']} → {after['Features']} "
                  f"({after['Features'] - before['Features']:+d})")
            print(Fore.CYAN + f"  Missing:    {before['Missing_Percentage']:.2f}% → {after['Missing_Percentage']:.2f}% "
                  f"({after['Missing_Percentage'] - before['Missing_Percentage']:+.2f}%)")
        
    except Exception as e:
        print(Fore.RED + f"\nErrore nella generazione dei grafici: {str(e)}")
        import traceback
        print(Fore.RED + traceback.format_exc())



# =============================
# MAIN: ESECUZIONE DELLO SCRIPT
# =============================


def main():
    try:
        print(Fore.YELLOW + "\n" + "="*70)
        print(Fore.YELLOW + "  CONFRONTO BEFORE vs AFTER DATA CLEANING")
        print(Fore.YELLOW + "  (Generazione 6 grafici PNG separati)")
        print(Fore.YELLOW + "="*70)
        
        # Carico tutti i dataset e ne estraggo le statistiche
        before_info, after_info = load_datasets(DATASET_FILES, BEFORE_PATH, AFTER_PATH)
        
        # Verifico che il caricamento sia stato completato con successo
        if not before_info or not after_info:
            print(Fore.RED + "\nErrore: Nessun dataset analizzato con successo!")
            return
        
        if len(before_info) != len(after_info):
            print(Fore.YELLOW + f"\nWarning: Numero diverso di file BEFORE ({len(before_info)}) "
                  f"e AFTER ({len(after_info)})")
        
        # Genero tutti i grafici
        generate_all_plots(before_info, after_info, OUTPUT_DIR)
        
    except Exception as e:
        print(Fore.RED + f"\nERRORE GENERALE: {e}")
        import traceback
        print(Fore.RED + traceback.format_exc())



if __name__ == "__main__":
    main()
