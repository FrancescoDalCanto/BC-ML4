"""
    Questo metodo mi genera un singolo report grafico per capire come sono i miei dati
    e se posso fare il classificatore
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, init

# Resetto il colore dopo ogni print
init(autoreset=True)

# Percorso da cui prendere i CSV
DATASET_PATH = Path('/Users/francesco/Tesi/BC-ML4/dataset/cleaned')

# Percorso per salvare il report
OUTPUT_REPORT_PATH = Path('/Users/francesco/Tesi/BC-ML4/report/report_statistico/dataset_validation_report.png')

# Nomi dei CSV su cui fare il grafico
FILENAME = [
    'medsam_dynamic.csv',
    'original_dynamic.csv',
    'preprocessed_dynamic.csv',
    't2_medsam_masks.csv',
    't2_original_masks.csv',
    't2_preprocessed_masks.csv'
]

# Lista globale per raccogliere TUTTI i dati
all_datasets_info = []


# *******************
# Analisi del dataset
# *******************
def DatasetAnalyses(dataset, nome_file):
    try:
        # Estraggo informazioni sul target
        target = dataset['tumor/benign']

        # Conto il numero di classi
        numero_benigne = (target == 0.0).sum()
        numero_maligne = (target == 1.0).sum()
        numero_mancanti = target.isna().sum()

        # Escludo le colonne dei metadati
        metadata_colonne = ['Patient ID', 'lesion idx', 'tumor/benign', 'isTN', 
                         'Breast', 'z_offset', 'y_offset', 'x_offset', 
                         'Pixel Spacing', 'Slice Thickness']
        numero_features = len(dataset.columns) - len(metadata_colonne)

        # Preparo le informazioni da restituire
        info = {
            'Dataset': nome_file.replace('.csv', ''),
            'Righe': len(dataset),
            'Features': numero_features,
            'Benign': numero_benigne,
            'Tumor': numero_maligne,
            'Missing': numero_mancanti,
            'Bilanciamento': 'Buono' if numero_benigne == numero_maligne else 'Sbilanciato'
        }

        return info
    
    except Exception as e:
        print(Fore.RED + f"Errore nell'analisi: {str(e)}")
        return None


# ****************************
# Creazione del report grafico
# ****************************
def ReportValidazione(info_list):
    try:
        # Converto LISTA di dizionari in DataFrame
        dataframe = pd.DataFrame(info_list)

        # Configuro lo stile
        sns.set_style("whitegrid")

        # Crea una figura con solo 3 plot 
        fig = plt.figure(figsize=(16, 10))
        
        # Definisci layout: 2 righe, 2 colonne
        # Prima riga: 2 subplot
        # Seconda riga: 1 subplot centrato che occupa entrambe le colonne
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Dataset Validation Report - Fattibilità Classificatore', 
                     fontsize=20, fontweight='bold', y=0.98)

        # ==================================
        # PLOT 1: Bilanciamento delle classi (top-left)
        # ==================================
        ax1 = fig.add_subplot(gs[0, 0])
        x = range(len(dataframe))
        width = 0.35

        ax1.bar([i - width/2 for i in x], dataframe['Benign'], width, 
                label='Benign', color='#3498db', alpha=0.8)
        ax1.bar([i + width/2 for i in x], dataframe['Tumor'], width, 
                label='Tumor', color='#e74c3c', alpha=0.8)

        ax1.set_xlabel('Dataset', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Numero Campioni', fontweight='bold', fontsize=12)

        # Aggiusto il titolo
        ax1.set_title('Distribuzione Classi (Tumor vs Benign)', 
                    fontweight='bold', fontsize=14, pad=35)

        ax1.set_xticks(x)
        ax1.set_xticklabels(dataframe['Dataset'], rotation=45, ha='right', fontsize=10)

        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), 
                ncol=2, frameon=False, fontsize=10)

        ax1.grid(axis='y', alpha=0.3)

        # Imposta limite superiore Y per dare più spazio
        max_value = max(dataframe['Benign'].max(), dataframe['Tumor'].max())
        ax1.set_ylim(0, max_value + 15)

        # Aggiungo le scritte
        for i, row in dataframe.iterrows():
            if row['Benign'] == row['Tumor']:
                ax1.text(i, max(row['Benign'], row['Tumor']) + 5, 
                        '✓ Bilanciato', ha='center', fontsize=8, 
                        color='green', fontweight='bold')



        # ============================================
        # PLOT 2: Numero features per dataset
        # ============================================
        ax2 = fig.add_subplot(gs[0, 1])
        colors = ['#2ecc71' if 'dynamic' in name else '#9b59b6' 
                  for name in dataframe['Dataset']]
        
        bars = ax2.barh(dataframe['Dataset'], dataframe['Features'], color=colors, alpha=0.8)
        ax2.set_xlabel('Numero di Features', fontweight='bold', fontsize=12)
        ax2.set_title('Complessità Dataset (Features) (poche=buono, tante=non buono)', 
                      fontweight='bold', fontsize=14)
        ax2.grid(axis='x', alpha=0.3)
        
        # Aggiungi valori sulle barre
        for bar, val in zip(bars, dataframe['Features']):
            ax2.text(val + 2, bar.get_y() + bar.get_height()/2, 
                    f'{val}', va='center', fontsize=10, fontweight='bold')

        # ============================================
        # PLOT 3: Tabella riepilogativa
        # ============================================
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('tight')
        ax3.axis('off')
        
        table_data = dataframe[['Dataset', 'Righe', 'Features', 'Benign', 'Tumor', 'Missing']].values
        table = ax3.table(cellText=table_data, 
                         colLabels=['Dataset', 'Righe', 'Features', 'Benign', 'Tumor', 'Missing'],
                         cellLoc='center', loc='center',
                         colColours=['#ecf0f1']*6)
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Colora le righe alternate
        for i in range(1, len(table_data) + 1):
            for j in range(6):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#f8f9fa')
                else:
                    cell.set_facecolor('#ffffff')
        
        ax3.set_title('Riepilogo Dettagliato', fontweight='bold', fontsize=14, pad=15)


        # ============================================
        # Salva il grafico
        # ============================================
        plt.tight_layout()
        plt.savefig(OUTPUT_REPORT_PATH, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(Fore.GREEN + f"\nReport grafico salvato: {OUTPUT_REPORT_PATH}")
        print(Fore.GREEN + f"Risoluzione: 300 DPI | Formato: PNG")
        plt.close()

    except Exception as e:
        print(Fore.RED + f"Errore nella generazione del report: {str(e)}")
        import traceback
        print(Fore.RED + traceback.format_exc())


# *************************
#   Lettura di tutti i file
# *************************
def process_all_file():
    global all_datasets_info
    
    try:
        print(Fore.YELLOW + "\n" + "="*60)
        print(Fore.YELLOW + "  ANALISI DATASET PER CLASSIFICATORE")
        print(Fore.YELLOW + "="*60)
        
        # Loop per raccogliere TUTTI i dati
        for nome_file in FILENAME:
            try:
                PATH_TO_READ = DATASET_PATH / nome_file
                dataset = pd.read_csv(PATH_TO_READ)
                
                # Analizza dataset
                info = DatasetAnalyses(dataset, nome_file)
                
                if info:
                    all_datasets_info.append(info)
                    print(Fore.CYAN + f"{nome_file} analizzato")
                    
            except FileNotFoundError:
                print(Fore.RED + f"File non trovato: {nome_file}")
            except Exception as e:
                print(Fore.RED + f"Errore in {nome_file}: {str(e)}")
        
        # Genera report UNA SOLA VOLTA con TUTTI i dati
        if all_datasets_info:
            print(Fore.YELLOW + "\n" + "="*60)
            print(Fore.YELLOW + "  GENERAZIONE REPORT GRAFICO")
            print(Fore.YELLOW + "="*60)
            ReportValidazione(all_datasets_info)
        else:
            print(Fore.RED + "\nNessun dataset analizzato con successo")

    except Exception as e:
        print(Fore.RED + f"\n ERRORE generale: {e}")
        import traceback
        print(Fore.RED + traceback.format_exc())


# ****************************************************
#   Avvio del programma per la generazione del grafico
# ****************************************************
if __name__ == "__main__":
    process_all_file()
