import pandas as pd
from pathlib import Path
import numpy as np

from colorama import Fore, init
# Resetto il colore dopo ogni print
init(autoreset=True)

# Percorso della CARTELLA dataset
DATASET_PATH = Path('/Users/francesco/Tesi/BC-ML4/dataset')
# Percorso della cartella cleaned
PATH_CLEANED_DATASET = DATASET_PATH / 'cleaned'

# Lista dei CSV da pulire
FILENAME = [
    'medsam_dynamic.csv',
    'original_dynamic.csv',
    'preprocessed_dynamic.csv',
    't2_medsam_masks.csv',
    't2_original_masks.csv',
    't2_preprocessed_masks.csv'
]

# Colonne da rimuovere
colonne_da_rimuovere = [
    'Unnamed: 0',
    'Registered Ax T2 FSE path',
    'Roi path',
    'Slice Location',
    'Pixel array',
    'z_indexes',
    'y_indexes',
    'x_indexes',
    'Roi mask Filepath',
    'Cleaned Roi mask Filepath',
    'Registered AX Sen Vibrant MultiPhase path',
    'TemporalResolution 1',
    'TemporalResolution 2',
    'TemporalResolution 3',
    'TemporalResolution 4',
    'TemporalResolution 5',
    'ER [%]',       
    'PR [%]',       
    'HER2 [%]',     
    'ER [SII]',     
    'PR [SII]',     
    'HER2 [SII]',  
    'GRADE',        
    'KI67 [%]'  
]


# **************************************
#   Funzione per standardizzare il ki-67
# **************************************
def standardizzazione_ki67(val):
    # Se il valore è NaN, lo ritorna come NaN
    if pd.isna(val):
        return np.nan

    # Converto a stringa e rimuovo spazi bianchi
    val_str = str(val).lower().strip()

    # Se il valore è vuoto, -1, o "nan", lo ritorna come NaN
    if val_str in ['-1', '', 'nan', 'none']:
        return np.nan

    # ===============================
    # Gestisce valori con intervalli 
    # ===============================
    if 'to' in val_str:
        # Provo a dividere per numeri e calcolo la media
        try:
            parts = val_str.split('to')
            num1 = float(parts[0].strip())
            num2 = float(parts[1].strip())
            return (num1 + num2) / 2
        except:
            pass

        # Se fallisce la conversione numerica, provo con le categorie
        if 'low' in val_str and 'intermediate' in val_str:
            return 20
        elif 'intermediate' in val_str and 'high' in val_str:
            return 36.5
        elif 'low' in val_str and 'high' in val_str:
            return 32.5

    # ============================================
    # Gestisce valori categorici (low, intermediate, high)
    # ============================================
    if 'low' in val_str:
        return 15      # < 15%
    elif 'intermediate' in val_str:
        return 23      # 16-30%
    elif 'high' in val_str:
        return 50      # > 30%

    # ============================================
    # Gestisce valori numerici puri
    # ============================================
    try:
        num = float(val_str)
        # Accetta solo valori nel range 0-100
        if 0 <= num <= 100:
            return num
        else:
            return np.nan
    except:
        # Ritorna NaN se non viene convertito
        return np.nan


# *****************************************************
#   Funzione per standardizzare IHC (Immunoistochimica)
# *****************************************************
def standardizzazione_IHC(val):
    """
        Mapping:
        0 = Negativo (neg)
        1 = Debole (weak, pos generici)
        2 = Moderato (moderate)
        3 = Forte (strong)
    """

    # Se il valore è NaN, lo ritorno NaN
    if pd.isna(val):
        return np.nan

    # Converto a stringa e normalizzo (minuscolo e senza spazi)
    val_str = str(val).lower().strip()

    # Se il valore è vuoto, -1, o "nan", lo ritorno NaN
    if val_str in ['-1', '', 'nan', 'none']:
        return np.nan

    # ====================
    # MAPPING categorico
    # ====================
    # Negativo
    if 'neg' in val_str or val_str == '0':
        return 0

    # Debole / Positivo generico
    elif 'weak' in val_str or val_str == '1' or 'pos' in val_str:
        return 1

    # Moderato
    elif 'moderate' in val_str or val_str == '2':
        return 2

    # Forte
    elif 'strong' in val_str or val_str == '3':
        return 3

    # ===============================================
    # Se non è una stringa categorica, prova numerico
    # ===============================================
    try:
        num = float(val_str)
        # Accetta solo valori nel range 0-3
        if 0 <= num <= 3:
            return num
        else:
            return np.nan
    except:
        # Se non riesce a convertire, ritorna NaN
        return np.nan


# **************************************
#   Funzione per la pulire il dataset
# **************************************
def clean_dataset(dataset):
    """
    Vado a rimuovere le colonne che non mi servono     
    """
    dataset = dataset.drop(columns=[col for col in colonne_da_rimuovere if col in dataset.columns])

    """
    Vago a standardizzare ki-67[%] da categorico/numerico
    """
    if 'KI67 [%]' in dataset.columns:
        # Applica la funzione che gestisce tutti questi casi
        dataset['KI67 [%]'] = dataset['KI67 [%]'].apply(standardizzazione_ki67)

    """
    Vago a standardizzare isTN (isTripleNegative)
    """
    if 'isTN' in dataset.columns:

        # Converto da obj a numerico
        # isTN è una colonna binaria: 0 o 1
        dataset['isTN'] = pd.to_numeric(dataset['isTN'], errors='coerce')

        # Sostituisco i valori mancanti (NaN) con 0
        dataset['isTN'] = dataset['isTN'].fillna(0)

        # Converto il valore in binario: 0 o 1
        # Qualsiasi valore != 1 diventa 0
        dataset['isTN'] = (dataset['isTN'] == 1).astype(int)

    return dataset


# **************************************
#   Lettura di tutti i file
# **************************************
def process_all_file():
    try:
        for nome_file in FILENAME:
            try:
                # Percorso del file originale
                RAW_PATH_DATASET = DATASET_PATH / nome_file
                # File di output
                OUTPUT_FILE = PATH_CLEANED_DATASET / nome_file

                print(Fore.BLUE + f"\n{'='*80}")
                print(Fore.BLUE + f"Sto pulendo: {nome_file}")
                print(Fore.BLUE + f"{'='*80}")

                # Carico il file CSV originale
                dataset = pd.read_csv(RAW_PATH_DATASET)
                # Mi salvo la lunghezza iniziale (numero di righe)
                start_len = len(dataset)

                # Pulisco il dataset con la funzione clean_dataset
                dataset_cleaned = clean_dataset(dataset)
                # Mi salvo la lunghezza dopo la pulizia
                end_len = len(dataset_cleaned)

                # Salvo il dataset pulito nel CSV di output
                # index=False = non salva l'indice delle righe
                dataset_cleaned.to_csv(OUTPUT_FILE, index=False)

                print(Fore.MAGENTA + f"\nRiepilogo:")
                print(Fore.MAGENTA + f"  Righe: {start_len} → {end_len}")
                print(Fore.MAGENTA + f"  Colonne: {len(dataset.columns)} → {len(dataset_cleaned.columns)}")
                print(Fore.GREEN + f"{nome_file} pulizia completata")

            except Exception as e:
                print(Fore.RED + f" Errore in {nome_file}: {str(e)}\n")

    except Exception as e:
        print(Fore.RED + f"ERRORE generale: {e}")


# **************************************
#   Avvio del programma per la Pulizia
# **************************************
if __name__ == "__main__":
    process_all_file()
