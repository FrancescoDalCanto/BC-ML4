import pandas as pd
import numpy as np
import sys
from pathlib import Path

from colorama import Fore, init
init(autoreset=True)  # Reset automatico del colore dopo ogni print

# =========================================================
# Impostazione dei path e import dei moduli di progetto
# =========================================================

# Aggiungo la cartella 'src' al path di Python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_validation.DataValidation import DataValidation

# Percorso principale del dataset
DATASET_PATH = Path('/Users/francesco/Tesi/BC-ML4/dataset')

# Cartella di output dei dataset puliti
PATH_CLEANED_DATASET = DATASET_PATH / 'cleaned'

# Lista dei file CSV da processare
FILENAME = [
    'ambl_lesions_radiomic_medsam.csv',
    'ambl_lesions.csv',
]

# =========================================================
# Colonne da rimuovere
# =========================================================
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
]

# =========================================================
# Funzione di standardizzazione del Ki-67
# =========================================================

def standardizzazione_ki67(val):
    """
    Standardizzo il valore di Ki-67 (%), che nel dataset può essere:
    - numerico
    - categorico (low / intermediate / high)
    - un intervallo (es. '10 to 20')
    
    Quando non interpretabile, ritorno NaN.
    """

    if pd.isna(val):
        return np.nan

    val_str = str(val).lower().strip()

    if val_str in ['-1', '', 'nan', 'none']:
        return np.nan

    # Gestione degli intervalli
    if 'to' in val_str:
        try:
            num1, num2 = map(float, val_str.split('to'))
            return (num1 + num2) / 2
        except:
            # Fallback su categorie testuali
            if 'low' in val_str and 'intermediate' in val_str:
                return 20
            elif 'intermediate' in val_str and 'high' in val_str:
                return 36.5
            elif 'low' in val_str and 'high' in val_str:
                return 32.5

    # Categorie qualitative
    if 'low' in val_str:
        return 15
    elif 'intermediate' in val_str:
        return 23
    elif 'high' in val_str:
        return 50

    # Valori numerici puri
    try:
        num = float(val_str)
        return num if 0 <= num <= 100 else np.nan
    except:
        return np.nan

# =========================================================
# Funzione di standardizzazione del GRADE istologico
# =========================================================

def standardizzazione_grade(val):
    """
    Standardizzo il GRADE istologico:
    - accetto valori 1, 2, 3
    - gestisco intervalli tipo '1 to 2'
    - tutto il resto viene considerato NaN
    """

    if pd.isna(val):
        return np.nan

    val_str = str(val).lower().strip()

    if val_str in ['-1', '-1.0', '', 'nan', 'none']:
        return np.nan

    if 'to' in val_str:
        try:
            parts = val_str.replace(' ', '').split('to')
            num1, num2 = map(float, parts)
            if 1 <= num1 <= 3 and 1 <= num2 <= 3:
                return (num1 + num2) / 2
        except:
            return np.nan

    try:
        num = float(val_str)
        return num if num in [1, 2, 3] else np.nan
    except:
        return np.nan

# =========================================================
# Funzione di standardizzazione IHC (ER, PR, HER2)
# =========================================================

def standardizzazione_IHC(val):
    """
    Converto i valori immunoistochimici su una scala 0 - 3:
    0 = negativo
    1 = debole
    2 = moderato
    3 = forte
    """

    if pd.isna(val):
        return np.nan

    val_str = str(val).lower().strip()

    if val_str in ['-1', '-1.0', '', 'nan', 'none']:
        return np.nan

    # Casi con intervalli qualitativi
    if 'weak' in val_str and 'moderate' in val_str:
        return 1.5
    if 'moderate' in val_str and 'strong' in val_str:
        return 2.5
    if 'weak' in val_str and 'strong' in val_str:
        return 2

    # Categorie singole
    if 'neg' in val_str or val_str == '0':
        return 0
    if 'weak' in val_str or val_str == '1':
        return 1
    if 'moderate' in val_str or val_str == '2':
        return 2
    if 'strong' in val_str or val_str == '3' or val_str == 'pos':
        return 3

    # Fallback numerico
    try:
        num = float(val_str)
        if num < 0.5:
            return 0
        elif num < 1.5:
            return 1
        elif num < 2.5:
            return 2
        else:
            return 3
    except:
        return np.nan

# =========================================================
# Funzione principale di pulizia del dataset
# =========================================================

def clean_dataset(dataset):
    """
    Applico tutte le operazioni di pulizia:
    - rimozione colonne inutili
    - filtro sui tumori maligni
    - standardizzazione delle variabili cliniche
    """

    # Rimuovo le colonne non necessarie
    dataset = dataset.drop(
        columns=[c for c in colonne_da_rimuovere if c in dataset.columns]
    )

    # Tengo solo le lesioni maligne
    dataset = dataset.drop(dataset[dataset["tumor/benign"] == 0].index)

    # Standardizzazione GRADE
    if 'GRADE' in dataset.columns:
        dataset['GRADE'] = dataset['GRADE'].apply(standardizzazione_grade)

    # Standardizzazione Ki-67
    if 'KI67 [%]' in dataset.columns:
        dataset['KI67 [%]'] = dataset['KI67 [%]'].apply(standardizzazione_ki67)

    # Standardizzazione biomarcatori IHC
    for col in ['ER [SII]', 'PR [SII]', 'HER2 [SII]']:
        if col in dataset.columns:
            dataset[col] = dataset[col].apply(standardizzazione_IHC)

    # Standardizzazione isTN (0/1)
    if 'isTN' in dataset.columns:
        dataset['isTN'] = pd.to_numeric(dataset['isTN'], errors='coerce').fillna(0)
        dataset['isTN'] = (dataset['isTN'] == 1).astype(int)

    return dataset

# =========================================================
# Elaborazione di tutti i file
# =========================================================

def process_all_file():
    for nome_file in FILENAME:
        try:
            RAW_PATH = DATASET_PATH / "original" / nome_file
            OUTPUT_PATH = PATH_CLEANED_DATASET / nome_file

            print(Fore.BLUE + f"\n{'='*80}")
            print(Fore.BLUE + f"Sto pulendo: {nome_file}")
            print(Fore.BLUE + f"{'='*80}")

            dataset = pd.read_csv(RAW_PATH)
            start_len = len(dataset)

            dataset_cleaned = clean_dataset(dataset)
            end_len = len(dataset_cleaned)

            checks = DataValidation(dataset_cleaned, nome_file)
            dataset_cleaned.to_csv(OUTPUT_PATH, index=False)

            print(Fore.MAGENTA + "\nRiepilogo:")
            print(Fore.MAGENTA + f"  Righe: {start_len} → {end_len}")
            print(Fore.MAGENTA + f"  Colonne: {len(dataset.columns)} → {len(dataset_cleaned.columns)}")
            print(Fore.LIGHTYELLOW_EX + f"  Controlli DataValidation superati: {checks} /5")
            print(Fore.GREEN + f"{nome_file} pulizia completata")

        except Exception as e:
            print(Fore.RED + f"Errore in {nome_file}: {e}")

# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    process_all_file()
