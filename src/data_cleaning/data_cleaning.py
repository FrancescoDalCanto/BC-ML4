import pandas as pd
from pathlib import Path
import numpy as np


from colorama import Fore, init
# Resetto il colore dopo ogni print
init(autoreset=True)



# Percorso della CARTELLA dataset
DATASET_DIR = Path('/Users/francesco/Tesi/BC-ML4/dataset')
# Nome del file
FILENAME = 't2_preprocessed_masks.csv'
# Percorso del file originale
PATH_RAW_DATASET = DATASET_DIR / FILENAME
# Percorso della cartella cleaned
PATH_CLEANED_DATASET = DATASET_DIR / 'cleaned'
# File di output
OUTPUT_FILE = PATH_CLEANED_DATASET / FILENAME


def standardizzazione_ki67(val):
    if pd.isna(val):
        return np.nan
    
    # Se è stringa
    if isinstance(val, str):
        val_lower = val.lower().strip()
        if 'high' in val_lower:
            # >30%
            return 50  
        elif 'intermediate' in val_lower:
            # 16-30%
            return 23  
        elif 'low' in val_lower:
            # <15%
            return 15  
        else:
            return np.nan
    
    # Se è numerico
    try:
        num = float(val)
        if num == -1:
            return np.nan
        return num if 0 <= num <= 100 else np.nan
    except:
        return np.nan

def clean_dataset(dataset):

    """
        Vado a standardizzare ER[SII], PR[SII] e HER2[SII]
    """
    for index in ['ER [SII]', 'PR [SII]', 'HER2 [SII]']:
        if index in dataset.columns:
            # Converto da obj a numerico
            dataset[index] = pd.to_numeric(dataset[index], errors = 'coerce')

            # Rimpiazzo -1 con Nan
            dataset[index] = dataset[index].replace(-1, np.nan)

            # Limito il valore tra 0-3
            dataset[index] = dataset[index].clip(0,3)

    print(Fore.GREEN + f"ER/PR/HER2 [SII] standardizzati (0-3)")


    """
        Vado a standardizzare ER[%], PR[%] e HER2 [%]
    """
    for index in ['ER [%]', 'PR [%]', 'HER2 [%]']:
        if index in dataset.columns:
            # Converto da obj a numerico
            dataset[index] = pd.to_numeric(dataset[index], errors = 'coerce')

            # Rimpiazzo -1 con Nan
            dataset[index] = dataset[index].replace(-1, np.nan)

            # Limito il valore tra 0-100
            dataset[index] = dataset[index].clip(0,100)

    print(Fore.GREEN + f"ER/PR/HER2 [%] standardizzati (0-100)")


    """
        Vado a standardizzare GRADE
    """
    if 'GRADE' in dataset.columns:
        # Converto da obj a numerico
        dataset['GRADE'] = pd.to_numeric(dataset['GRADE'], errors = 'coerce')
        
        # Rimpiazzo -1 con Nan
        dataset['GRADE'] = dataset['GRADE'].replace(-1, np.nan)

        # Limito il valore tra 1-3
        dataset['GRADE'] = dataset['GRADE'].clip(1,3)
    
    print(Fore.GREEN + f"GRADE standardizzato (1-3 o NaN)")


    """
        Vado a standardizzare ki-67[%] da categorico/numerico
        
    """
    if 'KI67 [%]' in dataset.columns:
        dataset['KI67 [%]'] = dataset['KI67 [%]'].apply(standardizzazione_ki67)
    
    print(Fore.GREEN + f"KI67 [%] standardizzato")
    
    
    """
        Vado a standardizzare isTN
    """
    if 'isTN' in dataset.columns:
        # Converto da obj a numerico
        dataset['isTN'] = pd.to_numeric(dataset['isTN'], errors='coerce')

        # Sostituisco i valori mancanti con 0
        dataset['isTN'] = dataset['isTN'].fillna(0)

        # Converto il valore in binario
        dataset['isTN'] = (dataset['isTN'] == 1).astype(int)
    
    print(Fore.GREEN + f"KisTN standardizzato (0 o 1)")


    """
        Validazione
    """
    bio_cols = ['ER [SII]', 'PR [SII]', 'HER2 [SII]', 'ER [%]', 'PR [%]', 'HER2 [%]', 'KI67 [%]', 'GRADE']
    
    # Se benigno, ER/PR/HER2 devono essere NaN
    benigni_mask = dataset['tumor/benign'] == 0
    dataset.loc[benigni_mask, bio_cols] = np.nan
    
    # Se triple-negative, ER/PR percentuali devono essere 0
    if 'isTN' in dataset.columns:
        tn_mask = dataset['isTN'] == 1
        dataset.loc[tn_mask, ['ER [%]', 'PR [%]']] = 0

    print(Fore.GREEN + f"Valido")


    return dataset



# Carico Dataset
dataset = pd.read_csv(PATH_RAW_DATASET)
# DEBUG: Vedo quante righe ho nel dataset sporco
print(Fore.MAGENTA + f"Dataset Iniziale: {len(dataset)} righe\n")


dataset_cleaned = clean_dataset(dataset)

# Elimino la colonna dell'indice
dataset_cleaned = dataset_cleaned.drop(columns=['Unnamed: 0'], errors='ignore')

# DEBUG: Vedo quante righe ho nel dataset pulito
print(Fore.MAGENTA + f"Dataset Finale: {len(dataset)} righe\n")

# Salvo il dataset pulito in csv 
dataset_cleaned.to_csv(OUTPUT_FILE, index=False)