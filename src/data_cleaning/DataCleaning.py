import pandas as pd
from pathlib import Path
import numpy as np

from colorama import Fore, init
# Resetto il colore dopo ogni print
init(autoreset=True)

from ValidationDataset import validation

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
        'TemporalResolution 5'
        ]


# **************************************
#   Funzione per standardizzare il ki-67
# **************************************
def standardizzazione_ki67(val):
    if pd.isna(val):
        return np.nan
    
    # Converti a stringa e normalizza
    val_str = str(val).lower().strip()
    
    # Gestisci valori vuoti o -1
    if val_str in ['-1', '', 'nan', 'none']:
        return np.nan
    
    """
        Gestisce valori misti
    """
    if 'to' in val_str:
        # Intervalli numerici: "20 to 25" lo imposto come  media
        try:
            parts = val_str.split('to')
            num1 = float(parts[0].strip())
            num2 = float(parts[1].strip())
            return (num1 + num2) / 2
        except:
            pass
        
        # Intervalli categorici: "low to intermediate"
        if 'low' in val_str and 'intermediate' in val_str:
            return 20  # Media(15, 25) = 20
        elif 'intermediate' in val_str and 'high' in val_str:
            return 36.5  # Media(23, 50) = 36.5
        elif 'low' in val_str and 'high' in val_str:
            return 32.5  # Media(15, 50) = 32.5
    
    """
        Gestisce una stringa
    """
    if 'low' in val_str:
        return 15  # < 15%
    elif 'intermediate' in val_str:
        return 23  # 16-30%
    elif 'high' in val_str:
        return 50  # > 30%
    
    """
        Valori numerici
    """
    try:
        num = float(val_str)
        # Accetta valori 0-100
        if 0 <= num <= 100:
            return num
        else:
            return np.nan
    except:
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
    

    """
        Vado a standardizzare ki-67[%] da categorico/numerico
        
    """
    if 'KI67 [%]' in dataset.columns:
        dataset['KI67 [%]'] = dataset['KI67 [%]'].apply(standardizzazione_ki67)
        
    
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
                OUTPUT_FILE  = PATH_CLEANED_DATASET / nome_file

                print(Fore.BLUE + f"Sto pulendo: {nome_file}\n")

                # Carico il file 
                dataset = pd.read_csv(RAW_PATH_DATASET)
                #Terminale:  mi salvo la lunghezza iniziale
                start_len = len(dataset)


                # Pulisco il dataset
                dataset_cleaned = clean_dataset(dataset)
                #Terminale: mi salvo la lunghezza del tataset pulito
                end_len = len(dataset)

                
                # Validazione
                checks = validation(dataset_cleaned, nome_file)


                # Salvo il dataset nella cartella
                dataset_cleaned.to_csv(OUTPUT_FILE, index=False)

                #Terminale: vedo se le lunghezze corrispondono e se non ho perso nulla
                print(Fore.MAGENTA + f"\n{start_len} -> {end_len}")
                print(Fore.YELLOW + f"{nome_file} pulizia completata")
                print(Fore.LIGHTGREEN_EX + f"{nome_file} completato ({checks}/6 check)\n")
                print("****************************************************************************************")

            except Exception as e:
                print(Fore.RED + f"Errore in {nome_file}: {str(e)[:60]}\n")
    except Exception as e:
        print(Fore.RED + f"ERRORE generale: {e}")


# **************************************
#   Avvio del programma per la Pulizia
# **************************************
if __name__ == "__main__":
    process_all_file()