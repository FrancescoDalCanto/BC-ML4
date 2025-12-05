import pandas as pd
import numpy as np
import sys
from pathlib import Path


from colorama import Fore, init
# Resetto il colore dopo ogni print
init(autoreset=True)


# Aggiungo la cartella 'src' al path di Python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from data_validation.DataValidation import DataValidation


# Percorso della CARTELLA dataset
DATASET_PATH = Path('/Users/francesco/Tesi/BC-ML4/dataset')
# Percorso della cartella cleaned
PATH_CLEANED_DATASET = DATASET_PATH / 'cleaned'


# Lista dei CSV da pulire
"""
FILENAME = [
    'medsam_dynamic.csv',
    'original_dynamic.csv',
    'preprocessed_dynamic.csv',
    't2_medsam_masks.csv',
    't2_original_masks.csv',
    't2_preprocessed_masks.csv'
]
"""

FILENAME = [
    'ambl_lesions.csv',
    'duke_lesions.csv'
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



# ***********************************
#   Funzione per standardizzare GRADE
# ***********************************
def standardizzazione_grade(val):
    """
        Standardizza i valori GRADE istologico:
            - Gestisce -1 e -1.0 come NaN (valori mancanti)
            - Gestisce intervalli come "1 to 2", "2 to 3", "2 to3" calcolando la media
            - Converte valori numerici validi (1, 2, 3)
            - Tutto il resto diventa NaN
    """
    # Se il valore è NaN, lo ritorna come NaN
    if pd.isna(val):
        return np.nan
    
    # Converto a stringa e rimuovo spazi bianchi
    val_str = str(val).lower().strip()
    
    # Se il valore è vuoto, -1, -1.0 o "nan", lo ritorna come NaN
    if val_str in ['-1', '-1.0', '', 'nan', 'none']:
        return np.nan
    
    # ===============================
    # Gestisce valori con intervalli 
    # ===============================
    if 'to' in val_str:
        try:
            # Rimuovo eventuali spazi e splitta per "to"
            parts = val_str.replace(' ', '').split('to')
            num1 = float(parts[0])
            num2 = float(parts[1])
            
            # Verifico che siano valori validi (1-3)
            if 1 <= num1 <= 3 and 1 <= num2 <= 3:
                # Ritorno la media dell'intervallo
                return (num1 + num2) / 2
            else:
                return np.nan
        except:
            return np.nan
    
    # ============================================
    # Gestisce valori numerici puri
    # ============================================
    try:
        num = float(val_str)
        # Accetta solo valori nel range 1-3
        if num in [1, 2, 3, 1.0, 2.0, 3.0]:
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
        Standardizza i valori IHC (ER, PR, HER2) in una scala 0-3:
        0 = Negativo (neg)
        1 = Debole (weak, weak to moderate)
        2 = Moderato (moderate, Moderate to strong)
        3 = Forte (strong, pos)
    """
    # Se il valore è NaN, lo ritorna come NaN
    if pd.isna(val):
        return np.nan
    
    # Converto a stringa, normalizzo (minuscolo, rimuovo spazi)
    val_str = str(val).lower().strip()
    
    # Se il valore è vuoto, -1, -1.0 o "nan", lo ritorna come NaN
    if val_str in ['-1', '-1.0', '', 'nan', 'none']:
        return np.nan
    
    # ===============================================
    # PRIMO: Gestisci i valori categorici con intervalli
    # ===============================================
    if 'weak' in val_str and 'moderate' in val_str:
        return 1.5
    
    # "moderate to strong" oppure "moderate strong" → 2.5 (medio tra moderate e strong)
    if 'moderate' in val_str and 'strong' in val_str:
        return 2.5
    
    # "weak to strong" → 2 (caso raro)
    if 'weak' in val_str and 'strong' in val_str:
        return 2
    
    # ===============================================
    # SECONDO: Gestisci i singoli valori categorici
    # ===============================================
    if 'neg' in val_str or val_str == '0':
        return 0
    
    # Debole / Positivo generico (ma non strong né moderate)
    elif 'weak' in val_str or val_str == '1':
        return 1
    elif val_str == 'pos' and 'moderate' not in val_str and 'strong' not in val_str:
        return 1
    
    # Moderato
    elif 'moderate' in val_str or val_str == '2':
        return 2
    
    # Forte / Positivo specifico
    elif 'strong' in val_str or val_str == '3':
        return 3
    elif val_str == 'pos':
        return 3
    
    # ===============================================
    # TERZO: Gestisci i valori numerici decimali
    # ===============================================
    try:
        num = float(val_str)
        
        # Mapping dei valori decimali sulla scala 0-3
        # Valori < 0.5 → 0 (negativo)
        if num < 0.5:
            return 0
        # Valori 0.5-1.5 → 1 (debole)
        elif num < 1.5:
            return 1
        # Valori 1.5-2.5 → 2 (moderato)
        elif num < 2.5:
            return 2
        # Valori >= 2.5 → 3 (forte)
        else:
            return 3
    except:
        # Se non riesce a convertire, ritorna NaN
        return np.nan



# **************************************
#   Funzione per la pulire il dataset
# **************************************
def clean_dataset(dataset):
    """
    In questa funzione mi occupo di:
    - uniformare i nomi delle colonne legate ai biomarcatori (ER, PR, HER2)
    - rimuovere le colonne che non mi servono
    - filtrare eventuali casi benigni (se la colonna tumor/benign è presente)
    - standardizzare GRADE, KI67 e i marcatori IHC
    - ripulire e binarizzare isTN
    - normalizzare il formato del Patient ID per far passare i controlli di formato
    """

    # Rendo coerenti i nomi delle colonne per i biomarcatori
    rename = {}
    if "ER [SII]" not in dataset.columns and "ER" in dataset.columns:
        rename["ER"] = "ER [SII]"
    if "PR [SII]" not in dataset.columns and "PR" in dataset.columns:
        rename["PR"] = "PR [SII]"
    if "HER2 [SII]" not in dataset.columns and "HER2" in dataset.columns:
        rename["HER2"] = "HER2 [SII]"

    if rename:
        dataset = dataset.rename(columns=rename)

    # Rimuovo tutte le colonne che ho marcato come non utili
    dataset = dataset.drop(columns=[col for col in colonne_da_rimuovere if col in dataset.columns])

    # Se ho la colonna tumor/benign, rimuovo le lesioni benigne (0)
    # Per DUKE la colonna non c'è ed è corretto così (solo lesioni maligne)
    if "tumor/benign" in dataset.columns:
        dataset = dataset.drop(dataset[dataset["tumor/benign"] == 0].index, axis=0)

    # Standardizzo il GRADE istologico, se presente
    if 'GRADE' in dataset.columns:
        dataset['GRADE'] = dataset['GRADE'].apply(standardizzazione_grade)

    # Standardizzo KI67 [%], se presente
    if 'KI67 [%]' in dataset.columns:
        dataset['KI67 [%]'] = dataset['KI67 [%]'].apply(standardizzazione_ki67)

    # Standardizzo i marcatori biologici IHC, se presenti
    if 'ER [SII]' in dataset.columns:
        dataset['ER [SII]'] = dataset['ER [SII]'].apply(standardizzazione_IHC)

    if 'PR [SII]' in dataset.columns:
        dataset['PR [SII]'] = dataset['PR [SII]'].apply(standardizzazione_IHC)

    if 'HER2 [SII]' in dataset.columns:
        dataset['HER2 [SII]'] = dataset['HER2 [SII]'].apply(standardizzazione_IHC)

    # Imputo i valori mancanti dei marcatori IHC con la mediana
    for bio in ['ER [SII]', 'PR [SII]', 'HER2 [SII]']:
        if bio in dataset.columns:
            mediana = dataset[bio].median(skipna=True)
            if not pd.isna(mediana):
                dataset[bio] = dataset[bio].fillna(mediana)

    # Standardizzo isTN (isTripleNegative), se presente
    if 'isTN' in dataset.columns:
        dataset['isTN'] = pd.to_numeric(dataset['isTN'], errors='coerce')
        dataset['isTN'] = dataset['isTN'].fillna(0)
        dataset['isTN'] = (dataset['isTN'] == 1).astype(int)

    # Normalizzo il formato del Patient ID, se presente
    if 'Patient ID' in dataset.columns:
        # Porto tutto a stringa ripulita
        pid = dataset['Patient ID'].astype(str).str.strip()

        # Per i casi tipo "291", "291.0", "305.00" li porto a intero e poi a stringa pura
        mask_floatlike = pid.str.match(r'^\d+(\.0+)?$')
        if mask_floatlike.any():
            pid_loc = pid[mask_floatlike].astype(float).astype(int).astype(str)
            pid.loc[mask_floatlike] = pid_loc

        # Assegno la colonna normalizzata
        dataset['Patient ID'] = pid

    return dataset





# **************************************
#   Lettura di tutti i file
# **************************************
def process_all_file():
    try:
        for nome_file in FILENAME:
            try:
                # Percorso del file originale
                #RAW_PATH_DATASET = DATASET_PATH / "original" / nome_file
                RAW_PATH_DATASET = DATASET_PATH / "new_dataset" / nome_file
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
                


                # Verifico il DataSet pulito
                checks = DataValidation(dataset_cleaned, nome_file)



                # Salvo il dataset pulito nel CSV di output
                # index=False = non salva l'indice delle righe
                dataset_cleaned.to_csv(OUTPUT_FILE, index=False)


                print(Fore.MAGENTA + f"\nRiepilogo:")
                print(Fore.MAGENTA + f"  Righe: {start_len} → {end_len}")
                print(Fore.MAGENTA + f"  Colonne: {len(dataset.columns)} → {len(dataset_cleaned.columns)}")
                print(Fore.LIGHTYELLOW_EX + f"  Numero di controlli superati con DataValidation: {checks} /5")
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
