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
PATH_CLEANED_DATASET = DATASET_PATH / 'cleaned/Test'

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
    Estrae il valore numerico del Ki-67 senza inventare medie arbitrarie
    per le etichette testuali (se possibile).
    """
    if pd.isna(val) or str(val).lower() in ['nan', 'none', '-1', '', '-1.0']:
        return np.nan

    val_str = str(val).lower().strip()

    # Gestione intervalli (es. '10 to 20') -> prendiamo la media dell'intervallo
    if 'to' in val_str:
        try:
            nums = [float(s.strip()) for s in val_str.split('to') if s.strip()]
            return sum(nums) / len(nums)
        except:
            pass

    # Gestione categorie qualitative: 
    # NOTA: In un paper, meglio mappare su soglie cliniche (Low < 20%, High >= 20%)
    if 'low' in val_str: return 10.0 # Valore rappresentativo basso
    if 'high' in val_str: return 40.0 # Valore rappresentativo alto
    if 'intermediate' in val_str: return 20.0

    # Tentativo di conversione numerica pura
    try:
        num = float(val_str.replace('%', ''))
        return num if 0 <= num <= 100 else np.nan
    except ValueError:
        return np.nan

# =========================================================
# Funzione di standardizzazione del GRADE istologico
# =========================================================

def standardizzazione_grade(val):
    """
    Standardizzazione del grado istologico (1, 2, 3).
    Gli intervalli (es. 1-2) vengono arrotondati al grado superiore 
    per prudenza clinica o mediati (2).
    """
    if pd.isna(val) or str(val).lower() in ['nan', 'none', '-1', '', '-1.0']:
        return np.nan

    val_str = str(val).lower().strip()

    # Gestione intervalli (es. '2 to 3') -> approccio conservativo (3) o media (2.5)
    if 'to' in val_str:
        try:
            nums = [float(s.strip()) for s in val_str.replace(' ', '').split('to')]
            return sum(nums) / len(nums) # Restituisce 1.5, 2.5 ecc.
        except:
            return np.nan

    try:
        num = float(val_str)
        return num if num in [1, 2, 3] else np.nan
    except ValueError:
        return np.nan

# =========================================================
# Funzione di standardizzazione IHC (ER, PR, HER2)
# =========================================================

def standardizzazione_IHC(val):
    """
    Standardizzazione binarizzata (0/1). 
    Se il valore è ambiguo o mancante (-1), restituisce NaN per tentare il recupero.
    """
    if pd.isna(val):
        return np.nan
        
    val_str = str(val).lower().strip()
    
    # Se il valore è -1, lo consideriamo mancante (NaN) per ora
    if val_str in ['-1', '-1.0', 'none', 'nan', '']:
        return np.nan

    # 1. Mapping testuale
    if any(x in val_str for x in ['neg', '0', 'zero']): 
        return 0
    if any(x in val_str for x in ['pos', 'strong', 'moderate', 'weak', '1+', '2+', '3+']): 
        return 1

    # 2. Gestione numerica
    try:
        num = float(val_str.replace('%', ''))
        return 1 if num >= 1 else 0
    except ValueError:
        return np.nan

# =========================================================
# Funzione principale di pulizia del dataset
# =========================================================

def clean_dataset(dataset):
    df = dataset.copy()

    # 1. Filtro tumori maligni
    if "tumor/benign" in df.columns:
        df = df[df["tumor/benign"] == 1].copy()

    # 2. Pulizia e SOVRASCRIZIONE colonne originali
    # 2. Pulizia e RECUPERO dati IHC (ER, PR, HER2)
    for rec in ['ER', 'PR', 'HER2']:
        sii_col = f'{rec} [SII]'
        perc_col = f'{rec} [%]'
        
        if sii_col in df.columns:
            # 1. Puliamo la colonna SII (Score)
            clean_sii = df[sii_col].apply(standardizzazione_IHC)
            
            # 2. Se abbiamo la colonna Perc (%), puliamo anche quella
            if perc_col in df.columns:
                clean_perc = df[perc_col].apply(standardizzazione_IHC)
                
                # IL RECUPERO AVVIENE QUI: 
                # Se clean_sii è NaN (perché c'era -1), prendiamo il valore da clean_perc
                df[sii_col] = clean_sii.fillna(clean_perc)
                
                # Controllo extra: se SII dice 0 ma la % dice 1 (es. >1%), fidati della %
                df[sii_col] = np.where((clean_sii == 0) & (clean_perc == 1), 1, df[sii_col])
            else:
                df[sii_col] = clean_sii

    # 3. Pulizia Ki-67 e Grade (sovrascrivendo le originali)
    if 'KI67 [%]' in df.columns:
        df['KI67 [%]'] = df['KI67 [%]'].apply(standardizzazione_ki67)
    
    if 'GRADE' in df.columns:
        df['GRADE'] = df['GRADE'].apply(standardizzazione_grade)

    # 4. Rimozione colonne tecniche (rimuoviamo solo le % che non servono più)
    # e le altre colonne inutili definite nella tua lista iniziale
    cols_to_drop = [c for c in colonne_da_rimuovere if c in df.columns]
    # Aggiungiamo le colonne [%] alla rimozione per non fare confusione
    cols_to_drop.extend(['ER [%]', 'PR [%]', 'HER2 [%]'])
    
    df = df.drop(columns=list(set(cols_to_drop)), errors='ignore')

    return df

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
