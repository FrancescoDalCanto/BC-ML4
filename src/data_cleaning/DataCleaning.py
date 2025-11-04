import pandas as pd
from pathlib import Path
import numpy as np


from colorama import Fore, init
# Resetto il colore dopo ogni print
init(autoreset=True)


from ValidationDataset import validation


# TODO: ricommentare tutto

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
    # Se il valore è NaN, lo ritorna come NaN
    if pd.isna(val):
        return np.nan

    # Converto a stringa e rimuovo spazi bianchi
    val_str = str(val).lower().strip()

    # Se il valore è vuoto, -1, o "nan", lo ritorna come NaN
    if val_str in ['-1', '', 'nan', 'none']:
        return np.nan

    # ============================================
    # Gestisce valori con intervalli (es: "20 to 25")
    # ============================================
    if 'to' in val_str:
        # Prova a dividere per numeri e calcola la media
        # Esempio: "20 to 25" -> (20+25)/2 = 22.5
        try:
            parts = val_str.split('to')
            num1 = float(parts[0].strip())
            num2 = float(parts[1].strip())
            return (num1 + num2) / 2
        except:
            pass

        # Se fallisce la conversione numerica, prova con le categorie
        # Esempio: "low to intermediate" -> 20 (media tra 15 e 25)
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
        # Se non riesce a convertire a numero, ritorna NaN
        return np.nan



# **************************************
#   Funzione per standardizzare IHC
# **************************************
def standardizzazione_IHC(val):
    """
    NUOVO: Converte valori categorici E numerici della scala IHC a scala 0-3
    Gestisce entrambi i formati:
      - Numerici: 0, 1, 2, 3, 0.5, 1.5, etc.
      - Categorici: "neg", "weak", "moderate", "strong", "pos"

    Mapping finale:
      0 = Negativo (neg)
      1 = Debole (weak, pos generici)
      2 = Moderato (moderate)
      3 = Forte (strong)
    """

    # Se il valore è NaN, lo ritorna come NaN
    if pd.isna(val):
        return np.nan

    # Converto a stringa e normalizzo (minuscolo e senza spazi)
    val_str = str(val).lower().strip()

    # Se il valore è vuoto, -1, o "nan", lo ritorna come NaN
    if val_str in ['-1', '', 'nan', 'none']:
        return np.nan

    # ============================================
    # MAPPING categorico
    # ============================================
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

    # ============================================
    # Se non è una stringa categorica, prova numerico
    # ============================================
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
        Vado a standardizzare ER[SII], PR[SII] e HER2[SII]

        ✅ MODIFICATO: Ora uso la funzione standardizzazione_IHC() che gestisce
           sia valori numerici che categorici (neg, weak, moderate, strong)

        Prima: pd.to_numeric() convertiva tutto il testo in NaN
        Adesso: apply(standardizzazione_IHC) preserva i dati categorici
    """
    for index in ['ER [SII]', 'PR [SII]', 'HER2 [SII]']:
        if index in dataset.columns:
            # Applica la funzione che gestisce sia numerici che categorici
            dataset[index] = dataset[index].apply(standardizzazione_IHC)



    """
        Vago a standardizzare ER[%], PR[%] e HER2[%]
    """
    for index in ['ER [%]', 'PR [%]', 'HER2 [%]']:
        if index in dataset.columns:
            # Converto da obj a numerico
            # errors='coerce' trasforma i valori non convertibili in NaN
            dataset[index] = pd.to_numeric(dataset[index], errors = 'coerce')

            # Rimpiazzo i valori -1 con NaN
            # (-1 è un placeholder per "valore mancante")
            dataset[index] = dataset[index].replace(-1, np.nan)

            # Limito il valore tra 0-100
            # Valori < 0 diventano 0, valori > 100 diventano 100
            dataset[index] = dataset[index].clip(0,100)


    """
        Vago a standardizzare GRADE
    """
    if 'GRADE' in dataset.columns:
        # Converto da obj a numerico
        dataset['GRADE'] = pd.to_numeric(dataset['GRADE'], errors = 'coerce')

        # Rimpiazzo -1 con NaN
        dataset['GRADE'] = dataset['GRADE'].replace(-1, np.nan)

        # Limito il valore tra 1-3
        # (I gradi del tumore vanno da 1 a 3)
        dataset['GRADE'] = dataset['GRADE'].clip(1,3)



    """
        Vago a standardizzare ki-67[%] da categorico/numerico

        Questa colonna può contenere:
          - Valori numerici: 15, 23.5, 50, etc.
          - Categorie: "low", "intermediate", "high"
          - Intervalli: "20 to 25", "low to intermediate"
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
        # (significa "non è triple-negative")
        dataset['isTN'] = dataset['isTN'].fillna(0)

        # Converto il valore in binario: 0 o 1
        # Qualsiasi valore != 1 diventa 0
        dataset['isTN'] = (dataset['isTN'] == 1).astype(int)



    """
        Questo codice trasformava i dati biologici in NaN per i pazienti benigni
        (aggiungeva 143 NaN artificiali per ogni colonna biologica)

        Ora: I pazienti benigni mantengono i loro dati biologici originali
    """

    # Se triple-negative, ER/PR percentuali devono essere 0
    # (Triple-negative = nessun recettore ER, PR, HER2)
    if 'isTN' in dataset.columns:
        tn_mask = dataset['isTN'] == 1
        # Per i pazienti triple-negative, imposta ER[%] e PR[%] a 0
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


                # Carico il file CSV originale
                dataset = pd.read_csv(RAW_PATH_DATASET)
                # Mi salvo la lunghezza iniziale (numero di righe)
                start_len = len(dataset)



                # Pulisco il dataset con la funzione clean_dataset
                dataset_cleaned = clean_dataset(dataset)
                # Mi salvo la lunghezza dopo la pulizia
                # (dovrebbe essere uguale a start_len perché non rimuoviamo righe)
                end_len = len(dataset)



                # Eseguo i check di validazione sul dataset pulito
                checks = validation(dataset_cleaned, nome_file)



                # Salvo il dataset pulito nel CSV di output
                # index=False = non salva l'indice delle righe
                dataset_cleaned.to_csv(OUTPUT_FILE, index=False)


                # Stampo i risultati nel terminale
                print(Fore.MAGENTA + f"\n{start_len} -> {end_len}")
                print(Fore.YELLOW + f"{nome_file} pulizia completata")
                print(Fore.LIGHTGREEN_EX + f"{nome_file} completato ({checks}/6 check)\n")
                print("*" * 88)


            except Exception as e:
                # Se c'è un errore nel file singolo, lo stampa e continua
                print(Fore.RED + f"Errore in {nome_file}: {str(e)[:60]}\n")
    except Exception as e:
        # Se c'è un errore generale, lo stampa
        print(Fore.RED + f"ERRORE generale: {e}")



# **************************************
#   Avvio del programma per la Pulizia
# **************************************
if __name__ == "__main__":
    process_all_file()
