
"""
    Questa funzione mi va a valutare i dati, ovvero
    mi valuta se i dati siano puliti, accurati e 
    pronti per essere utilizzati;

    Mi deve valutare:
        1. Controllo del tipo di dati
        2. Controlli di completezza
        3. Controlli di coerenza
        4. Controlli di unicità
        5. Controlli di formato
"""


import pandas as pd

from colorama import Fore, init
# Resetto il colore dopo ogni print
init(autoreset=True)

def DataValidation(dataset, nome_file):

    check = 0

    #=====================================
    # Step 1: Controllo del tipo di dati
    #=====================================
    try:
        colonne_numeriche = dataset.select_dtypes(include=['number']).columns
        for col in colonne_numeriche:
            dataset[col] = pd.to_numeric(dataset[col], errors='coerce')
        check += 1
    except Exception as e:
        print(Fore.RED + f"Errore nel controllo tipo di dati: {e}")
    
    #=====================================
    # Step 2: Controlli di completezza
    #=====================================
    try:
        dati_mancanti = dataset.isnull().sum()
        
        if dati_mancanti.sum() == 0:
            print("Dataset completo - nessun valore mancante")
            check += 1
        else:
            print(f"Valori mancanti trovati:\n{dati_mancanti[dati_mancanti > 0]}")
            dataset = dataset.dropna()
    except Exception as e:
        print(Fore.RED + f"Errore nel controllo completezza: {e}")



    #=====================================
    # Step 3: Controlli di coerenza
    #=====================================
    try:
        errori_coerenza = []
        
        # Controlli specifici per colonne che EFFETTIVAMENTE esistono
        if 'tumor/benign' in dataset.columns:
            valori_validi = dataset['tumor/benign'].isin([0.0, 1.0])
            if not valori_validi.all():
                errori_coerenza.append(f"tumor/benign: valori non validi")
        
        if 'isTN' in dataset.columns:
            valori_validi = dataset['isTN'].isin([0, 1])
            if not valori_validi.all():
                errori_coerenza.append(f"isTN: valori non validi")
        
        if 'Breast' in dataset.columns:
            valori_validi = dataset['Breast'].isin(['L', 'R'])
            if not valori_validi.all():
                errori_coerenza.append(f"Breast: valori non validi (deve essere L o R)")
        
        # Controllo offsets positivi
        offset_cols = ['z_offset', 'y_offset', 'x_offset']
        for col in offset_cols:
            if col in dataset.columns:
                negativi = (dataset[col] < 0).sum()
                if negativi > 0:
                    errori_coerenza.append(f"{col}: {negativi} valori negativi")
        
        if len(errori_coerenza) == 0:
            print("Controllo coerenza: tutti i dati sono coerenti")
            check += 1
        else:
            print(Fore.RED + f"Problemi di coerenza:\n{chr(10).join(errori_coerenza)}")
    except Exception as e:
        print(Fore.RED + f"Errore nel controllo coerenza: {e}")


    #=====================================
    # Step 4: Controlli di unicità
    #=====================================
    try:
        duplicati = dataset.duplicated().sum()
        
        if duplicati == 0:
            print("Nessuna riga duplicata trovata")
            check += 1
        else:
            print(f"{duplicati} righe duplicate trovate")
            dataset = dataset.drop_duplicates()
    except Exception as e:
        print(Fore.RED + f"Errore nel controllo unicità: {e}")



    #=====================================
    # Step 5: Controlli di formato
    #=====================================
    try:
        errori_formato = []
        
        # Controllo Pixel Spacing e Slice Thickness (dovrebbero essere positivi)
        if 'Pixel Spacing' in dataset.columns:
            invalidi = (dataset['Pixel Spacing'] <= 0).sum()
            if invalidi > 0:
                errori_formato.append(f"Pixel Spacing: {invalidi} valori non positivi")
        
        if 'Slice Thickness' in dataset.columns:
            invalidi = (dataset['Slice Thickness'] <= 0).sum()
            if invalidi > 0:
                errori_formato.append(f"Slice Thickness: {invalidi} valori non positivi")
        
        # Controllo Patient ID formato
        if 'Patient ID' in dataset.columns:
            pattern = r'^AMBL-\d{3,}$'
            invalidi = (~dataset['Patient ID'].astype(str).str.match(pattern)).sum()
            if invalidi > 0:
                errori_formato.append(f"Patient ID: {invalidi} ID non nel formato AMBL-XXX")
        
        if len(errori_formato) == 0:
            print("Tutti i formati sono validi")
            check += 1
        else:
            print(Fore.RED + f"Problemi di formato:\n{chr(10).join(errori_formato)}")
    except Exception as e:
        print(Fore.RED + f"Errore nel controllo formato: {e}")

    return check