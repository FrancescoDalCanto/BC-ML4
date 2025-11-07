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
    except Exception as e:
        print(Fore.RED + f"Errore nel controllo completezza: {e}")





    #=====================================
    # Step 3: Controlli di coerenza
    #=====================================
    try:
        motivo_errori = []
        
        # Controlli specifici per colonne che EFFETTIVAMENTE esistono
        if 'tumor/benign' in dataset.columns:
            valori_validi = dataset['tumor/benign'].isin([0.0, 1.0])
            if not valori_validi.all():
                motivo_errori.append(f"tumor/benign: valori non validi")
        
        if 'isTN' in dataset.columns:
            valori_validi = dataset['isTN'].isin([0, 1])
            if not valori_validi.all():
                motivo_errori.append(f"isTN: valori non validi")
        
        if 'Breast' in dataset.columns:
            valori_validi = dataset['Breast'].isin(['L', 'R'])
            if not valori_validi.all():
                motivo_errori.append(f"Breast: valori non validi (deve essere L o R)")
        
        # Controllo offsets positivi
        offset_cols = ['z_offset', 'y_offset', 'x_offset']
        for col in offset_cols:
            if col in dataset.columns:
                negativi = (dataset[col] < 0).sum()
                if negativi > 0:
                    motivo_errori.append(f"{col}: {negativi} valori negativi")
        
        # MODIFICATO: Controllo i biomarcatori IHC
        # Ora accetto anche i valori intermedi (1.5, 2.5) generati dalla standardizzazione
        biomarcatori_ihc = ['ER [SII]', 'PR [SII]', 'HER2 [SII]']
        for bio in biomarcatori_ihc:
            if bio in dataset.columns:
                # Ignoro i NaN per questo controllo
                valori_non_nan = dataset[bio].dropna()
                if len(valori_non_nan) > 0:
                    # Accetto valori interi (0, 1, 2, 3) e intermedi (1.5, 2.5)
                    valori_validi = valori_non_nan.isin([0, 1, 1.5, 2, 2.5, 3])
                    if not valori_validi.all():
                        invalidi = (~valori_validi).sum()
                        motivo_errori.append(f"{bio}: {invalidi} valori non validi (attesi: 0, 1, 1.5, 2, 2.5, 3)")
        
        # Controllo GRADE (accetta anche 1.5, 2.5)
        if 'GRADE' in dataset.columns:
            valori_non_nan = dataset['GRADE'].dropna()
            if len(valori_non_nan) > 0:
                # Accetto valori tra 1 e 3, inclusi intermedi (1.5, 2.5)
                valori_validi = (valori_non_nan >= 1) & (valori_non_nan <= 3)
                if not valori_validi.all():
                    invalidi = (~valori_validi).sum()
                    motivo_errori.append(f"GRADE: {invalidi} valori fuori range [1-3]")
        
        # Controllo KI-67 (0-100)
        if 'KI67 [%]' in dataset.columns:
            valori_non_nan = dataset['KI67 [%]'].dropna()
            if len(valori_non_nan) > 0:
                invalidi = ((valori_non_nan < 0) | (valori_non_nan > 100)).sum()
                if invalidi > 0:
                    motivo_errori.append(f"KI67 [%]: {invalidi} valori fuori range [0-100]")



        if len(motivo_errori) == 0:
            print("Controllo coerenza: tutti i dati sono coerenti")
            check += 1
        else:
            print(Fore.RED + f"Problemi di coerenza:\n{chr(10).join(motivo_errori)}")
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
            # Creo un regex, mi cerca tutti i pazienti del tipo AMBL-<numero>
            pattern = r'^AMBL-\d{3,}$'
            # Tramite la tilde inverto i valori booleani
            # In pratica vado a contare quanti patientID non corrispondono al regex richiesto
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
