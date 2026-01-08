import pandas as pd
from colorama import Fore, init

# Reset automatico del colore dopo ogni print
init(autoreset=True)


def DataValidation(dataset, nome_file):
    """
    Eseguo una serie di controlli di validazione sul dataset pulito.
    Ogni controllo superato incrementa il contatore finale.
    I controlli NON modificano la logica della pulizia, ma servono
    solo a verificare la qualità dei dati ottenuti.
    """

    check = 0

    # =========================================================
    # 1. Controllo del tipo di dati
    # =========================================================
    # Qui verifico che tutte le colonne numeriche siano effettivamente
    # convertibili in formato numerico. In caso contrario vengono
    # forzate a NaN, ma il controllo fallisce solo se avviene un errore.
    try:
        colonne_numeriche = dataset.select_dtypes(include=['number']).columns
        for col in colonne_numeriche:
            dataset[col] = pd.to_numeric(dataset[col], errors='coerce')
        check += 1
    except Exception as e:
        print(Fore.RED + f"Errore nel controllo del tipo di dati: {e}")

    # =========================================================
    # 2. Controllo di completezza (valori mancanti)
    # =========================================================
    # Verifico la presenza di valori NaN nel dataset.
    # La presenza di NaN NON è considerata un errore bloccante,
    # ma comporta il mancato superamento di questo controllo.
    try:
        dati_mancanti = dataset.isnull().sum()

        if dati_mancanti.sum() == 0:
            print("Dataset completo: nessun valore mancante")
            check += 1
        else:
            print("Valori mancanti trovati:")
            print(dati_mancanti[dati_mancanti > 0])
    except Exception as e:
        print(Fore.RED + f"Errore nel controllo di completezza: {e}")

    # =========================================================
    # 3. Controlli di coerenza semantica dei valori
    # =========================================================
    # In questa sezione verifico che alcune colonne chiave
    # contengano solo valori ammessi dal dominio clinico.
    try:
        motivo_errori = []

        # tumor/benign deve essere binario (0 o 1)
        if 'tumor/benign' in dataset.columns:
            if not dataset['tumor/benign'].isin([0.0, 1.0]).all():
                motivo_errori.append("tumor/benign: valori non validi")

        # isTN deve essere binario (0 o 1)
        if 'isTN' in dataset.columns:
            if not dataset['isTN'].isin([0, 1]).all():
                motivo_errori.append("isTN: valori non validi")

        # Breast deve essere L o R
        if 'Breast' in dataset.columns:
            if not dataset['Breast'].isin(['L', 'R']).all():
                motivo_errori.append("Breast: valori non validi (attesi L o R)")

        # Offset spaziali: devono essere non negativi
        for col in ['z_offset', 'y_offset', 'x_offset']:
            if col in dataset.columns:
                negativi = (dataset[col] < 0).sum()
                if negativi > 0:
                    motivo_errori.append(f"{col}: {negativi} valori negativi")

        # Biomarcatori IHC (scala 0–3 con valori intermedi)
        biomarcatori_ihc = ['ER [SII]', 'PR [SII]', 'HER2 [SII]']
        for bio in biomarcatori_ihc:
            if bio in dataset.columns:
                valori_non_nan = dataset[bio].dropna()
                if len(valori_non_nan) > 0:
                    valori_validi = valori_non_nan.isin([0, 1, 1.5, 2, 2.5, 3])
                    if not valori_validi.all():
                        invalidi = (~valori_validi).sum()
                        motivo_errori.append(
                            f"{bio}: {invalidi} valori non validi (attesi: 0, 1, 1.5, 2, 2.5, 3)"
                        )

        # GRADE istologico (valori compresi tra 1 e 3)
        if 'GRADE' in dataset.columns:
            valori_non_nan = dataset['GRADE'].dropna()
            if len(valori_non_nan) > 0:
                valori_validi = (valori_non_nan >= 1) & (valori_non_nan <= 3)
                if not valori_validi.all():
                    invalidi = (~valori_validi).sum()
                    motivo_errori.append(f"GRADE: {invalidi} valori fuori range [1–3]")

        # KI-67 (%) deve essere compreso tra 0 e 100
        if 'KI67 [%]' in dataset.columns:
            valori_non_nan = dataset['KI67 [%]'].dropna()
            invalidi = ((valori_non_nan < 0) | (valori_non_nan > 100)).sum()
            if invalidi > 0:
                motivo_errori.append(f"KI67 [%]: {invalidi} valori fuori range [0–100]")

        if len(motivo_errori) == 0:
            print("Controllo di coerenza: tutti i dati sono coerenti")
            check += 1
        else:
            print(Fore.RED + "Problemi di coerenza riscontrati:")
            print("\n".join(motivo_errori))

    except Exception as e:
        print(Fore.RED + f"Errore nel controllo di coerenza: {e}")

    # =========================================================
    # 4. Controllo di unicità delle righe
    # =========================================================
    # Verifico l’assenza di righe duplicate.
    try:
        duplicati = dataset.duplicated().sum()

        if duplicati == 0:
            print("Nessuna riga duplicata trovata")
            check += 1
        else:
            print(f"{duplicati} righe duplicate trovate")
            dataset = dataset.drop_duplicates()
    except Exception as e:
        print(Fore.RED + f"Errore nel controllo di unicità: {e}")

    # =========================================================
    # 5. Controlli di formato
    # =========================================================
    # Verifico che alcune colonne rispettino il formato atteso,
    # in particolare i parametri di acquisizione e il Patient ID.
    try:
        errori_formato = []

        # Pixel Spacing e Slice Thickness devono essere positivi
        if 'Pixel Spacing' in dataset.columns:
            invalidi = (dataset['Pixel Spacing'] <= 0).sum()
            if invalidi > 0:
                errori_formato.append(f"Pixel Spacing: {invalidi} valori non positivi")

        if 'Slice Thickness' in dataset.columns:
            invalidi = (dataset['Slice Thickness'] <= 0).sum()
            if invalidi > 0:
                errori_formato.append(f"Slice Thickness: {invalidi} valori non positivi")

        # Controllo formato Patient ID (dipendente dal dataset)
        if 'Patient ID' in dataset.columns:
            nome_lower = str(nome_file).lower()
            pid = dataset['Patient ID'].astype(str).str.strip()

            # AMBL → formato AMBL-XXX
            if nome_lower.startswith("ambl"):
                pattern = r'^AMBL-\d{3,}$'
                invalidi = (~pid.str.match(pattern)).sum()
                if invalidi > 0:
                    errori_formato.append(
                        f"Patient ID: {invalidi} ID non nel formato AMBL-XXX"
                    )

            # DUKE → formato DUKE_XXX
            elif nome_lower.startswith("duke"):
                pattern = r'^DUKE_\d{3,}$'
                invalidi = (~pid.str.match(pattern)).sum()
                if invalidi > 0:
                    errori_formato.append(
                        f"Patient ID: {invalidi} ID non nel formato DUKE_XXX"
                    )

        if len(errori_formato) == 0:
            print("Tutti i formati sono validi")
            check += 1
        else:
            print(Fore.RED + "Problemi di formato riscontrati:")
            print("\n".join(errori_formato))

    except Exception as e:
        print(Fore.RED + f"Errore nel controllo di formato: {e}")

    # =========================================================
    # Ritorno il numero di controlli superati
    # =========================================================
    return check
