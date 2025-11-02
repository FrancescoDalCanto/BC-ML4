from scipy import stats
import numpy as np
from colorama import Fore, Style, init

# Inizializza colorama
init(autoreset=True)

def validation(dataset, filename):

    check = 0

    # 1. Valori mancanti
    print("CHECK 1: Valori mancanti critici")
    colonne_critiche = ['Patient ID', 'tumor/benign']
    if 'lesion idx' in dataset.columns:
        colonne_critiche.append('lesion idx')
    
    mancanti_critici = False
    for col in colonne_critiche:
        if col in dataset.columns:
            nan_count = dataset[col].isnull().sum()
            if nan_count > 0:
                print(f"  {Fore.RED}FAIL{Style.RESET_ALL}: Colonna critica {col} ha {nan_count} NaN")
                mancanti_critici = True
    
    if not mancanti_critici:
        print(f"  {Fore.GREEN}PASS{Style.RESET_ALL} (nessun NaN nelle colonne critiche)")
        check += 1
        
        # Info aggiuntiva
        bio_cols_with_nan = dataset[['GRADE', 'ER [SII]', 'PR [SII]', 'HER2 [SII]', 'KI67 [%]']].isnull().sum()
        bio_cols_with_nan = bio_cols_with_nan[bio_cols_with_nan > 0]
            
    else:
        print(f"  {Fore.RED}FAIL{Style.RESET_ALL}: NaN nelle colonne critiche")

    # 2. Range valido
    print("\nCHECK 2: Range validi")
    ranges = {
        'ER [SII]': (0, 3), 'PR [SII]': (0, 3), 'HER2 [SII]': (0, 3),
        'ER [%]': (0, 100), 'PR [%]': (0, 100), 'HER2 [%]': (0, 100),
        'GRADE': (1, 3), 'isTN': (0, 1)
    }
    all_valid = True
    for col, (min_val, max_val) in ranges.items():
        if col in dataset.columns:
            col_data = dataset[col].dropna()
            if len(col_data) > 0:
                actual_min = col_data.min()
                actual_max = col_data.max()
                is_valid = (actual_min >= min_val - 0.01) and (actual_max <= max_val + 0.01)
                 
                if not is_valid:
                    all_valid = False
    
    if all_valid:
        print(f"  {Fore.GREEN}PASS{Style.RESET_ALL}")
        check += 1
    else:
        print(f"  {Fore.RED}FAIL{Style.RESET_ALL}: Range non validi")
    
    # 3. Duplicati
    print("\nCHECK 3: Duplicati")
    duplicati = 0
    if 'lesion idx' in dataset.columns:
        duplicati = dataset.duplicated(subset=['Patient ID', 'lesion idx']).sum()
        print(f"  Controllo con Patient ID + lesion idx")
    elif 'Patient ID' in dataset.columns:
        duplicati = dataset.duplicated(subset=['Patient ID']).sum()
    
    if duplicati == 0:
        print(f"  {Fore.GREEN}PASS{Style.RESET_ALL}")
        check += 1
    else:
        print(f"  {Fore.RED}FAIL{Style.RESET_ALL}")

    # 4. Coerenza biologica
    print("\nCHECK 4: Coerenza biologica (benigni senza GRADE)")
    if 'tumor/benign' in dataset.columns and 'GRADE' in dataset.columns:
        benigni_mask = dataset['tumor/benign'] == 0
        benigni_with_grade = dataset.loc[benigni_mask, 'GRADE'].notna().sum()

        
        if benigni_with_grade == 0:
            print(f"  {Fore.GREEN}PASS{Style.RESET_ALL}")
            check += 1
        else:
            print(f"  {Fore.RED}FAIL{Style.RESET_ALL}: Alcuni benigni hanno GRADE")
            check += 0.5
    else:
        print("  SKIP: Colonne non presenti")

    # 5. Outlier
    print("\nCHECK 5: Outlier (Z-score > 3)")
    radiomic_cols = [col for col in dataset.columns if 'original_' in col]
    
    if len(radiomic_cols) > 0:
        X_radiomics = dataset[radiomic_cols].dropna()
        
        if len(X_radiomics) > 0:
            z_scores = np.abs(stats.zscore(X_radiomics, nan_policy='omit'))
            outliers_count = (z_scores > 3).sum().sum()
            total_values = len(X_radiomics) * len(radiomic_cols)
            outliers_pct = (outliers_count / total_values * 100) if total_values > 0 else 0
                        
            if outliers_pct < 5:
                print(f"  {Fore.GREEN}PASS{Style.RESET_ALL}")
                check += 1
            else:
                print(f"  {Fore.RED}FAIL{Style.RESET_ALL}: Outlier > 5%")
        else:
            print("  SKIP: Nessuna riga con radiomiche complete")
            print(f"  {Fore.GREEN}PASS{Style.RESET_ALL} (no radiomiche da verificare)")
            check += 1
    else:
        print("  SKIP: Nessuna feature radiomica nel file")
        print(f"  {Fore.GREEN}PASS{Style.RESET_ALL} (è normale per file dinamici)")
        check += 1

    # 6. Classi sbilanciate
    print("\nCHECK 6: Classi bilanciate")
    if 'tumor/benign' in dataset.columns:
        class_dist = dataset['tumor/benign'].value_counts()
        if len(class_dist) == 2:
            print(f"  {Fore.GREEN}PASS{Style.RESET_ALL}")
            check += 1
        else:
            print(f"  {Fore.RED}FAIL{Style.RESET_ALL}: Non ci sono esattamente 2 classi")
    else:
        print("  SKIP: Colonna tumor/benign non presente")
    
    return int(check)
