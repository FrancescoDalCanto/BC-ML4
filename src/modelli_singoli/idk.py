import pandas as pd

# Carica i due file
radiomic = pd.read_csv('/Users/francesco/Tesi/BC-ML4/dataset/cleaned/ambl_lesions_radiomic_medsam_binary.csv')
dynamic = pd.read_csv('/Users/francesco/Tesi/BC-ML4/dataset/cleaned/ambl_original_dynamic_binary.csv')

print(f"Radiomic shape: {radiomic.shape}")
print(f"Dynamic shape: {dynamic.shape}")

# Verifica se 'lesion idx' esiste in dynamic
if 'lesion idx' in dynamic.columns:
    # Merge su ENTRAMBE le colonne: Patient ID + lesion idx
    merged_data = pd.merge(radiomic, dynamic, on=['Patient ID', 'lesion idx'], how='inner', suffixes=('', '_drop'))
    
    # Rimuovi colonne duplicate (_drop)
    merged_data = merged_data[[col for col in merged_data.columns if not col.endswith('_drop')]]
else:
    print("⚠️ 'lesion idx' non trovato in dynamic, verifico se l'ordine delle righe è identico...")
    
    # Verifica se Patient ID è nello stesso ordine
    if radiomic['Patient ID'].equals(dynamic['Patient ID']):
        print("✓ Patient ID corrisponde! Concatenazione orizzontale...")
        
        # Rimuovi colonne duplicate da dynamic
        cols_to_drop = [col for col in dynamic.columns if col in radiomic.columns and col != 'Patient ID']
        dynamic_clean = dynamic.drop(columns=cols_to_drop)
        
        # Concatena orizzontalmente
        merged_data = pd.concat([radiomic.reset_index(drop=True), 
                                dynamic_clean.drop('Patient ID', axis=1).reset_index(drop=True)], 
                               axis=1)
    else:
        print("❌ ERRORE: Patient ID non corrisponde e lesion idx mancante!")
        exit(1)

print(f"\n✓ Dataset unito creato: {merged_data.shape[0]} righe, {merged_data.shape[1]} colonne")

# Verifica che non ci siano duplicati
duplicates = merged_data[merged_data.duplicated(subset=['Patient ID'], keep=False)]
if len(duplicates) > 0:
    print(f"⚠️ ATTENZIONE: {len(duplicates)} righe duplicate trovate!")
    print(duplicates[['Patient ID']].value_counts().head())
else:
    print("✓ Nessun duplicato trovato")

# Verifica colonne con suffisso _x o _y
suffix_cols = [col for col in merged_data.columns if '_x' in col or '_y' in col]
if suffix_cols:
    print(f"⚠️ Colonne con suffisso trovate: {suffix_cols}")
else:
    print("✓ Nessuna colonna con suffisso _x/_y")

# Salva il file unito
merged_data.to_csv('/Users/francesco/Tesi/BC-ML4/dataset/cleaned/ambl_medsam_radiomic_dynamic_binary.csv', index=False)
print("✓ File salvato!")