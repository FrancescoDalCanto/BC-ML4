import pandas as pd
from pathlib import Path

# File input
AMBL_FILE = Path('/Users/francesco/Tesi/BC-ML4/dataset/cleaned/ambl_lesions_binary.csv')
DUKE_FILE = Path('/Users/francesco/Tesi/BC-ML4/dataset/cleaned/duke_lesions.csv')
OUTPUT_FILE = Path('/Users/francesco/Tesi/BC-ML4/dataset/cleaned/merged_lesions.csv')

print("=" * 80)
print("FUSIONE DATASET AMBL + DUKE")
print("=" * 80)

# Carica i due dataset
print(f"\n1. Caricamento dataset...")
df_ambl = pd.read_csv(AMBL_FILE)
df_duke = pd.read_csv(DUKE_FILE)

print(f"   AMBL shape: {df_ambl.shape}")
print(f"   DUKE shape: {df_duke.shape}")

# Rinomina colonne DUKE per farle combaciare con AMBL
print("\n2. Rinomina colonne target in DUKE...")
df_duke = df_duke.rename(columns={
    'ER': 'ER [SII]',
    'PR': 'PR [SII]',
    'HER2': 'HER2 [SII]'
})
print("   ✓ Colonne rinominate!")

# Verifica che ora abbiano le stesse colonne
ambl_cols = set(df_ambl.columns)
duke_cols = set(df_duke.columns)

if ambl_cols != duke_cols:
    print("\n⚠ ATTENZIONE: Le colonne non combaciano perfettamente!")
    print(f"   Colonne solo in AMBL: {ambl_cols - duke_cols}")
    print(f"   Colonne solo in DUKE: {duke_cols - ambl_cols}")
else:
    print("\n✓ Le colonne combaciano perfettamente!")

# Concatena i dataset
print("\n3. Concatenazione...")
df_merged = pd.concat([df_ambl, df_duke], ignore_index=True)
print(f"   Shape finale: {df_merged.shape}")
print(f"   Righe totali: {df_merged.shape[0]} (AMBL: {df_ambl.shape[0]} + DUKE: {df_duke.shape[0]})")

# Verifica distribuzione Patient ID
print("\n4. Verifica Patient ID...")
ambl_count = df_merged['Patient ID'].str.startswith('AMBL').sum()
duke_count = df_merged['Patient ID'].str.startswith('DUKE').sum()
print(f"   Pazienti AMBL: {ambl_count}")
print(f"   Pazienti DUKE: {duke_count}")

# Verifica distribuzione target
print("\n5. Distribuzione target nel dataset fusionato:")
for col in ['ER [SII]', 'PR [SII]', 'HER2 [SII]']:
    print(f"\n   {col}:")
    print(f"   {df_merged[col].value_counts().to_dict()}")

# Salva file
print("\n6. Salvataggio...")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df_merged.to_csv(OUTPUT_FILE, index=False)
print(f"   ✓ File salvato: {OUTPUT_FILE}")

print("\n" + "=" * 80)
print("✓ FUSIONE COMPLETATA!")
print("=" * 80)