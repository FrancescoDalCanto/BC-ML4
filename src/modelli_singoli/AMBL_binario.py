import pandas as pd
from pathlib import Path

INPUT_FILE = Path('/Users/francesco/Tesi/BC-ML4/dataset/cleaned/ambl_lesions.csv')
OUTPUT_FILE = Path('/Users/francesco/Tesi/BC-ML4/dataset/cleaned/ambl_lesions_binary.csv')

# ============================================================================
# CONVERSIONE
# ============================================================================

print("=" * 80)
print("CONVERSIONE AMBL A BINARIO")
print("=" * 80)

# Carica CSV
print(f"\n1. Caricamento: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
print(f"   Shape: {df.shape}")
print(f"\nPrime righe:")
print(df.head())

# Verifica colonne
print("\n" + "=" * 80)
print("2. VERIFICA COLONNE")
print("=" * 80)

required = ["ER [SII]", "PR [SII]", "HER2 [SII]"]
missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Colonne mancanti: {missing}")

print("✓ Colonne trovate!")

# Rimuovi colonne
print("\n" + "=" * 80)
print("3. RIMOZIONE COLONNE")
print("=" * 80)

cols_to_remove = ["lesion idx", "tumor/benign"]
cols_removed = [c for c in cols_to_remove if c in df.columns]

if cols_removed:
    df = df.drop(columns=cols_removed)
    print(f"✓ Colonne rimosse: {cols_removed}")
else:
    print(f"⚠ Nessuna colonna da rimuovere trovata")

print(f"Shape dopo rimozione: {df.shape}")

# Conversione a binario
print("\n" + "=" * 80)
print("4. CONVERSIONE A BINARIO")
print("=" * 80)

# ER binary (>= 1)
er_values = pd.to_numeric(df["ER [SII]"], errors="coerce")
df["ER [SII]"] = (er_values >= 1).astype(int)
print(f"\nER [SII] >= 1 → ER [SII]")
print(f"  Distribuzione: {df['ER [SII]'].value_counts().to_dict()}")

# PR binary (>= 1)
pr_values = pd.to_numeric(df["PR [SII]"], errors="coerce")
df["PR [SII]"] = (pr_values >= 1).astype(int)
print(f"\nPR [SII] >= 1 → PR [SII]")
print(f"  Distribuzione: {df['PR [SII]'].value_counts().to_dict()}")

# HER2 binary (>= 3)
her2_values = pd.to_numeric(df["HER2 [SII]"], errors="coerce")
df["HER2 [SII]"] = (her2_values >= 3).astype(int)
print(f"\nHER2 [SII] >= 3 → HER2 [SII]")
print(f"  Distribuzione: {df['HER2 [SII]'].value_counts().to_dict()}")

# Rimuovi righe con NaN nei target binari
print("\n" + "=" * 80)
print("5. RIMOZIONE NaN")
print("=" * 80)

print(f"\nShape PRIMA: {df.shape}")
df_clean = df.dropna(subset=["ER [SII]", "PR [SII]", "HER2 [SII]"]).copy()
print(f"Shape DOPO:  {df_clean.shape}")
print(f"Righe rimosse: {df.shape[0] - df_clean.shape[0]}")

# Salva file
print("\n" + "=" * 80)
print("6. SALVATAGGIO")
print("=" * 80)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df_clean.to_csv(OUTPUT_FILE, index=False)

print(f"\n✓ File salvato: {OUTPUT_FILE}")
print(f"  Righe: {df_clean.shape[0]}")
print(f"  Colonne: {df_clean.shape[1]}")

# Verifica finale
print("\n" + "=" * 80)
print("7. VERIFICA COLONNE BINARIE")
print("=" * 80)

print(f"\nPrime righe con colonne binarie:")
cols_show = ["Patient ID", "ER [SII]", "PR [SII]", "HER2 [SII]"]
print(df_clean[cols_show].head(10))

print("\n" + "=" * 80)
print("✓ CONVERSIONE COMPLETATA!")
print("=" * 80)