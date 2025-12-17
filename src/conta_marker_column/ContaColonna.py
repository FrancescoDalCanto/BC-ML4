from pathlib import Path
import pandas as pd
import numpy as np
import re

# Marker "logici" che vuoi auditare
MARKERS = ["ER", "PR", "HER2", "KI67", "isTN"]

# Per ogni marker, lista di possibili nomi colonna nei vari CSV
MARKER_ALIASES = {
    "ER":   ["ER [SII]", "ER", "ER [%]", "ER_SII"],
    "PR":   ["PR [SII]", "PR", "PR [%]", "PR_SII"],
    "HER2": ["HER2 [SII]", "HER2", "HER2_SII"],
    "KI67": ["KI67 [%]", "KI67", "Ki67", "Ki-67", "Ki67 [%]"],
    "isTN": ["isTN", "isTN ", "TN", "IsTN", "is_tn"]
}


def pick_column(df: pd.DataFrame, marker: str):
    """Ritorna il nome colonna effettivo nel df per quel marker, oppure None."""
    for cand in MARKER_ALIASES.get(marker, []):
        if cand in df.columns:
            return cand
    return None


def _count_types(series: pd.Series) -> dict:
    """
    Conta:
      - missing (NaN)
      - numerici_veri: int/float già numerici
      - numeric_formato_stringa: stringhe convertibili a numero (es '3', '3.0', '3+')
      - intervalli_stringa: stringhe che rappresentano intervalli (es '1-2', '1 – 2', '1 to 2', '1 a 2')
      - qualitativi_stringa: stringhe NON convertibili a numero e NON intervalli (es 'low', 'pos', 'neg')
    """
    s = series
    missing = int(s.isna().sum())

    # Numerico vero (int/float/np.number) e non NaN
    numerici_veri_row = s.apply(lambda x: isinstance(x, (int, float, np.number)) and pd.notna(x))

    # Stringhe "pulite" (attenzione: NaN -> "nan", ma poi filtriamo con ~s.isna())
    s_str = s.astype(str).str.strip().str.lower()

    # Intervalli tipo:
    # "1-2", "1 - 2", "1–2", "1 — 2", "1 to 2", "1 a 2"
    # Supporta anche decimali e numeri negativi.
    interval_regex = r"^\s*-?\d+(?:\.\d+)?\s*(?:\-|–|—|to|a)\s*-?\d+(?:\.\d+)?\s*$"
    intervalli_row = s_str.str.match(interval_regex, na=False) & (~s.isna())

    # Numeric-like: convertibile a numero anche se era stringa (gestendo '3+')
    # ESCLUDE gli intervalli (che altrimenti risulterebbero "non numerici" e finirebbero nei qualitativi)
    s_num_like = pd.to_numeric(
        s_str.str.replace('+', '', regex=False),
        errors="coerce"
    )
    numeric_like_row = s_num_like.notna() & (~s.isna()) & (~intervalli_row)

    # numeric_formato_stringa = convertibile a numero MA non già numerico vero
    numeric_formato_stringa_row = numeric_like_row & (~numerici_veri_row)

    # qualitativi = non missing e non numerico vero e non numeric-formato-stringa e non intervallo
    qualitativi_row = (~s.isna()) & (~numerici_veri_row) & (~numeric_formato_stringa_row) & (~intervalli_row)

    return {
        "missing": missing,
        "numerici_veri": int(numerici_veri_row.sum()),
        "numeric_formato_stringa": int(numeric_formato_stringa_row.sum()),
        "intervalli_stringa": int(intervalli_row.sum()),
        "qualitativi_stringa": int(qualitativi_row.sum()),
    }


def audit_one_dataset(csv_path: Path, dataset_name: str, top_k: int = 15) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    rows = []
    for marker in MARKERS:
        col = pick_column(df, marker)

        if col is None:
            rows.append({
                "dataset": dataset_name,
                "marker": marker,
                "colonna_usata": None,
                "presente": False,
                "missing": np.nan,
                "numerici_veri": np.nan,
                "numeric_formato_stringa": np.nan,
                "intervalli_stringa": np.nan,
                "qualitativi_stringa": np.nan,
                "top_valori": np.nan
            })
            continue

        counts = _count_types(df[col])

        # top valori raw: i più frequenti così come compaiono nel CSV
        vc = df[col].value_counts(dropna=False).head(top_k)
        top_vals = "; ".join([f"{idx}={val}" for idx, val in vc.items()])

        rows.append({
            "dataset": dataset_name,
            "marker": marker,
            "colonna_usata": col,
            "presente": True,
            **counts,
            "top_valori": top_vals
        })

    return pd.DataFrame(rows)


def audit_all_datasets(datasets: dict, save_csv: bool = True,
                       out_path: str = "conteggio.csv") -> pd.DataFrame:
    all_df = []
    for name, path in datasets.items():
        print(f"Audit: {name} -> {path}")
        all_df.append(audit_one_dataset(path, name))

    audit_df = pd.concat(all_df, ignore_index=True)

    if save_csv:
        audit_df.to_csv(out_path, index=False)
        print(f"\nSalvato: {out_path}")

    return audit_df




DATASET_PATH = Path('/Users/francesco/Tesi/BC-ML4/dataset/original')

datasets = {
    'ambl_lesions_radiomic' : DATASET_PATH / 'ambl_lesions_radiomic_medsam.csv',
    'duke_lesions_radiomic' : DATASET_PATH / 'duke_lesions_radiomic_medsam.csv',
    'ambl_lesions' : DATASET_PATH / 'ambl_lesions.csv',
    'duke_lesions' : DATASET_PATH / 'duke_lesions.csv',
    't2_medsam': DATASET_PATH / 't2_medsam_masks.csv',
    't2_preprocessed': DATASET_PATH / 't2_preprocessed_masks.csv',
    't2_original': DATASET_PATH / 't2_original_masks.csv',
    'medsam_dynamic': DATASET_PATH / 'medsam_dynamic.csv',
    'preprocessed_dynamic': DATASET_PATH / 'preprocessed_dynamic.csv',
    'original_dynamic': DATASET_PATH / 'original_dynamic.csv'
}

audit_df = audit_all_datasets(datasets, save_csv=True)

print("\nRISULTATO (prime righe):")
print(audit_df.head(20))

# Pivot: qualitativi_stringa per marker
pivot_qual = audit_df.pivot(index="dataset", columns="marker", values="qualitativi_stringa")
print("\nPIVOT: qualitativi_stringa (dataset x marker)")
print(pivot_qual)

# Pivot: intervalli_stringa per marker (nuovo)
pivot_intervalli = audit_df.pivot(index="dataset", columns="marker", values="intervalli_stringa")
print("\nPIVOT: intervalli_stringa (dataset x marker)")
print(pivot_intervalli)

# Pivot: numeric_formato_stringa per marker
pivot_numstr = audit_df.pivot(index="dataset", columns="marker", values="numeric_formato_stringa")
print("\nPIVOT: numeric_formato_stringa (dataset x marker)")
print(pivot_numstr)

# Pivot: numerici_veri per marker
pivot_num = audit_df.pivot(index="dataset", columns="marker", values="numerici_veri")
print("\nPIVOT: numerici_veri (dataset x marker)")
print(pivot_num)

# Pivot: quale colonna è stata usata (ottimo per controllare Duke)
pivot_used = audit_df.pivot(index="dataset", columns="marker", values="colonna_usata")
print("\nPIVOT: colonna_usata (dataset x marker)")
print(pivot_used)
