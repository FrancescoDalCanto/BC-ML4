from pathlib import Path
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold,StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings("ignore")

FILE_PATH = Path('/Users/francesco/Tesi/BC-ML4/dataset/cleaned')

TIPO_FOLD = "kfold"
BALANCE_TEST = True

# MODIFICATO: Ora c'è un solo target per il dataset merged
TARGET_COL = "HER2 [SII]"  # Colonna unificata dopo la fusione


# Funzioni
def load_dataset(file_path, csv_name):
    """Carica il CSV"""
    df = pd.read_csv(file_path)

    # Informazioni base sul dataset_name
    print(f"  Shape: {df.shape}")
    print(f"  Colonne: {df.shape[1]}")
    print(f"  Righe: {df.shape[0]}")
    # Mostro solo le prime 5 righe
    print(df.head(5))

    if "Patient ID" not in df.columns:
        raise ValueError(f"{csv_name} - manca 'Patient ID'")
    return df.copy()


def select_target(df, target_col):
    """Verifica che il target sia valido"""
    
    if target_col not in df.columns:
        raise ValueError(f"Colonna '{target_col}' mancante")
    
    # Mostro valori PRIMA della selezione
    print(f"\n VALORI TARGET PRIMA DELLA PULIZIA:")
    print(f"  Tipo: {df[target_col].dtype}")
    print(f"  Valori unici: {sorted(df[target_col].dropna().unique())}")
    print(f"  NaN: {df[target_col].isna().sum()}")
    print(f"\n  Distribuzione:")
    print(df[target_col].value_counts(dropna=False))
    
    # NUOVO: Mostra distribuzione per dataset di origine
    print(f"\n  Distribuzione per dataset:")
    df_temp = df.copy()
    df_temp['dataset_origin'] = df_temp['Patient ID'].apply(
        lambda x: 'DUKE' if str(x).startswith('DUKE') else 'AMBL'
    )
    print(df_temp.groupby('dataset_origin')[target_col].value_counts().unstack(fill_value=0))
    
    y = df[target_col]
    
    if y.nunique() < 2:
        raise ValueError(f"Target '{target_col}' ha una sola classe")
    
    # Statistiche più dettagliate
    print(f"\n Target selezionato:")
    print(f"  Classe 0: {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
    print(f"  Classe 1: {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
    
    return df, y


def build_features_and_groups(df, target_col):
    """SEMPLIFICATO: Non serve più distinguere tra duke/ambl"""
    print(f"\n{'='*80}")
    print(f"COSTRUZIONE FEATURES E GROUPS")
    print(f"{'='*80}")
    
    # Serve per la StratifiedGroupKFold
    groups = df["Patient ID"].copy()
    
    # MODIFICATO: Lista unificata di colonne da rimuovere
    drop_cols = [
        "Patient ID",
        target_col,
    ]
    
    # Mostro quali colonne vengono rimosse
    print(f"\n COLONNE DA RIMUOVERE:")  
    print(f"  Metadati + target: Patient ID, {target_col}")
    print(f"  Altri target: ER [SII], HER2 [SII]")
    print(f"  Totale: {len(drop_cols)}")
    
    # Verifico l'esistenza delle colonne
    cols_found = [c for c in drop_cols if c in df.columns]
    cols_missing = [c for c in drop_cols if c not in df.columns]
    
    print(f"\n  Trovate: {len(cols_found)}")
    if cols_missing:
        print(f"  NON trovate (OK, verranno ignorate): {cols_missing}")
    
    # Rimuovo le colonne
    X = df.drop(columns=drop_cols, errors="ignore")

    print(f"\n Features estratte: {X.shape[1]} colonne, {X.shape[0]} righe")
    
    # Verifico tipi e Nan
    print(f"\n{'='*80}")
    print("VERIFICA DATI")
    print(f"{'='*80}")
    
    # Mostra tipi di dato
    print(f"\n TIPI DI DATO:")
    print(X.dtypes.value_counts())
    
    # Conta NaN
    nan_count = X.isna().sum().sum()
    print(f"\n NaN PRESENTI: {nan_count}")
    
    if nan_count > 0:
        print(f"\n  Colonne con NaN (top 10):")
        nan_cols = X.isna().sum()
        print(nan_cols[nan_cols > 0].sort_values(ascending=False).head(10))
        
        # Conta righe con NaN
        rows_with_nan = X.isna().any(axis=1).sum()
        print(f"\n Righe con almeno un NaN: {rows_with_nan}/{len(X)}")
        
        print(f"\n  Queste righe verranno rimosse nella pipeline principale")
    else:
        print(f"\n Nessun NaN presente")
    
    # Mostra prime righe
    print(f"\n PRIME 3 RIGHE DEL DATASET FINALE:")
    print(X.head(3).iloc[:, :5]) 
    
    print(f"\n{'='*80}")
    print("VERIFICA PULIZIA NOMI COLONNE")
    print(f"{'='*80}")

    cols_with_special = [c for c in X.columns if any(char in c for char in ['[', ']', '<'])]

    print(f"\nColonne con caratteri speciali: {len(cols_with_special)}")

    if len(cols_with_special) > 0:
        print(f"  Esempi: {cols_with_special[:5]}")
        X.columns = [re.sub(r"\[|\]|<", "", c) for c in X.columns]
        print(f" Nomi colonne puliti")
    else:
        print(f" Nessuna colonna con caratteri speciali")
        print(f" Pulizia NON necessaria per questo dataset")

    print(f"\n FEATURES FINALI: {X.shape}")
    
    return X, groups


# StratifiedGroupKFold, StratifiedKFold
def stratifiedgroup(X, y, n_splits, groups):
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )
    return list(sgkf.split(X, y, groups))


def stratified(X, y, n_splits):
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )
    return list(skf.split(X, y))


# Bilanciamento
def balance_test_sets(cv_splits, y):
    """
    Bilancia i test set a 50/50.
    I campioni in eccesso vengono spostati al training set.
    """
    balanced_splits = []
    
    print(f"\n{'='*80}")
    print("BILANCIAMENTO DEI TEST SET")
    print(f"{'='*80}\n")
    
    for fold_num, (train_idx, test_idx) in enumerate(cv_splits, 1):
        y_test = y.iloc[test_idx]
        
        # Indici per classe
        mask_0 = (y_test == 0).values
        mask_1 = (y_test == 1).values
        
        idx_0 = test_idx[mask_0]
        idx_1 = test_idx[mask_1]
        
        n_0 = len(idx_0)
        n_1 = len(idx_1)
        n_min = min(n_0, n_1)
        
        print(f"Fold {fold_num}:")
        print(f"  PRIMA  - Test: Classe 0={n_0}, Classe 1={n_1}, Totale={len(test_idx)}")
        
        # Campiona n_min per classe (bilanciamento 50/50)
        np.random.seed(42 + fold_num)
        idx_0_sampled = np.random.choice(idx_0, size=n_min, replace=False)
        idx_1_sampled = np.random.choice(idx_1, size=n_min, replace=False)
        
        # Test bilanciato
        test_idx_balanced = np.concatenate([idx_0_sampled, idx_1_sampled])
        
        # Campioni non usati → training
        unused = np.setdiff1d(test_idx, test_idx_balanced)
        train_idx_new = np.concatenate([train_idx, unused])
        
        print(f"  DOPO   - Test: Classe 0={n_min}, Classe 1={n_min}, Totale={n_min*2}")
        print(f"           Train: {len(train_idx)} → {len(train_idx_new)} (+{len(unused)} campioni)\n")
        
        # Verifica che sia perfettamente bilanciato
        assert len(idx_0_sampled) == len(idx_1_sampled), "Test set non bilanciato!"
        
        balanced_splits.append((train_idx_new, test_idx_balanced))
    
    print(f" Split bilanciati creati con successo\n")
    return balanced_splits


# Grid Search
def gridsearch(X, y, splits):
    model = RandomForestClassifier(
        random_state=42,
        n_jobs=1,
        class_weight="balanced_subsample"
    )

    param_grid = {
        "n_estimators": [50, 75, 100],
        "max_depth": [2, 4, 6],
        "min_samples_leaf": [1, 2, 3]
    }


    grid = GridSearchCV(
        model,
        param_grid,
        cv=splits,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X, y)
    return grid


# Valutazione fold
def evaluate_folds(X, y, splits, best_params):
    """
    Valuta il modello con i best params su ogni fold separatamente.
    Restituisce metriche per ogni fold.
    """
    fold_metrics = []
    
    for fold_num, (train_idx, test_idx) in enumerate(splits):
        # Train
        model = RandomForestClassifier(
            random_state=42,
            n_jobs=1,
            class_weight="balanced_subsample"
        )

        model.set_params(**best_params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        
        # Test
        y_test = y.iloc[test_idx]
        y_pred = model.predict(X.iloc[test_idx])
        y_prob = model.predict_proba(X.iloc[test_idx])[:, 1]
        
        # Metriche
        f1 = f1_score(y.iloc[test_idx], y_pred, zero_division=0)
        acc = accuracy_score(y.iloc[test_idx], y_pred)
        bal_acc = balanced_accuracy_score(y.iloc[test_idx], y_pred)
        auc = roc_auc_score(y.iloc[test_idx], y_prob)
        
        # Se il test set è bilanciato, ACC dovrebbe = BAL_ACC
        diff = abs(acc - bal_acc)

        # Stampo le metrice dei test set
        if True:
            print(f"\n{'='*80}")
            print(f"FOLD {fold_num}/{len(splits)}")
            print(f"{'='*80}")
            print(f"\n{'─'*80}")
            print("METRICHE TEST SET")
            print(f"{'─'*80}")
            print(f"  F1-score:          {f1:.4f}")
            print(f"  Accuracy:          {acc:.4f}")
            print(f"  Balanced Accuracy: {bal_acc:.4f}")
            print(f"  Differenza:        {diff:.6f}", end="")
            
            # Stampo la matrice di confusione
            print(f"\n{'─'*80}")
            print("CONFUSION MATRIX")
            print(f"{'─'*80}")
            
            cm = confusion_matrix(y_test, y_pred)
            print(cm)
            
            tn, fp, fn, tp = cm.ravel()
            print(f"\n  TN={tn} | FP={fp} | FN={fn} | TP={tp}")
            
            if (tn + fp) > 0:
                recall_0 = tn / (tn + fp)
                print(f"  Classe 0: {tn}/{tn+fp} corretti ({recall_0*100:.1f}%)")
            
            if (fn + tp) > 0:
                recall_1 = tp / (fn + tp)
                print(f"  Classe 1: {tp}/{fn+tp} corretti ({recall_1*100:.1f}%)")

            # Stampo il classification report
            print(f"\n{'─'*80}")
            print("CLASSIFICATION REPORT")
            print(f"{'─'*80}")
            print(classification_report(y_test, y_pred, zero_division=0))
        fold_metrics.append({
            "fold": fold_num,
            "f1": f1,
            "acc": acc,
            "bal_acc": bal_acc,
            "auc": auc,
            "acc_bal_diff": diff
        })
    # Stampa riepilogo performance per fold
    print(f"\n{'='*80}")
    print("RIEPILOGO PERFORMANCE PER FOLD")
    print(f"{'='*80}")
    print(f"{'Fold':<8} {'F1-Score':<12} {'Accuracy':<12} {'Bal. Acc':<12} {'AUC':<12}")
    print(f"{'─'*80}")
    for m in fold_metrics:
        print(f"{m['fold']:<8} {m['f1']:<12.4f} {m['acc']:<12.4f} {m['bal_acc']:<12.4f} {m['auc']:<12.4f}")
    print(f"{'─'*80}")
    print(f"{'Mean':<8} {np.mean([m['f1'] for m in fold_metrics]):<12.4f} "
          f"{np.mean([m['acc'] for m in fold_metrics]):<12.4f} "
          f"{np.mean([m['bal_acc'] for m in fold_metrics]):<12.4f} "
          f"{np.mean([m['auc'] for m in fold_metrics]):<12.4f}")
    print(f"{'Std':<8} {np.std([m['f1'] for m in fold_metrics]):<12.4f} "
          f"{np.std([m['acc'] for m in fold_metrics]):<12.4f} "
          f"{np.std([m['bal_acc'] for m in fold_metrics]):<12.4f} "
          f"{np.std([m['auc'] for m in fold_metrics]):<12.4f}")
    print(f"{'='*80}\n")
    return fold_metrics


# Training - SEMPLIFICATO
def train(file_path, csv_name, target_col, tipo_fold, balance_test=False):
    """
    Pipeline completa di training per predizione PR sul dataset merged.
    Esegue Grid Search con cross-validation e restituisce i migliori parametri.
    """
    print("\n" + "="*80)
    print(f"PROCESSING: {csv_name}")
    print("="*80)

    df = load_dataset(file_path, csv_name)

    df, y = select_target(df, target_col)

    print(f"\nDistribuzione target:")
    for val in sorted(y.unique()):
        count = (y == val).sum()
        perc = count / len(y) * 100
        print(f"  Classe {val}: {count} ({perc:.1f}%)")

    # Preparo le feature e il group
    X, groups = build_features_and_groups(df, target_col)
    print(f"\n Features: {X.shape[1]} colonne")
    print(f" Campioni: {X.shape[0]}")
    print(f" Pazienti: {groups.nunique()}")

    # Creo gli split
    print(f"\n{'='*80}")
    print(f"CROSS-VALIDATION: {tipo_fold}")
    print(f"{'='*80}")

    if tipo_fold == "groupkfold":
        print(">>> StratifiedGroupKFold (pazienti separati)")
        splits_raw = stratifiedgroup(X, y, n_splits=5, groups=groups)
        
    elif tipo_fold == "kfold":
        print(">>> StratifiedKFold (standard)")
        splits_raw = stratified(X, y, n_splits=5)
        
    else:
        raise ValueError(
            f"tipo_fold '{tipo_fold}' non riconosciuto. "
            f"Usa: 'groupkfold' o 'kfold'"
        )
    
    # Bilancia test set se richiesto
    if balance_test:
        splits = balance_test_sets(splits_raw, y)
    else:
        splits = splits_raw
        print(f" Creati {len(splits)} fold (senza bilanciamento)")

    print(f" Creati {len(splits)} fold")

    # Eseguo la grid search
    grid = gridsearch(X, y, splits)

    fold_metrics = evaluate_folds(X, y, splits, grid.best_params_)

    # Calcola medie e std
    f1_scores = [m["f1"] for m in fold_metrics]
    acc_scores = [m["acc"] for m in fold_metrics]
    bal_acc_scores = [m["bal_acc"] for m in fold_metrics]
    auc_scores = [m["auc"] for m in fold_metrics]

    return {
        "dataset": csv_name,
        "target": target_col,
        "best_params": grid.best_params_,
        "mean_f1": np.mean(f1_scores),
        "std_f1": np.std(f1_scores),
        "mean_acc": np.mean(acc_scores),
        "std_acc": np.std(acc_scores),
        "mean_bal_acc": np.mean(bal_acc_scores),
        "std_bal_acc": np.std(bal_acc_scores),
        "mean_auc": np.mean(auc_scores),
        "std_auc": np.std(auc_scores),
        "fold_metrics": fold_metrics
    }


# ============================================================================
# ESECUZIONE - MODIFICATO: Solo un dataset merged
# ============================================================================

results = {}

results["MERGED_PR"] = train(
    FILE_PATH / "merged_lesions.csv",
    "merged_lesions",
    target_col=TARGET_COL,
    tipo_fold=TIPO_FOLD,
    balance_test=BALANCE_TEST
)

# Stampa risultati finali
print("\n" + "="*80)
print("RIEPILOGO FINALE")
print("="*80)

for name, res in results.items():
    print(f"\nDataset: {res['dataset']}")
    print("-" * 80)
    print(f" Target: {res['target']}")
    print(f"  F1-score         = {res['mean_f1']:.3f}  ±  {res['std_f1']:.3f}")
    print(f"  Accuracy         = {res['mean_acc']:.3f}  ±  {res['std_acc']:.3f}")
    print(f"  Balanced Acc     = {res['mean_bal_acc']:.3f}  ±  {res['std_bal_acc']:.3f}") 
    print(f"  AUC              = {res['mean_auc']:.3f}  ±  {res['std_auc']:.3f}")

# ============================================================================
# SALVATAGGIO RISULTATI IN CSV
# ============================================================================
rows = []

for name, res in results.items():
    rows.append({
        "dataset": res["dataset"],
        "target": res["target"],
        "mean_f1": res["mean_f1"],
        "std_f1": res["std_f1"],
        "mean_accuracy": res["mean_acc"],
        "std_accuracy": res["std_acc"],
        "mean_balanced_accuracy": res["mean_bal_acc"],
        "std_balanced_accuracy": res["std_bal_acc"],
        "mean_auc": res["mean_auc"],
        "std_auc": res["std_auc"]
    })

results_df = pd.DataFrame(rows)

# Arrotonda a 3 cifre decimali
results_df = results_df.round(3)

output_csv = Path("/Users/francesco/Tesi/BC-ML4/src/modelli_singoli/RandomForest/rf_merged.csv")

write_header = not output_csv.exists()

results_df.to_csv(
    output_csv,
    index=False,
    mode="a",
    header=write_header
)

print("\nRisultati aggiunti a:")
print(output_csv)

