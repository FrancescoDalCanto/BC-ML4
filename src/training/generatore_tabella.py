import pandas as pd

# Vanno inseriti a mano
all_models_results = {
    # Modificato con gli iperparametri
    'RandomForest': {
        't2_medsam': {'mean_score': 0.757, 'std_score': 0.065},
        't2_preprocessed': {'mean_score': 0.758, 'std_score': 0.083},
        't2_original': {'mean_score': 0.768, 'std_score': 0.092},
        'medsam_dynamic': {'mean_score': 0.747, 'std_score': 0.0066},
        'preprocessed_dynamic': {'mean_score': 0.739, 'std_score': 0.050},
        'original_dynamic': {'mean_score': 0.750, 'std_score': 0.028}
    },
    # TODO: da fare
    'LogisticRegression': {
        't2_medsam': {'mean_score': 0.660, 'std_score': 0.079},
        't2_preprocessed': {'mean_score': 0.728, 'std_score': 0.062},
        't2_original': {'mean_score': 0.729, 'std_score': 0.075},
        'medsam_dynamic': {'mean_score': 0.708, 'std_score': 0.061},
        'preprocessed_dynamic': {'mean_score': 0.706, 'std_score': 0.103},
        'original_dynamic': {'mean_score': 0.672, 'std_score': 0.077}
    },
    # TODO: da fare
    'XGBoost': {
        't2_medsam': {'mean_score': 0.707, 'std_score': 0.105},
        't2_preprocessed': {'mean_score': 0.723, 'std_score': 0.107},
        't2_original': {'mean_score': 0.718, 'std_score': 0.051},
        'medsam_dynamic': {'mean_score': 0.688, 'std_score': 0.097},
        'preprocessed_dynamic': {'mean_score': 0.724, 'std_score': 0.075},
        'original_dynamic': {'mean_score': 0.709, 'std_score': 0.082}
    },
    # Modificato con gli iperparametri
    'BoostedDecisionTree': {
        't2_medsam': {'mean_score': 0.638, 'std_score': 0.079},
        't2_preprocessed': {'mean_score': 0.677, 'std_score': 0.105},
        't2_original': {'mean_score': 0.697, 'std_score': 0.075},
        'medsam_dynamic': {'mean_score': 0.687, 'std_score': 0.086},
        'preprocessed_dynamic': {'mean_score': 0.708, 'std_score': 0.091},
        'original_dynamic': {'mean_score': 0.689, 'std_score': 0.037}
    }
}


# ==========================
# CREA LA TABELLA COMBINATA
# ==========================

table_data = []

for model_name, datasets_results in all_models_results.items():
    for dataset_name, metrics in datasets_results.items():
        table_data.append({
            'Modello': model_name,
            'Dataset': dataset_name,
            'F1-Score (Media ± Std)': f"{metrics['mean_score']:.3f} ± {metrics['std_score']:.3f}"
        })

df_results = pd.DataFrame(table_data)
df_pivot = df_results.pivot(index='Modello', columns='Dataset', values='F1-Score (Media ± Std)')

# Stampo la tabella
print(df_pivot.to_string())

# Lo stampo in un formato per latex
print(df_pivot.to_markdown())


# Salvo in CSV
df_pivot.to_csv('risultati.csv')
