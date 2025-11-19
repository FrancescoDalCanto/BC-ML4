import pandas as pd

# Vanno inseriti a mano
all_models_results = {
    # Fatto
    'RandomForest': {
        't2_medsam': {'mean_score': 0.788, 'std_score': 0.083},
        't2_preprocessed': {'mean_score': 0.775, 'std_score': 0.060},
        't2_original': {'mean_score': 0.776, 'std_score': 0.072},
        'medsam_dynamic': {'mean_score': 0.755, 'std_score': 0.080},
        'preprocessed_dynamic': {'mean_score': 0.770, 'std_score': 0.031},
        'original_dynamic': {'mean_score': 0.760, 'std_score': 0.054}
    },
    # Non ho usato gli iperparametri
    'LogisticRegression': {
        't2_medsam': {'mean_score': 0.660, 'std_score': 0.079},
        't2_preprocessed': {'mean_score': 0.728, 'std_score': 0.062},
        't2_original': {'mean_score': 0.729, 'std_score': 0.075},
        'medsam_dynamic': {'mean_score': 0.708, 'std_score': 0.061},
        'preprocessed_dynamic': {'mean_score': 0.706, 'std_score': 0.103},
        'original_dynamic': {'mean_score': 0.672, 'std_score': 0.077}
    },

    'XGBoost': {
        't2_medsam': {'mean_score': 0.707, 'std_score': 0.105},
        't2_preprocessed': {'mean_score': 0.723, 'std_score': 0.107},
        't2_original': {'mean_score': 0.718, 'std_score': 0.051},
        'medsam_dynamic': {'mean_score': 0.688, 'std_score': 0.097},
        'preprocessed_dynamic': {'mean_score': 0.724, 'std_score': 0.075},
        'original_dynamic': {'mean_score': 0.709, 'std_score': 0.082}
    },
    # Fatto
    'BoostedDecisionTree': {
        't2_medsam': {'mean_score': 0.702, 'std_score': 0.0090},
        't2_preprocessed': {'mean_score': 0.710, 'std_score': 0.107},
        't2_original': {'mean_score': 0.706, 'std_score': 0.063},
        'medsam_dynamic': {'mean_score': 0.706, 'std_score': 0.108},
        'preprocessed_dynamic': {'mean_score': 0.709, 'std_score': 0.061},
        'original_dynamic': {'mean_score': 0.710, 'std_score': 0.075}
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
