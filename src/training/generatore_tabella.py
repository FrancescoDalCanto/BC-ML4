import pandas as pd

# Vanno inseriti a mano
all_models_results = {
    # ok
    'RandomForest': {
        't2_medsam': {'f1_score': 0.667, 'std_score': 0.096},
        't2_preprocessed': {'f1_score':  0.631, 'std_score': 0.066},
        't2_original': {'f1_score': 0.641, 'std_score': 0.070},
        'medsam_dynamic': {'f1_score': 0.633, 'std_score': 0.073},
        'preprocessed_dynamic': {'f1_score': 0.634, 'std_score': 0.055},
        'original_dynamic': {'f1_score': 0.657, 'std_score': 0.077}
    },
    # ok
    'LogisticRegression': {
        't2_medsam': {'f1_score': 0.294, 'std_score': 0.117},
        't2_preprocessed': {'f1_score': 0.244, 'std_score': 0.053},
        't2_original': {'f1_score': 0.244, 'std_score': 0.053},
        'medsam_dynamic': {'f1_score': 0.578, 'std_score': 0.015},
        'preprocessed_dynamic': {'f1_score': 0.578, 'std_score': 0.015},
        'original_dynamic': {'f1_score': 0.578, 'std_score': 0.015}
    },

    'XGBoost': {
        't2_medsam': {'f1_score': 0.649, 'std_score': 0.095},
        't2_preprocessed': {'f1_score': 0.608 , 'std_score': 0.104},
        't2_original': {'f1_score': 0.638, 'std_score': 0.079},
        'medsam_dynamic': {'f1_score': 0.552, 'std_score': 0.082},
        'preprocessed_dynamic': {'f1_score': 0.594, 'std_score': 0.075},
        'original_dynamic': {'f1_score': 0.628, 'std_score': 0.090}
    },
    # ok
    'BoostedDecisionTree': {
        't2_medsam': {'f1_score': 0.646, 'std_score': 0.093},
        't2_preprocessed': {'f1_score': 0.594, 'std_score': 0.076},
        't2_original': {'f1_score': 0.628, 'std_score': 0.049},
        'medsam_dynamic': {'f1_score': 0.588, 'std_score': 0.024},
        'preprocessed_dynamic': {'f1_score': 0.552, 'std_score': 0.056},
        'original_dynamic': {'f1_score': 0.560, 'std_score': 0.083}
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
            'F1-Score (Media ± Std)': f"{metrics['f1_score']:.3f} ± {metrics['std_score']:.3f}"
        })

df_results = pd.DataFrame(table_data)
df_pivot = df_results.pivot(index='Modello', columns='Dataset', values='F1-Score (Media ± Std)')

# Stampo la tabella
print(df_pivot.to_string())

# Lo stampo in un formato per latex
print(df_pivot.to_markdown())


# Salvo in CSV
df_pivot.to_csv('risultati.csv')
