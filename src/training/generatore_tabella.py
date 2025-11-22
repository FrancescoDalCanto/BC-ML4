import pandas as pd

# Vanno inseriti a mano
all_models_results = {
    # Modificato (fanno pena) => fatto
    'RandomForest': {
        't2_medsam': {'mean_score': 0.524, 'std_score': 0.034},
        't2_preprocessed': {'mean_score':  0.573, 'std_score': 0.100},
        't2_original': {'mean_score': 0.605, 'std_score': 0.138},
        'medsam_dynamic': {'mean_score': 0.539, 'std_score': 0.121},
        'preprocessed_dynamic': {'mean_score': 0.506, 'std_score': 0.107},
        'original_dynamic': {'mean_score': 0.494, 'std_score': 0.164}
    },
    # Modificato (fanno pena) => fatto
    'LogisticRegression': {
        't2_medsam': {'mean_score': 0.259, 'std_score': 0.086},
        't2_preprocessed': {'mean_score': 0.269, 'std_score': 0.094},
        't2_original': {'mean_score': 0.275, 'std_score': 0.095},
        'medsam_dynamic': {'mean_score': 0.481, 'std_score': 0.066},
        'preprocessed_dynamic': {'mean_score': 0.481, 'std_score': 0.066},
        'original_dynamic': {'mean_score': 0.481, 'std_score': 0.066}
    },
    # Modificato (fanno pena) => fatto
    'XGBoost': {
        't2_medsam': {'mean_score': 0.542, 'std_score': 0.062},
        't2_preprocessed': {'mean_score': 0.588, 'std_score': 0.129},
        't2_original': {'mean_score': 0.565, 'std_score': 0.133},
        'medsam_dynamic': {'mean_score': 0.536, 'std_score': 0.125},
        'preprocessed_dynamic': {'mean_score': 0.531, 'std_score': 0.117},
        'original_dynamic': {'mean_score': 0.482, 'std_score': 0.105}
    },
    # Modificato (fanno pena) => fatto
    'BoostedDecisionTree': {
        't2_medsam': {'mean_score': 0.491, 'std_score': 0.081},
        't2_preprocessed': {'mean_score': 0.534, 'std_score': 0.082},
        't2_original': {'mean_score': 0.540, 'std_score': 0.085},
        'medsam_dynamic': {'mean_score': 0.509, 'std_score': 0.126},
        'preprocessed_dynamic': {'mean_score': 0.488, 'std_score': 0.093},
        'original_dynamic': {'mean_score': 0.526, 'std_score': 0.143}
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
