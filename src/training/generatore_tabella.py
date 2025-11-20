import pandas as pd

# Vanno inseriti a mano
all_models_results = {

    'RandomForest': {
        't2_medsam': {'mean_score': 0.760, 'std_score': 0.082},
        't2_preprocessed': {'mean_score': 0.763, 'std_score': 0.087},
        't2_original': {'mean_score': 0.745, 'std_score': 0.106},
        'medsam_dynamic': {'mean_score': 0.743, 'std_score': 0.075},
        'preprocessed_dynamic': {'mean_score': 0.739, 'std_score': 0.065},
        'original_dynamic': {'mean_score': 0.733, 'std_score': 0.063}
    },

    'LogisticRegression': {
        't2_medsam': {'mean_score': 0.738, 'std_score': 0.071},
        't2_preprocessed': {'mean_score': 0.774, 'std_score': 0.070},
        't2_original': {'mean_score': 0.773, 'std_score': 0.058},
        'medsam_dynamic': {'mean_score': 0.750, 'std_score': 0.052},
        'preprocessed_dynamic': {'mean_score': 0.754, 'std_score': 0.052},
        'original_dynamic': {'mean_score': 0.747, 'std_score': 0.050}
    },

    'XGBoost': {
        't2_medsam': {'mean_score': 0.778, 'std_score': 0.086},
        't2_preprocessed': {'mean_score': 0.763, 'std_score': 0.057},
        't2_original': {'mean_score': 0.752, 'std_score': 0.058},
        'medsam_dynamic': {'mean_score': 0.739, 'std_score': 0.049},
        'preprocessed_dynamic': {'mean_score': 0.759, 'std_score': 0.054},
        'original_dynamic': {'mean_score': 0.758, 'std_score': 0.057}
    },

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
