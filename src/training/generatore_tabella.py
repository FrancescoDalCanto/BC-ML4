import pandas as pd

# Vanno inseriti a mano
all_models_results = {
    'RandomForest': {
        't2_medsam': {'mean_score': 0.742, 'std_score': 0.050},
        't2_preprocessed': {'mean_score': 0.735, 'std_score': 0.087},
        't2_original': {'mean_score': 0.762, 'std_score': 0.075},
        'medsam_dynamic': {'mean_score': 0.704, 'std_score': 0.053},
        'preprocessed_dynamic': {'mean_score': 0.725, 'std_score': 0.049},
        'original_dynamic': {'mean_score': 0.733, 'std_score': 0.060}
    },
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
    'BoostedDecisionTree': {
        't2_medsam': {'mean_score': 0.758, 'std_score': 0.085},
        't2_preprocessed': {'mean_score': 0.717, 'std_score': 0.095},
        't2_original': {'mean_score': 0.694, 'std_score': 0.072},
        'medsam_dynamic': {'mean_score': 0.696, 'std_score': 0.108},
        'preprocessed_dynamic': {'mean_score': 0.699, 'std_score': 0.086},
        'original_dynamic': {'mean_score': 0.667, 'std_score': 0.082}
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
df_pivot = df_results.pivot(index='Modello', columns='Dataset', 
                              values='F1-Score (Media ± Std)')

# Stampo la tabella
print(df_pivot.to_string())

# Lo stampo in un formato per latex
#print(df_pivot.to_markdown())


# Salvo in CSV
df_pivot.to_csv('risultati.csv')
