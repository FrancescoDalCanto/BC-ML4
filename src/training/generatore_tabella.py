import pandas as pd

# Vanno inseriti a mano
all_models_results = {

    'RandomForest': {
        't2_medsam': {'f1_score': 0.671, 'std_score': 0.097},
        't2_preprocessed': {'f1_score':  0.641, 'std_score': 0.066},
        't2_original': {'f1_score': 0.652, 'std_score': 0.070},
        'medsam_dynamic': {'f1_score': 0.614, 'std_score': 0.084},
        'preprocessed_dynamic': {'f1_score': 0.627, 'std_score': 0.057},
        'original_dynamic': {'f1_score': 0.646, 'std_score': 0.066}
    },
 
    'LogisticRegression': {
        't2_medsam': {'f1_score': 0.588, 'std_score': 0.057},
        't2_preprocessed': {'f1_score': 0.582, 'std_score': 0.084},
        't2_original': {'f1_score': 0.586, 'std_score': 0.063},
        'medsam_dynamic': {'f1_score': 0.542, 'std_score': 0.067},
        'preprocessed_dynamic': {'f1_score': 0.556, 'std_score': 0.081},
        'original_dynamic': {'f1_score': 0.524, 'std_score': 0.060}
    },

    'XGBoost': {
        't2_medsam': {'f1_score': 0.654, 'std_score': 0.100},
        't2_preprocessed': {'f1_score': 0.608 , 'std_score': 0.105},
        't2_original': {'f1_score': 0.642, 'std_score': 0.080},
        'medsam_dynamic': {'f1_score': 0.556, 'std_score': 0.088},
        'preprocessed_dynamic': {'f1_score': 0.603, 'std_score': 0.090},
        'original_dynamic': {'f1_score': 0.627, 'std_score': 0.083}
    },
    
    'BoostedDecisionTree': {
        't2_medsam': {'f1_score': 0.611, 'std_score': 0.089},
        't2_preprocessed': {'f1_score': 0.534, 'std_score': 0.055},
        't2_original': {'f1_score': 0.589, 'std_score': 0.059},
        'medsam_dynamic': {'f1_score': 0.544, 'std_score': 0.036},
        'preprocessed_dynamic': {'f1_score': 0.567, 'std_score': 0.092},
        'original_dynamic': {'f1_score': 0.579, 'std_score': 0.068}
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
