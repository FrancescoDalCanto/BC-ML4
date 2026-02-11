import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Path dei file
base_path = '/Users/francesco/Tesi/BC-ML4/src/modelli_singoli/XGBoost/'
csv_files = [
    'feature_importance_AMBL_ER_medsam.csv',
    'feature_importance_AMBL_ER.csv',
    'feature_importance_DUKE_ER_medsam.csv',
    'feature_importance_DUKE_ER.csv'
]

# Titoli più leggibili
titles = [
    'AMBL ER MedSAM',
    'AMBL ER',
    'DUKE ER MedSAM',
    'DUKE ER'
]

# Colori professionali
colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']

# Crea un grafico separato per ogni dataset
for idx, (file, title) in enumerate(zip(csv_files, titles)):
    # Carica il CSV
    df = pd.read_csv(base_path + file)
    df_sorted = df.sort_values('importance', ascending=False)
    top_10 = df_sorted.head(10)
    
    # Stampa le top 10
    print(f"\n{'='*80}")
    print(f"TOP 10 FEATURES - {title}")
    print('='*80)
    print(top_10.to_string(index=False))
    
    # Crea figura singola
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Inverti per avere il migliore in alto
    top_10_reversed = top_10.iloc[::-1]
    
    # Grafico
    bars = ax.barh(range(len(top_10_reversed)), 
                    top_10_reversed['importance'].values,
                    color=colors[idx],
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=1.2)
    
    ax.set_yticks(range(len(top_10_reversed)))
    # Imposta i nomi in obliquo (rotation) e allineamento
    ax.set_yticklabels(top_10_reversed['feature'].values, 
                       fontsize=10, 
                       fontweight='bold',
                       rotation=15,  # Obliquo
                       ha='right')   # Allineamento a destra
    
    ax.set_xlabel('Importance', fontsize=14, fontweight='bold')
    ax.set_ylabel('Features', fontsize=14, fontweight='bold')
    ax.set_title(f'Top 10 Feature Importance - {title}', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Valori sulle barre
    for bar, value in zip(bars, top_10_reversed['importance'].values):
        ax.text(value, bar.get_y() + bar.get_height()/2, 
                f' {value:.4f}', 
                va='center', ha='left', fontsize=10, fontweight='bold')
    
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Aggiungi margini per evitare che i nomi vengano tagliati
    plt.tight_layout()
    plt.show()

print("\n✅ Tutte le visualizzazioni sono state mostrate con successo!")