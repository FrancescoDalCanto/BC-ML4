import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configurazione stile grafici
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

# ========================================================================
# CARICAMENTO DATI
# ========================================================================

# Carica tutti i file
bdt_merged = pd.read_csv('/Users/francesco/Tesi/BC-ML4/src/modelli_singoli/BoostedDecisionTree/bdt_merged.csv')
bdt_single = pd.read_csv('/Users/francesco/Tesi/BC-ML4/src/modelli_singoli/BoostedDecisionTree/bdt_single.csv')
rf_merged = pd.read_csv('/Users/francesco/Tesi/BC-ML4/src/modelli_singoli/RandomForest/rf_merged.csv')
rf_single = pd.read_csv('/Users/francesco/Tesi/BC-ML4/src/modelli_singoli/RandomForest/rf_single.csv')
rl_merged = pd.read_csv('/Users/francesco/Tesi/BC-ML4/src/modelli_singoli/RegressioneLogistica/rl_merged.csv')
rl_single = pd.read_csv('/Users/francesco/Tesi/BC-ML4/src/modelli_singoli/RegressioneLogistica/rl_single.csv')
xgb_merged = pd.read_csv('/Users/francesco/Tesi/BC-ML4/src/modelli_singoli/XGBoost/xgb_merge.csv')
xgb_single = pd.read_csv('/Users/francesco/Tesi/BC-ML4/src/modelli_singoli/XGBoost/xgb_single.csv')

# ========================================================================
# STANDARDIZZA COLONNE (MERGED vs SINGLE hanno nomi diversi)
# ========================================================================

def standardize_merged(df, model_name):
    """Standardizza formato merged"""
    df = df.copy()
    df['model'] = model_name
    df['approach'] = 'Merged'
    df = df.rename(columns={
        'target': 'Target',
        'mean_accuracy': 'Accuracy',
        'mean_balanced_accuracy': 'Balanced_Accuracy',
        'mean_f1': 'F1_score',
        'mean_auc': 'AUC'
    })
    return df[['model', 'approach', 'Target', 'Accuracy', 'Balanced_Accuracy', 'F1_score', 'AUC']]

def standardize_single(df, model_name):
    """Standardizza formato single"""
    df = df.copy()
    df['model'] = model_name
    df['approach'] = 'Single'
    df = df.rename(columns={
        'F1-score': 'F1_score',
        'Balanced Acc': 'Balanced_Accuracy'
    })
    return df[['model', 'approach', 'Target', 'Accuracy', 'Balanced_Accuracy', 'F1_score', 'AUC']]

# Standardizza tutti i dataframe
bdt_m = standardize_merged(bdt_merged, 'BDT')
bdt_s = standardize_single(bdt_single, 'BDT')
rf_m = standardize_merged(rf_merged, 'RF')
rf_s = standardize_single(rf_single, 'RF')
rl_m = standardize_merged(rl_merged, 'RL')
rl_s = standardize_single(rl_single, 'RL')
xgb_m = standardize_merged(xgb_merged, 'XGBoost')
xgb_s = standardize_single(xgb_single, 'XGBoost')

# Combina tutto
df_all = pd.concat([
    bdt_m, bdt_s, 
    rf_m, rf_s, 
    rl_m, rl_s, 
    xgb_m, xgb_s
], ignore_index=True)

# Pulisci Target (rimuovi " [SII]" se presente)
df_all['Target'] = df_all['Target'].str.replace(' \[SII\]', '', regex=True)

print("="*70)
print("DATASET COMBINATO")
print("="*70)
print(df_all.head(10))
print(f"\nShape: {df_all.shape}")
print(f"\nModelli: {df_all['model'].unique()}")
print(f"Approcci: {df_all['approach'].unique()}")
print(f"Target: {df_all['Target'].unique()}")

# ========================================================================
# GRAFICO 1: Confronto Accuracy per Target (Tutti i modelli)
# ========================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Confronto Single vs Merged - Performance per Target', 
             fontsize=16, fontweight='bold', y=0.995)

metrics = ['Accuracy', 'Balanced_Accuracy', 'F1_score', 'AUC']
titles = ['Accuracy', 'Balanced Accuracy', 'F1-score', 'AUC-ROC']

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[idx // 2, idx % 2]
    
    # Pivot per confronto
    pivot_data = df_all.pivot_table(
        index=['model', 'Target'], 
        columns='approach', 
        values=metric
    ).reset_index()
    
    # Calcola differenza (Merged - Single)
    pivot_data['Difference'] = pivot_data['Merged'] - pivot_data['Single']
    
    # Crea barplot
    x_pos = np.arange(len(pivot_data))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, pivot_data['Single'], width, 
                   label='Single', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, pivot_data['Merged'], width, 
                   label='Merged', color='coral', alpha=0.8)
    
    # Linea della differenza
    ax2 = ax.twinx()
    line = ax2.plot(x_pos, pivot_data['Difference'], 
                    'go-', linewidth=2, markersize=8, 
                    label='Δ (Merged - Single)')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_ylabel('Differenza', fontsize=10, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Labels
    ax.set_xlabel('Modello - Target', fontsize=10)
    ax.set_ylabel(title, fontsize=10)
    ax.set_title(f'{title} - Single vs Merged', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{row['model']}\n{row['Target']}" 
                        for _, row in pivot_data.iterrows()], 
                       rotation=45, ha='right', fontsize=9)
    ax.set_ylim([0, 1])
    ax.legend(loc='upper left', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✓ Grafico 1 salvato: comparison_single_vs_merged_all_metrics.png")

# ========================================================================
# GRAFICO 2: Heatmap delle differenze (Merged - Single)
# ========================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Heatmap Differenze: Merged - Single (valori positivi = Merged migliore)', 
             fontsize=16, fontweight='bold', y=0.995)

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[idx // 2, idx % 2]
    
    # Pivot per heatmap
    heatmap_data = df_all.pivot_table(
        index='Target', 
        columns=['model', 'approach'], 
        values=metric
    )
    
    # Calcola differenze
    diff_data = pd.DataFrame()
    for model in df_all['model'].unique():
        if (model, 'Merged') in heatmap_data.columns and (model, 'Single') in heatmap_data.columns:
            diff_data[model] = heatmap_data[(model, 'Merged')] - heatmap_data[(model, 'Single')]
    
    # Heatmap
    sns.heatmap(diff_data, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                vmin=-0.2, vmax=0.2, linewidths=1, linecolor='gray',
                cbar_kws={'label': 'Differenza'}, ax=ax)
    
    ax.set_title(f'{title} - Differenza (Merged - Single)', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Modello', fontsize=10)
    ax.set_ylabel('Target', fontsize=10)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/heatmap_differences_merged_single.png', 
            dpi=300, bbox_inches='tight')
plt.show()

print("✓ Grafico 2 salvato: heatmap_differences_merged_single.png")

# ========================================================================
# GRAFICO 3: Performance medie per approccio (aggregato)
# ========================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Calcola medie per approccio
mean_by_approach = df_all.groupby('approach')[metrics].mean()

# Barplot confronto
ax1 = axes[0]
x_pos = np.arange(len(metrics))
width = 0.35

bars1 = ax1.bar(x_pos - width/2, mean_by_approach.loc['Single'], width,
                label='Single', color='steelblue', alpha=0.8)
bars2 = ax1.bar(x_pos + width/2, mean_by_approach.loc['Merged'], width,
                label='Merged', color='coral', alpha=0.8)

# Aggiungi valori sopra le barre
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

ax1.set_xlabel('Metrica', fontsize=11)
ax1.set_ylabel('Score medio', fontsize=11)
ax1.set_title('Performance Medie - Single vs Merged\n(Aggregato su tutti i modelli e target)', 
              fontsize=12, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(titles)
ax1.legend(fontsize=10)
ax1.set_ylim([0, 1])
ax1.grid(axis='y', alpha=0.3)

# Differenze percentuali
ax2 = axes[1]
differences = ((mean_by_approach.loc['Merged'] - mean_by_approach.loc['Single']) / 
               mean_by_approach.loc['Single'] * 100)

colors = ['green' if x > 0 else 'red' for x in differences]
bars = ax2.barh(titles, differences, color=colors, alpha=0.7)

# Aggiungi valori
for idx, (bar, val) in enumerate(zip(bars, differences)):
    ax2.text(val, idx, f'{val:+.1f}%', 
            va='center', ha='left' if val > 0 else 'right',
            fontsize=10, fontweight='bold')

ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax2.set_xlabel('Differenza percentuale (%)', fontsize=11)
ax2.set_title('Merged vs Single\n(% miglioramento, positivo = Merged migliore)', 
              fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/average_performance_comparison.png', 
            dpi=300, bbox_inches='tight')
plt.show()

print("✓ Grafico 3 salvato: average_performance_comparison.png")

# ========================================================================
# TABELLA RIASSUNTIVA
# ========================================================================

print("\n" + "="*70)
print("TABELLA RIASSUNTIVA - SINGLE vs MERGED")
print("="*70)

summary = df_all.groupby(['approach', 'model', 'Target'])[metrics].mean().round(3)
print(summary)

# Salva tabella
summary.to_csv('/mnt/user-data/outputs/summary_single_vs_merged.csv')
print("\n✓ Tabella salvata: summary_single_vs_merged.csv")

# ========================================================================
# STATISTICHE FINALI
# ========================================================================

print("\n" + "="*70)
print("STATISTICHE AGGREGATE")
print("="*70)

print("\n📊 Performance medie per approccio:")
print(mean_by_approach.round(3))

print("\n📈 Differenze assolute (Merged - Single):")
diff_abs = (mean_by_approach.loc['Merged'] - mean_by_approach.loc['Single']).round(3)
print(diff_abs)

print("\n📈 Differenze percentuali (%):")
diff_pct = ((mean_by_approach.loc['Merged'] - mean_by_approach.loc['Single']) / 
            mean_by_approach.loc['Single'] * 100).round(1)
print(diff_pct)

# Conteggio vittorie
print("\n🏆 Conteggio 'vittorie' per approccio:")
wins = df_all.pivot_table(
    index=['model', 'Target'], 
    columns='approach', 
    values=metrics
)

win_counts = {'Single': 0, 'Merged': 0, 'Tie': 0}
for metric in metrics:
    for idx in range(len(wins)):
        single_val = wins[metric, 'Single'].iloc[idx]
        merged_val = wins[metric, 'Merged'].iloc[idx]
        
        if pd.notna(single_val) and pd.notna(merged_val):
            if abs(merged_val - single_val) < 0.001:
                win_counts['Tie'] += 1
            elif merged_val > single_val:
                win_counts['Merged'] += 1
            else:
                win_counts['Single'] += 1

print(f"  Single migliore:  {win_counts['Single']} volte")
print(f"  Merged migliore:  {win_counts['Merged']} volte")
print(f"  Pareggio:         {win_counts['Tie']} volte")

winner = 'MERGED' if win_counts['Merged'] > win_counts['Single'] else 'SINGLE'
print(f"\n🎯 VINCITORE COMPLESSIVO: {winner}")

print("\n" + "="*70)