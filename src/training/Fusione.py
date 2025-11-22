import pandas as pd

# Caricamento dei dataset
preprocessed_dynamic = pd.read_csv('/Users/francesco/Tesi/BC-ML4/dataset/cleaned/preprocessed_dynamic.csv')
t2_medsam_masks = pd.read_csv('/Users/francesco/Tesi/BC-ML4/dataset/cleaned/t2_medsam_masks.csv')
t2_preprocessed_masks = pd.read_csv('/Users/francesco/Tesi/BC-ML4/dataset/cleaned/t2_preprocessed_masks.csv')
medsam_dynamic = pd.read_csv('/Users/francesco/Tesi/BC-ML4/dataset/cleaned/medsam_dynamic.csv')
t2_original_masks = pd.read_csv('/Users/francesco/Tesi/BC-ML4/dataset/cleaned/t2_original_masks.csv')
original_dynamic = pd.read_csv('/Users/francesco/Tesi/BC-ML4/dataset/cleaned/original_dynamic.csv')

# Colonne comuni usate come chiave di merge
common_cols = ['Patient ID', 'lesion idx', 'tumor/benign', 'GRADE', 'ER [SII]', 'PR [SII]', 'HER2 [SII]', 
               'isTN', 'KI67 [%]', 'Breast', 'z_offset', 'y_offset', 'x_offset', 'Pixel Spacing', 'Slice Thickness']

# Unione progressiva dei dataset con chiave comune, evitando duplicati di colonne
merged_df = preprocessed_dynamic
for df in [medsam_dynamic, original_dynamic, t2_medsam_masks, t2_preprocessed_masks, t2_original_masks]:
    merged_df = pd.merge(merged_df, df, how='outer', on=common_cols, suffixes=('', '_dup'))
    # Rimuove colonne duplicate create dal merge
    merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith('_dup')]

print('Dimensione del dataset fuso:', merged_df.shape)

# Salvare il dataset fuso in un nuovo CSV
merged_df.to_csv('dataset_fuso.csv', index=False)
