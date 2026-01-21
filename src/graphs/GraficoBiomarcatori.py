from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, init



init(autoreset=True)



# Percorsi
CLEANED_FILE = Path('dataset/cleaned/ambl_lesions.csv')
OUTPUT_DIR = Path('report/report_biomarcatori/istogrammi')




# Configurazione biomarcatori
BIOMARCATORI_CONFIG = {
    'ER [SII]': {
        'labels': {0: 'Negativo', 1: 'Positivo'},
        'colors': ['#e74c3c', '#2ecc71']
    },
    'PR [SII]': {
        'labels': {0: 'Negativo', 1: 'Positivo'},
        'colors': ['#e74c3c', '#2ecc71']
    },
    'HER2 [SII]': {
        'labels': {0: 'Negativo', 1: 'Positivo'},
        'colors': ['#e74c3c', '#2ecc71']
    }
}





def plot_biomarkers(cleaned_file, output_dir):

    # Leggo il csv pulito
    df_after = pd.read_csv(cleaned_file)

    print(Fore.YELLOW + f"\nLesioni maligne dopo pulizia: {len(df_after)}")
    print(Fore.BLUE + "\n" + "="*70)
    print(Fore.BLUE + "GENERAZIONE GRAFICI DISTRIBUZIONE (SOLO LESIONI MALIGNE)")
    print(Fore.BLUE + "="*70 + "\n")

    for bio, config in BIOMARCATORI_CONFIG.items():
        data = pd.to_numeric(df_after[bio], errors='coerce')
        data = data.dropna()

        # Creo figura
        fig, ax = plt.subplots(figsize=(10, 6))

        # ER, PR, HER2 → binario
        negativo = (data == 0.0).sum()
        positivo = (data > 0.0).sum()
        categorie = ['Negativo', 'Positivo']
        valori = [negativo, positivo]
        colori = config['colors']

        # Plot barre
        bars = ax.bar(
            categorie,
            valori,
            color=colori,
            alpha=0.8,
            edgecolor='black',
            linewidth=2,
            width=0.6
        )

        # Titolo e assi
        ax.set_title(
            f'{bio} - Distribuzione delle Lesioni Maligne',
            fontsize=17,
            fontweight='bold',
            pad=18
        )
        ax.set_ylabel(
            'Numero di pazienti',
            fontsize=15,
            fontweight='bold'
        )

        # Griglia
        ax.grid(alpha=0.3, axis='y')

        # Limite asse Y
        y_max = max(valori) * 1.15
        ax.set_ylim(0, y_max)

        # Tick assi
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)

        for label in ax.get_xticklabels():
            label.set_fontweight('bold')

        # Etichette sopra le barre (valore + percentuale)
        for bar, val in zip(bars, valori):
            height = bar.get_height()
            pct = (val / len(data) * 100) if len(data) > 0 else 0
            offset = y_max * 0.02  # ~2% dell'altezza totale

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + offset,
                f'{val}\n({pct:.1f}%)',
                ha='center',
                va='bottom',
                fontsize=15,
                fontweight='bold'
            )


        plt.tight_layout()

        # Salvataggio
        filename = f'{bio.replace(" ", "_").replace("[", "").replace("]", "")}_distribuzione.png'
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(Fore.GREEN + f" Grafico creato: {filename}")
        print(Fore.CYAN + f"  Distribuzione: {dict(zip(categorie, valori))}")





if __name__ == "__main__":
    plot_biomarkers(CLEANED_FILE, OUTPUT_DIR)
