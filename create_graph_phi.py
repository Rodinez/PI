import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('phi_results.txt', sep='\t')

df[['library', 'method', 'eps']] = df['Attack'].str.rsplit('_', n=2, expand=True)
df['eps'] = df['eps'].astype(float)
df['AvgPhi'] = df['AvgPhi'].astype(float)

custom_colors = [
    "yellow",  
    '#FF8C00',  
    "#D564B3",  
    "red",  
    "#4399CE",  
    "#5D71CC",  
    "#3ABE73"   
]




for dataset in ['RAF', 'FER']:
    subset = df[df['Dataset'] == dataset]
    plt.figure()
    for i, ((library, method), group) in enumerate(subset.groupby(['library', 'method'])):
        group_sorted = group.sort_values('eps')
        color = custom_colors[i % len(custom_colors)]  # usa cores na ordem definida
        plt.plot(group_sorted['eps'], group_sorted['AvgPhi'],
                 marker='o', label=f"{method} ({library})", color=color)
    plt.xlabel('ε (perturbação)')
    plt.ylabel('Φ (KL-divergence)')
    plt.title(f'Variação de Φ com ε no dataset {dataset}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
