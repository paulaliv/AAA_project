import pandas as pd

## dir loading
plasma_data = pd.read_excel('plasma_data_imputed_30.xlsx')
tissue_data = pd.read_excel('tissue_data_imputed_30.xlsx')

assert plasma_data.shape[1] == tissue_data.shape[1]

plasma_proteins = plasma_data['Genes'].values
tissue_proteins = tissue_data['Genes'].values

print(f'Number of plasma proteins: {len(plasma_proteins)}')
print(f'Number of tissue proteins: {len(tissue_proteins)}')


common_proteins = [protein for protein in tissue_proteins if protein in plasma_proteins]

print(common_proteins)
print(len(common_proteins))


#Pathway enrichment analysis
