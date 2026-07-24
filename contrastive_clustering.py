import pandas as pd
import numpy as np
## dir loading
plasma_data = pd.read_excel('plasma_data_imputed_30.xlsx')
tissue_data = pd.read_excel('tissue_data_imputed_30.xlsx')

assert plasma_data.shape[1] == tissue_data.shape[1]

plasma_proteins = plasma_data['Genes'].values
tissue_proteins = tissue_data['Genes'].values

print(f'Number of plasma proteins: {len(plasma_proteins)}')
print(f'Number of tissue proteins: {len(tissue_proteins)}')


common_proteins = [protein for protein in tissue_proteins if protein in plasma_proteins]

#print(common_proteins)
print(f' Common proteins: {len(common_proteins)}')
all_proteins = list(set(plasma_proteins) | set(tissue_proteins))
print(len(all_proteins))
#print(set(plasma_proteins) & set(tissue_proteins))
# for protein in all_proteins:
#     print(protein)
#Pathway enrichment analysis
string_df = pd.read_csv(
    "/scratch/bmep/plalfken/Venkat/pathway_datasets/9606.protein.links.v12.0.txt.gz",
    sep=" ",
    compression="gzip"
)
aliases= pd.read_csv(
    "/scratch/bmep/plalfken/Venkat/pathway_datasets/9606.protein.aliases.v12.0.txt.gz",
    sep="\t",
    compression="gzip"
)
outdated_symbols = pd.read_csv("/scratch/bmep/plalfken/Venkat/pathway_datasets/hgnc_complete_set.txt", sep="\t")
#print(outdated_symbols.columns)
#print(outdated_symbols[['prev_symbol','alias_symbol']].head())
#print(aliases_df.head())
#print(tissue_data.head())
rename_dict_1 = {}

for _, row in outdated_symbols.iterrows():
    current = row["symbol"]

    for col in ["prev_symbol", "alias_symbol"]:
        if pd.notna(row[col]):
            for old_symbol in row[col].split("|"):
                rename_dict_1[old_symbol] = current

#print(rename_dict_1)

# gene_aliases = aliases[
#     aliases["source"] == "Ensembl_HGNC"
# ]
gene_aliases = aliases[
    aliases["source"].isin([
        "Ensembl_HGNC",
        "Ensembl_HGNC_prev_symbol",
        "Ensembl_HGNC_alias_symbol",
        "Ensembl_external_synonym_HGNC",
        "BioMart_HUGO",
        "Ensembl_UniProt",
        "UniProt_GN_Name",
        "UniProt_GN_Synonyms",
        "KEGG_NAME",
        "KEGG_NAME_SYNONYM"
    ])
]
# gene2string = (
#     gene_aliases
#     .drop_duplicates("alias")
#     .set_index("alias")["#string_protein_id"]
#     .to_dict()
# )
gene2string = (
    gene_aliases
    .groupby("alias")["#string_protein_id"]
    .apply(list)
    .to_dict()
)
#print(gene2string)
def clean_gene(gene):
    # remove contaminants
    gene = gene.split(";cRAP-")[0]

    # take first protein from protein groups
    gene = gene.split(";")[0]

    return gene
rename_dict = {}
for protein in all_proteins:
    new_name = clean_gene(protein)
    if new_name != protein:
        rename_dict[protein] = new_name
rename_df = (
    pd.DataFrame.from_dict(rename_dict, orient="index", columns=["new_gene"])
    .rename_axis("original_gene")
    .reset_index()
)

print(rename_df.head())


all_proteins = [rename_dict.get(p, p) for p in all_proteins]
print(rename_dict)
proteins_found = [protein for protein in all_proteins if protein in gene2string.keys()]

print(f'Number of proteins found: {len(proteins_found)}/{len(all_proteins)}')
leftover = [protein for protein in all_proteins if protein not in proteins_found]
for protein in leftover:
    if protein in rename_dict_1.keys():
        rename_dict[protein] = rename_dict_1[protein]
        print(protein, rename_dict_1[protein])
rename_df.to_csv('protein_renaming.csv', index=False)
print('final mapping:',rename_dict)
all_proteins = [rename_dict.get(p, p) for p in all_proteins]
proteins_found = [protein for protein in all_proteins if protein in gene2string.keys()]

print(f'Number of proteins found: {len(proteins_found)}/{len(all_proteins)}')
leftover = [protein for protein in all_proteins if protein not in proteins_found]
print(f'proteins left: {leftover}')

tissue_data["Genes"] = tissue_data["Genes"].map(
    lambda x: rename_dict.get(x, x)
)

plasma_data["Genes"] = plasma_data["Genes"].map(
    lambda x: rename_dict.get(x, x)
)


tissue_data.to_csv('tissue_data_imputed_30_renamed.csv', index=False)
plasma_data.to_csv('plasma_data_imputed_30_renamed.csv', index=False)

hgnc_aliases = aliases[
    aliases["source"].isin([
        "Ensembl_HGNC",
        "BioMart_HUGO"
    ])
]

string2gene = (
    hgnc_aliases
    .drop_duplicates("#string_protein_id")
    .set_index("#string_protein_id")["alias"]
    .to_dict()
)

print('string2gene format:')
print(list(string2gene.items())[:5])

gene_edges = string_df.copy()

gene_edges["gene1"] = gene_edges["protein1"].map(string2gene)
gene_edges["gene2"] = gene_edges["protein2"].map(string2gene)

gene_edges = gene_edges.dropna(
    subset=["gene1", "gene2"]
)

print(gene_edges.head())

print(
    f"Mapped edges: {len(gene_edges)}/{len(string_df)}"
)


# make undirected edges
gene_edges[["gene_a", "gene_b"]] = np.sort(
    gene_edges[["gene1", "gene2"]],
    axis=1
)


# collapse isoform-level interactions
gene_edges_collapsed = (
    gene_edges
    .groupby(
        ["gene_a", "gene_b"],
        as_index=False
    )
    ["combined_score"]
    .max()
)


gene_edges_collapsed = gene_edges_collapsed.rename(
    columns={
        "gene_a": "gene1",
        "gene_b": "gene2"
    }
)


print("Original STRING edges:", len(string_df))
print("Mapped protein edges:", len(gene_edges))
print("Collapsed gene edges:", len(gene_edges_collapsed))

print(gene_edges_collapsed.head())
gene_edges_model = gene_edges_collapsed[
    gene_edges_collapsed["gene1"].isin(proteins_found)
    &
    gene_edges_collapsed["gene2"].isin(proteins_found)
]
gene_edges_model.to_csv('gene_edges_model.csv', index=False)
gene2string_df = pd.DataFrame(
    list(gene2string.items()),
    columns=["gene", "string_id"]
)

gene2string_df.to_csv(
    "gene2string.csv",
    index=False
)

graph_genes = set(
    gene_edges_model["gene1"]
).union(
    set(gene_edges_model["gene2"])
)


print('Number of edges:', len(gene_edges_model))
print('Actual number of nodes:',len(graph_genes))
connectivity_400 = gene_edges_model[
    gene_edges_model["combined_score"] >= 400
]
graph_genes_400 = set(
    connectivity_400["gene1"]
).union(
    set(connectivity_400["gene2"])
)
connectivity_700 = gene_edges_model[
    gene_edges_model["combined_score"] >= 700
]
graph_genes_700 = set(
    connectivity_700["gene1"]
).union(
    set(connectivity_700["gene2"])
)

print('connectivity values stats:')
print('edges above value of 400:',len(connectivity_400))
print('nodes with only edges above 400:',len(graph_genes_400))
print('edges above value of 700:',len(connectivity_700))
print('nodes with only edges above 700:',len(graph_genes_700))
print('mean:',gene_edges_model["combined_score"].mean())
print('max:',gene_edges_model["combined_score"].max())
print('min:',gene_edges_model["combined_score"].min())
# for p in leftover:
#     print(
#         p,
#         aliases[aliases["alias"] == p]["source"].unique()
#     )