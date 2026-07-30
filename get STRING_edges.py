import pandas as pd
import numpy as np
## dir loading
plasma_data = pd.read_excel('PR80_30/plasma_data_imputed_30.xlsx')
tissue_data = pd.read_excel('PR80_30/tissue_data_imputed_30.xlsx')

assert plasma_data.shape[1] == tissue_data.shape[1]

plasma_proteins = plasma_data['Genes'].values
tissue_proteins = tissue_data['Genes'].values

print(f'Number of plasma proteins: {len(plasma_proteins)}')
print(f'Number of tissue proteins: {len(tissue_proteins)}')


common_proteins = [protein for protein in tissue_proteins if protein in plasma_proteins]

#print(common_proteins)
print(f'Common proteins: {len(common_proteins)}')
all_proteins = list(set(plasma_proteins) | set(tissue_proteins))
print(f'all proteins: {len(all_proteins)}')
#print(set(plasma_proteins) & set(tissue_proteins))
# for protein in all_proteins:
#     print(protein)
#Pathway enrichment analysis
remote = False
if remote:

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

else:
    string_df = pd.read_csv(
        r"C:\Users\P095551\OneDrive - Amsterdam UMC\Bureaublad\pathway_datasets\9606.protein.links.v12.0.txt.gz",
        sep=" ",
        compression="gzip"
    )
    aliases = pd.read_csv(
        r"C:\Users\P095551\OneDrive - Amsterdam UMC\Bureaublad\pathway_datasets\9606.protein.aliases.v12.0.txt.gz",
        sep="\t",
        compression="gzip"
    )
    outdated_symbols = pd.read_csv(r"C:\Users\P095551\OneDrive - Amsterdam UMC\Bureaublad\pathway_datasets\hgnc_complete_set.txt", sep="\t")

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
# def clean_gene(gene):
#     # remove contaminants
#     gene = gene.split(";cRAP-")[0]
#
#     # take first protein from protein groups
#     gene = gene.split(";")[0]
#
#     return gene
# rename_dict = {}
# for protein in all_proteins:
#     new_name = clean_gene(protein)
#     if new_name != protein:
#         rename_dict[protein] = new_name
# rename_df = (
#     pd.DataFrame.from_dict(rename_dict, orient="index", columns=["new_gene"])
#     .rename_axis("original_gene")
#     .reset_index()
# )
#
# print(rename_df.head())

#
# all_proteins = [rename_dict.get(p, p) for p in all_proteins]
# print(rename_dict)
proteins_found_shared = [protein for protein in all_proteins if protein in gene2string.keys()]


print(f'Number of proteins found: {len(proteins_found_shared)}/{len(all_proteins)}')
leftover = [protein for protein in all_proteins if protein not in proteins_found_shared]
# for protein in leftover:
#     if protein in rename_dict_1.keys():
#         rename_dict[protein] = rename_dict_1[protein]
#         print(protein, rename_dict_1[protein])
# rename_df.to_csv('protein_renaming.csv', index=False)
# print('final mapping:',rename_dict)
# all_proteins = [rename_dict.get(p, p) for p in all_proteins]
# proteins_found = [protein for protein in all_proteins if protein in gene2string.keys()]

# print(f'Number of proteins found: {len(proteins_found)}/{len(all_proteins)}')
# leftover = [protein for protein in all_proteins if protein not in proteins_found]
print(f'proteins left: {leftover}')

# tissue_data["Genes"] = tissue_data["Genes"].map(
#     lambda x: rename_dict.get(x, x)
# )
#
# plasma_data["Genes"] = plasma_data["Genes"].map(
#     lambda x: rename_dict.get(x, x)
# )
proteins_found_plasma= [protein for protein in plasma_proteins if protein in gene2string.keys()]
proteins_found_tissue= [protein for protein in tissue_proteins if protein in gene2string.keys()]
proteins_found_common= [protein for protein in common_proteins if protein in gene2string.keys()]

print(f'Number of TISSUE proteins found: {len(proteins_found_tissue)}/{len(tissue_proteins)}')
print(f'Number of PLASMA proteins found: {len(proteins_found_plasma)}/{len(plasma_proteins)}')

print(f'Number of common proteins found: {len(proteins_found_common)}/{len(common_proteins)}')
leftover = pd.DataFrame(leftover, columns=["Genes not found in STRING"])
leftover.to_csv('PR80_30/genes_not_found_by_string.csv', index=False)
#tissue_data.to_csv('tissue_data_imputed_30_renamed.csv', index=False)
#plasma_data.to_csv('plasma_data_imputed_30_renamed.csv', index=False)

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
    gene_edges_collapsed["gene1"].isin(proteins_found_shared)
    &
    gene_edges_collapsed["gene2"].isin(proteins_found_shared)
]

gene_edges_model_plasma = gene_edges_collapsed[
    gene_edges_collapsed["gene1"].isin(plasma_proteins)
    &
    gene_edges_collapsed["gene2"].isin(plasma_proteins)
]
gene_edges_model_tissue = gene_edges_collapsed[
    gene_edges_collapsed["gene1"].isin(tissue_proteins)
    &
    gene_edges_collapsed["gene2"].isin(tissue_proteins)
]

gene_edges_model_common_only = gene_edges_collapsed[
    gene_edges_collapsed["gene1"].isin(common_proteins)
    &
    gene_edges_collapsed["gene2"].isin(common_proteins)
]

#gene_edges_model.to_csv('gene_edges_model.csv', index=False)

gene2string_df = pd.DataFrame(
    list(gene2string.items()),
    columns=["gene", "string_id"]
)

# gene2string_df.to_csv(
#     "gene2string.csv",
#     index=False
# )

graph_genes = set(
    gene_edges_model["gene1"]
).union(
    set(gene_edges_model["gene2"])
)


print('Number of edges:', len(gene_edges_model))
print('Actual number of nodes:',len(graph_genes))
model_400 = gene_edges_model[
    gene_edges_model["combined_score"] >= 400
]
genes_400 = set(
    model_400["gene1"]
).union(
    set(model_400["gene2"])
)
model_700 = gene_edges_model[
    gene_edges_model["combined_score"] >= 700
]
graph_genes_700 = set(
    model_700["gene1"]
).union(
    set(model_700["gene2"])
)

model_400_tissue = gene_edges_model_tissue[
    gene_edges_model_tissue["combined_score"] >= 400
]
model_700_tissue = gene_edges_model_tissue[
    gene_edges_model_tissue["combined_score"] >= 700
]

model_400_plasma= gene_edges_model_plasma[
    gene_edges_model_plasma["combined_score"] >= 400
]
model_700_plasma= gene_edges_model_plasma[
    gene_edges_model_plasma["combined_score"] >= 700
]

model_400_common= gene_edges_model_common_only[
    gene_edges_model_common_only["combined_score"] >= 400
]
model_700_common= gene_edges_model_common_only[
    gene_edges_model_common_only["combined_score"] >= 700
]

model_400_tissue.to_csv('PR80_30/gene_edges_model_tissue_400.csv', index=False)
model_700_tissue.to_csv('PR80_30/gene_edges_model_tissue_700.csv', index=False)


model_400_plasma.to_csv('PR80_30/gene_edges_model_plasma_400.csv', index=False)
model_700_plasma.to_csv('PR80_30/gene_edges_model_plasma_700.csv', index=False)


model_400_common.to_csv('PR80_30/gene_edges_model_common_400.csv', index=False)
model_700_common.to_csv('PR80_30/gene_edges_model_common_700.csv', index=False)


model_400.to_csv('PR80_30/gene_edges_model_union_400.csv', index=False)
model_700.to_csv('PR80_30/gene_edges_model_union_700.csv', index=False)

print('connectivity values stats:')

import pandas as pd

def graph_stats(edge_df, model_name):
    stats = {"Model": model_name}

    # Overall graph
    stats["Total edges"] = len(edge_df)
    stats["Total nodes"] = len(set(edge_df["gene1"]).union(edge_df["gene2"]))
    stats["Mean score"] = edge_df["combined_score"].mean()
    stats["Max score"] = edge_df["combined_score"].max()
    stats["Min score"] = edge_df["combined_score"].min()

    # Threshold = 400
    edges_400 = edge_df[edge_df["combined_score"] >= 400]
    nodes_400 = set(edges_400["gene1"]).union(edges_400["gene2"])

    stats["Edges ≥400"] = len(edges_400)
    stats["Nodes ≥400"] = len(nodes_400)

    # Threshold = 700
    edges_700 = edge_df[edge_df["combined_score"] >= 700]
    nodes_700 = set(edges_700["gene1"]).union(edges_700["gene2"])

    stats["Edges ≥700"] = len(edges_700)
    stats["Nodes ≥700"] = len(nodes_700)

    return stats

summary = pd.DataFrame([
    graph_stats(gene_edges_model, "Union"),
    graph_stats(gene_edges_model_tissue, "Tissue"),
    graph_stats(gene_edges_model_plasma, "Plasma"),
    graph_stats(gene_edges_model_common_only, "Common")
])

print(summary)

summary.to_csv("PR80_30/graph_summary_statistics.csv", index=False)
# for p in leftover:
#     print(
#         p,
#         aliases[aliases["alias"] == p]["source"].unique()
#     )