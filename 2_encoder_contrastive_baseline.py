import numpy as np
import pandas as pd
import torch
import sys
import torch_geometric
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from sklearn.model_selection import KFold
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GlobalAttention
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import igraph
import gseapy as gp

from itertools import combinations
import copy

from torch_geometric.nn import global_mean_pool

#for gcn
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
#tissue_data = pd.read_csv('PR80_30/tissue_data_imputed_30.csv')
# patients = tissue_data.columns[2:].to_numpy()
# print(patients)
gene_to_idx_plasma = pd.read_csv('PR80_30/gene_to_idx_400_plasma.csv')
gene_to_idx_tissue = pd.read_csv('PR80_30/gene_to_idx_400_tissue.csv')


graphs_plasma = torch.load("PR80_30/proteins_graphs_400_plasma.pt", weights_only=False)
graphs_tissue = torch.load("PR80_30/proteins_graphs_400_tissue.pt", weights_only=False)

# # Separate sample IDs from pathway scores
# pathway_data = tissue_scores_common.drop(columns=["Name"])
#
# # Calculate mean score for each pathway across samples
# mean_scores = pathway_data.mean(axis=0)
# # print(mean_scores.describe())
# # Get the top 100 pathways
# top_100_pathways_mean = mean_scores.nlargest(286).index
#
# std_scores = pathway_data.std(axis=0)
# # print(std_scores.describe())
#
#
# top_100_pathways_std = std_scores.nlargest(100).index
#
# pathway_scores_top100_mean = tissue_scores_common[["Name"] + list(top_100_pathways_mean)]
# # Keep only those pathways
# pathway_scores_top100_std = tissue_scores_common[["Name"] + list(top_100_pathways_std)]
#
# top_scores= set(pathway_scores_top100_mean.columns).intersection(pathway_scores_top100_std.columns)
# top_scores.discard('Name')
# top_scores.discard('REACTOME_INFLUENZA_INFECTION')
# top_scores.discard('REACTOME_SARS_COV_1_MODULATES_HOST_TRANSLATION_MACHINERY')
# print(f'top pathways: {len(top_scores)}')
# for pathway in top_scores:
#     print(pathway)

# pd.DataFrame({"pathway":list(top_scores)}).to_csv(
#     "top_std_tissue_pathways.csv",
#     index=False
# )


print(graphs_plasma)
print(graphs_tissue)
g0= graphs_plasma[0]
g0_tissue = graphs_tissue[0]
for i, g in enumerate(graphs_plasma):
    assert torch.equal(g.edge_index, g0.edge_index)
    assert torch.equal(g.edge_attr, g0.edge_attr)

for i, g in enumerate(graphs_tissue):
    assert torch.equal(g.edge_index, g0_tissue.edge_index)
    assert torch.equal(g.edge_attr, g0_tissue.edge_attr)


tissue_dict = {
    g.patient_id:g
    for g in graphs_tissue
}
plasma_dict = {
    g.patient_id:g
    for g in graphs_plasma
}
print(plasma_dict)

patients = sorted(
    set(tissue_dict.keys())
    &
    set(plasma_dict.keys())
)
patients = np.array(patients)
print(patients)


class CreatePairs(Dataset):
    def __init__(
        self,
        patients,
        tissue_graphs,
        plasma_graphs
    ):

        self.patients = patients
        self.tissue_graphs = tissue_graphs
        self.plasma_graphs = plasma_graphs


    def __len__(self):

        return len(self.patients)


    def __getitem__(self, idx):

        patient = self.patients[idx]

        tissue = self.tissue_graphs[patient]

        plasma = self.plasma_graphs[patient]

        return tissue, plasma, patient

class GraphEncoder_Tissue(nn.Module):

    def __init__(self, hidden_dim=64):

        super().__init__()

        self.conv1 = SAGEConv(
            1,
            hidden_dim
        )

        self.conv2 = SAGEConv(
            hidden_dim,
            hidden_dim
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # self.conv1 = GCNConv(
        #     1,
        #     hidden_dim
        # )

        # self.conv2 = GCNConv(
        #     hidden_dim,
        #     hidden_dim
        # )

        # self.pool = global_mean_pool

        self.pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim,32),
                nn.ReLU(),
                nn.Linear(32,1)
            )
        )


    def forward(self,x,edge_index,batch):


        x = self.conv1(
            x,
            edge_index
        )

        x = torch.relu(x)


        x = self.conv2(
            x,
            edge_index
        )

        x = torch.relu(x)


        x = self.pool(
            x,
            batch
        )

        # x = global_mean_pool(
        #     x,
        #     batch
        # )

        #x = self.mlp(x)

        return x

class GraphEncoder_Plasma(nn.Module):

    def __init__(self, hidden_dim=64):

        super().__init__()

        self.conv1 = SAGEConv(
                        1,
                        hidden_dim
                    )

        self.conv2 = SAGEConv(
                        hidden_dim,
                        hidden_dim
                    )



        self.pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim,32),
                nn.ReLU(),
                nn.Linear(32,1)
            )
        )



    def forward(self,x,edge_index,batch):

        x= self.conv1(
            x,
            edge_index
        )

        x = torch.relu(x)


        x = self.conv2(
            x,
            edge_index
        )

        x = torch.relu(x)

        x = self.pool(x, batch)

        # x = global_mean_pool(
        #     x,
        #     batch
        # )
        #x= self.mlp(x)

        return x

class ProjectionHead_Shared(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,32)
        )


    def forward(self,x):
        return self.net(x)


class ProjectionHead_Plasma(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        return self.net(x)

class ProjectionHead_Tissue(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        return self.net(x)


class Plasma_Graph_Similarity(nn.Module):
    def __init__(self,
                 plasma_encoder,
                 shared_projector,
                 tissue_z):
        super().__init__()

        self.plasma_encoder = plasma_encoder
        self.shared_projector = shared_projector
        self.register_buffer(
            "tissue_z",
            tissue_z.detach()
        )


    def forward(self, x, edge_index, batch):
        plasma_emb = self.plasma_encoder(x, edge_index, batch)
        plasma_z = self.shared_projector(plasma_emb)
        similarity = F.cosine_similarity(
            plasma_z,
            self.tissue_z,
            dim=-1
        )
        return similarity


class Tissue_Graph_Similarity(nn.Module):
    def __init__(self,
                 tissue_encoder,
                 shared_projector,
                 plasma_z):
        super().__init__()

        self.tissue_encoder = tissue_encoder
        self.shared_projector = shared_projector
        self.register_buffer(
            "plasma_z",
            plasma_z.detach()
        )


    def forward(self,x, edge_index, batch):
        tissue_emb = self.tissue_encoder(x, edge_index, batch)
        tissue_z = self.shared_projector(tissue_emb)
        similarity = F.cosine_similarity(
            tissue_z,
            self.plasma_z,
            dim=-1
        )
        return similarity


def pair_collate(batch):

    tissues = []
    plasmas = []

    for tissue, plasma in batch:

        tissues.append(tissue)
        plasmas.append(plasma)


    tissue_batch = Batch.from_data_list(
        tissues
    )

    plasma_batch = Batch.from_data_list(
        plasmas
    )


    return tissue_batch, plasma_batch

def symmetric_contrastive_loss(
    tissue_z,
    plasma_z,
    temperature=0.1
):

    # normalize embeddings
    tissue_z = F.normalize(
        tissue_z,
        dim=1
    )

    plasma_z = F.normalize(
        plasma_z,
        dim=1
    )

    # similarity matrix
    logits = (
        tissue_z @ plasma_z.T
    ) / temperature

    print(f'Loss logits: {logits[:3, :3]}')


    labels = torch.arange(
        logits.size(0),
        device=logits.device
    )


    # tissue -> plasma
    loss_tissue = F.cross_entropy(
        logits,
        labels
    )


    # plasma -> tissue
    loss_plasma = F.cross_entropy(
        logits.T,
        labels
    )


    # symmetric loss
    loss = (
        loss_tissue +
        loss_plasma
    ) / 2


    return loss


def node_to_edge_importance(edge_index, node_importance):
    source = edge_index[0]
    target = edge_index[1]

    edge_importance = (
                              node_importance[source] +
                              node_importance[target]
                      ) / 2.0

    return edge_importance


def find_communities_safe(
        edge_index,
        edge_masks=None,
        detection_alg="louvain"
):
    assert detection_alg in [
        "louvain",
        "opt_modularity"
    ]

    edge_index = np.asarray(edge_index, dtype=int)

    if edge_masks is not None:
        edge_masks = np.abs(
            np.asarray(edge_masks, dtype=float)
        )

    edges = [
        (int(row[0]), int(row[1]))
        for row in edge_index.T
    ]

    nodes = set(edge_index.flatten())

    g = igraph.Graph()

    g.add_vertices(max(nodes) + 1)

    g.add_edges(edges)

    if edge_masks is not None:
        g.es["weight"] = edge_masks

    if detection_alg == "louvain":

        partition = g.community_multilevel(
            weights=edge_masks
        )

    elif detection_alg == "opt_modularity":

        partition = g.community_optimal_modularity(
            weights=edge_masks
        )

    valid_communities = []
    avg_edge_masks = []

    for community in partition:

        community = list(community)

        # All possible pairs of nodes in this community
        community_pairs = set(
            combinations(community, r=2)
        )

        # Only retain PPI edges that actually exist
        internal_edges = [
            edge
            for edge in edges
            if edge in community_pairs
        ]

        # Ignore communities without internal PPI edges
        if len(internal_edges) == 0:
            continue

        valid_communities.append(community)

        if edge_masks is not None:

            weights = []

            for edge in internal_edges:
                edge_id = g.get_eid(
                    edge[0],
                    edge[1]
                )

                weights.append(
                    g.es[edge_id]["weight"]
                )

            avg_edge_masks.append(
                np.mean(weights)
            )

    return (
        avg_edge_masks,
        valid_communities
    )


def explanation_dict_to_df(explainer_results, idx_to_gene):
    """Convert fold-level GNNExplainer results to a tidy DataFrame."""

    all_explanations = []

    for fold, fold_results in explainer_results.items():

        for sample_key, explanation in fold_results.items():

            sample = sample_key[0]
            importance = explanation["mean"].squeeze()

            for idx, score in enumerate(importance):
                all_explanations.append({
                    "fold": fold,
                    "sample": sample,
                    "idx": idx,
                    "gene": idx_to_gene[idx],
                    "importance": float(score)
                })

    return pd.DataFrame(all_explanations)

    # ==================================================
    # Helper: create patient-specific subnetworks
    # ==================================================

def create_subnetworks(
        explanations_df,
        idx_to_gene,
        edge_index,
        label
):

    all_subnetworks = []

    samples = explanations_df["sample"].unique()

    for sample in samples:

        print(f"\nProcessing {label} | {sample}")

        sample_df = explanations_df[
            explanations_df["sample"] == sample
        ]

        node_importance = (
            sample_df
            .sort_values("idx")
            .set_index("idx")["importance"]
            .reindex(range(len(idx_to_gene)))
            .fillna(0)
            .values
        )

        # Node importance → edge importance
        edge_importance = node_to_edge_importance(
            edge_index,
            node_importance
        )

        threshold = np.percentile(edge_importance, 95)

        keep = edge_importance >= threshold

        edge_index_filtered = edge_index[:, keep]
        edge_importance_filtered = edge_importance[keep]

        # Community detection
        avg_edge_masks, communities = find_communities_safe(
            edge_index=edge_index_filtered,
            edge_masks=edge_importance_filtered,
            detection_alg="louvain"
        )

        print(
            f"Found {len(communities)} {label} communities"
        )

        # Store communities
        for community_idx, community in enumerate(communities):

            genes = [
                idx_to_gene[idx]
                for idx in community
                if idx in idx_to_gene
            ]

            all_subnetworks.append({
                "sample": sample,
                "subnetwork": community_idx,
                "n_nodes": len(community),
                "mean_edge_importance": avg_edge_masks[
                    community_idx
                ],
                "genes": genes
            })

    return pd.DataFrame(all_subnetworks)

def create_subnetworks_global(
        explanations_df,
        idx_to_gene,
        edge_index,
        label
):

    all_subnetworks = []
    node_importance = explanations_df.groupby('idx')['importance'].median()


    node_importance = (
        node_importance
        .reindex(range(len(idx_to_gene)))
        .fillna(0)
        .values
    )

    # Node importance → edge importance
    edge_importance = node_to_edge_importance(
        edge_index,
        node_importance
    )

    threshold = np.percentile(edge_importance, 99)

    keep = edge_importance >= threshold

    edge_index_filtered = edge_index[:, keep]
    edge_importance_filtered = edge_importance[keep]

    # Community detection
    avg_edge_masks, communities = find_communities_safe(
        edge_index=edge_index_filtered,
        edge_masks=edge_importance_filtered,
        detection_alg="louvain"
    )

    print(
        f"Found {len(communities)} {label} communities"
    )

    # Store communities
    for community_idx, community in enumerate(communities):

        genes = [
            idx_to_gene[idx]
            for idx in community
            if idx in idx_to_gene
        ]
        if (len(genes)) < 2:
            continue

        all_subnetworks.append({
            "subnetwork": community_idx,
            "n_nodes": len(community),
            "mean_edge_importance": avg_edge_masks[
                community_idx
            ],
            "genes": genes
        })


    return pd.DataFrame(all_subnetworks)

    # ==================================================
    # Helper: Reactome ORA
    # ==================================================

def run_ora(
        subnetworks_df,
        background_genes,
        output_path
):

    ora_results = []

    for _, row in subnetworks_df.iterrows():

        genes = row["genes"]

        if len(genes) == 1:
            continue

        enr = gp.enrichr(
            gene_list=genes,
            gene_sets="Reactome_2022",
            background=background_genes,
            organism="human",
            outdir=None
        )

        results = enr.results.copy()

        #results["sample"] = row["sample"]
        results["subnetwork"] = row["subnetwork"]

        ora_results.append(results)

    if not ora_results:
        print(f"No ORA results for {output_path}")
        return

    ora_df = pd.concat(
        ora_results,
        ignore_index=True
    )

    ora_significant_df = ora_df[
        ora_df["Adjusted P-value"] < 0.05
    ].copy()

    ora_significant_df.to_excel(
        output_path,
        index=False
    )

    print(
        f"Saved {len(ora_significant_df)} significant pathways → "
        f"{output_path}"
    )
def training_loop():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    explainer_fold_results_tissue = {
    }
    explainer_fold_results_plasma = {}

    shared_loss_all_folds = []
    shared_loss_val_all_folds = []

    retrieval_acc_all_folds = []
    retrieval_acc_tissue_all_folds =[]
    retrieval_acc_plasma_all_folds = []
    retrieval_acc_val_all_folds = []


    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    for fold, (train_idx, val_idx) in enumerate(kf.split(patients)):
        train_losses = []
        train_losses_shared = []

        val_losses = []
        val_losses_shared = []

        positive_similarities_emb = []
        negative_similarities_emb = []
        positive_similarities_proj = []
        negative_similarities_proj = []

        retrieval_accuracy_emb_train = []
        retrieval_accuracy_proj_train = []
        retrieval_accuracy_tissue_train = []
        retrieval_accuracy_plasma_train = []

        retrieval_accuracy_emb_val = []
        retrieval_accuracy_proj_val = []

        val_recall = []
        train_recall = []
        val_rank = []
        train_rank = []



        fold_embeddings = {"epoch0": {"tissue": {}, "plasma": {}},
                           "epoch99": {"tissue": {}, "plasma": {}}}

        fold_embeddings_proj = {
            "epoch0": {
                "tissue": {

                },
                "plasma": {

                }
            },
            "epoch99": {
                "tissue": {

                },
                "plasma": {

                }
            }
        }
        best_embeddings_proj = {
            "epoch":0,
            "tissue": {

            },
            "plasma": {

            }
        }

        print(
            f"Fold {fold + 1}"
        )


        train_patients = patients[train_idx]

        val_patients = patients[val_idx]

        train_dataset = CreatePairs(
            train_patients,
            tissue_dict,
            plasma_dict
        )

        val_dataset = CreatePairs(
            val_patients,
            tissue_dict,
            plasma_dict
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=True,
            drop_last=False,
            collate_fn=pair_collate
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=False,
            drop_last=False,
            collate_fn=pair_collate
        )


        encoder_tissue = GraphEncoder_Tissue(
            hidden_dim=64
        ).to(device)

        encoder_plasma = GraphEncoder_Plasma().to(device)

        projector_shared = ProjectionHead_Shared(
        ).to(device)


        optimizer = torch.optim.AdamW(
            list(encoder_tissue.parameters())
            +
            list(encoder_plasma.parameters())
            +
            list(projector_shared.parameters())
            ,
            lr=1e-4,
            weight_decay=1e-5
        )


        epochs = 100
        best_val_retrieval = 0
        for epoch in range(epochs):

            encoder_tissue.train()
            encoder_plasma.train()
            projector_shared.train()

            total_loss = 0
            total_loss_shared = 0


            correct_retrieval = 0
            total_samples = 0
            correct_retrieval_proj = 0


            correct_retrieval_proj_tissue = 0
            correct_retrieval_proj_plasma = 0

            all_ranks = []


            for tissue, plasma, patient in train_loader:

                tissue = tissue.to(device)
                plasma = plasma.to(device)

                # graph encoder

                tissue_emb = encoder_tissue(
                    tissue.x,
                    tissue.edge_index,
                    tissue.batch
                )

                plasma_emb = encoder_plasma(
                    plasma.x,
                    plasma.edge_index,
                    plasma.batch
                )


                tissue_emb_1 = F.normalize(tissue_emb, dim=1)
                plasma_emb_1 = F.normalize(plasma_emb, dim=1)

                tissue_sim = tissue_emb_1 @ tissue_emb_1.T
                plasma_sim = plasma_emb_1 @ plasma_emb_1.T
                cross_sim = tissue_emb_1 @ plasma_emb_1.T
                mask = ~torch.eye(
                    tissue_sim.size(0),
                    dtype=torch.bool,
                    device=tissue_sim.device
                )
                mask_cross =~torch.eye(
                    cross_sim.size(0),
                    dtype=torch.bool,
                    device=tissue_sim.device

                )
                predicted = cross_sim.argmax(dim=1)
                true = torch.arange(
                    cross_sim.size(0)
                )


                correct_retrieval += (
                        predicted == true
                ).sum().item()

                total_samples += cross_sim.size(0)



                # projection space

                tissue_proj_shared = projector_shared(
                    tissue_emb
                )

                plasma_proj_shared = projector_shared(
                    plasma_emb
                )


                ### Compute cosine similarity and retrieval accuracy for shared embeddings
                tissue_proj_norm = F.normalize(tissue_proj_shared, dim=1)
                plasma_proj_norm = F.normalize(plasma_proj_shared, dim=1)
                cross_sim_proj = tissue_proj_norm @ plasma_proj_norm.T

                predicted_proj = cross_sim_proj.argmax(dim=1)
                true_proj = torch.arange(
                    cross_sim_proj.size(0)
                )
                correct_retrieval_proj += (
                        predicted_proj == true_proj
                ).sum().item()



                ranks = torch.argsort(
                    torch.argsort(cross_sim_proj, dim=1, descending=True),
                    dim=1
                ) + 1

                true_ranks = ranks[torch.arange(cross_sim_proj.size(0)), torch.arange(cross_sim_proj.size(0))]
                all_ranks.extend(true_ranks.cpu().tolist())


                ### Similarity Loss: currently contrastive loss. Could add CMD loss for general distribution alignment(not patient specifc)
                loss_shared = symmetric_contrastive_loss(
                    tissue_proj_shared,
                    plasma_proj_shared
                )

                loss_total = loss_shared

                optimizer.zero_grad()

                loss_total.backward()


                optimizer.step()

                total_loss_shared += loss_shared.item()

                total_loss += loss_total.item()

            epoch_recall_5 = np.mean(
                np.array(all_ranks) <= 5
            )
            train_recall.append(epoch_recall_5)

            epoch_train_loss = total_loss / len(train_loader)
            epoch_train_loss_shared = total_loss_shared / len(train_loader)

            train_losses.append(epoch_train_loss)
            train_losses_shared.append(epoch_train_loss_shared)


            epoch_retrieval_accuracy = (correct_retrieval / total_samples)
            epoch_retrieval_accuracy_proj = (
                    correct_retrieval_proj / total_samples
            )
            epoch_retrieval_accuracy_tissue = (correct_retrieval_proj_tissue/ total_samples)
            epoch_retrieval_accuracy_plasma = (correct_retrieval_proj_plasma / total_samples)

            median_rank = np.median(np.array(all_ranks))
            train_rank.append(median_rank)

            retrieval_accuracy_emb_train.append(epoch_retrieval_accuracy)
            retrieval_accuracy_proj_train.append(epoch_retrieval_accuracy_proj)
            retrieval_accuracy_tissue_train.append(epoch_retrieval_accuracy_tissue)
            retrieval_accuracy_plasma_train.append(epoch_retrieval_accuracy_plasma)

            print(
                f"Epoch {epoch}: "
                f"Train Loss: {epoch_train_loss:.4f} | Train Loss Shared: {epoch_train_loss_shared:.4f} |\n "
            
                f"retrieval accuracy embedding: {epoch_retrieval_accuracy:.4f} | retrieval accuracy proj: {epoch_retrieval_accuracy_proj:.4f} | Median Rank: {median_rank} | \n"

            )

            val_loss_Total = 0
            val_loss_Total_shared = 0

            val_batches = 0
            correct_retrieval = 0
            correct_retrieval_proj = 0

            total_samples = 0
            all_ranks_val = []

            with torch.no_grad():
                encoder_tissue.eval()
                encoder_plasma.eval()
                projector_shared.eval()

                for tissue, plasma, patient in val_loader:

                    tissue = tissue.to(device)
                    plasma = plasma.to(device)

                    # graph encoder

                    tissue_emb = encoder_tissue(
                        tissue.x,
                        tissue.edge_index,
                        tissue.batch
                    )

                    plasma_emb = encoder_plasma(
                        plasma.x,
                        plasma.edge_index,
                        plasma.batch
                    )


                    tissue_proj_shared = projector_shared(
                        tissue_emb
                    )

                    plasma_proj_shared = projector_shared(
                        plasma_emb
                    )



                    ### Similarity Loss: currently contrastive loss. Could add CMD loss for general distribution alignment(not patient specifc)
                    val_loss_shared = symmetric_contrastive_loss(
                        tissue_proj_shared,
                        plasma_proj_shared
                    )


                    val_loss_total = val_loss_shared


                    val_loss_Total += val_loss_total.item()
                    val_loss_Total_shared += val_loss_shared.item()

                    val_batches += 1

                    tissue_proj_shared_norm = F.normalize(tissue_proj_shared, dim=1)
                    plasma_proj_shared_norm = F.normalize(plasma_proj_shared, dim=1)


                    tissue_emb_norm = F.normalize(
                        tissue_emb,
                        dim=1
                    )

                    plasma_emb_norm = F.normalize(
                        plasma_emb,
                        dim=1
                    )


                    similarity_matrix = tissue_emb_norm @ plasma_emb_norm.T

                    positive_similarity = similarity_matrix.diag()
                    mean_positive_similarity = positive_similarity.mean()
                    mask = ~torch.eye(
                        similarity_matrix.size(0),
                        dtype=torch.bool
                    )

                    negative_similarity = similarity_matrix[mask]
                    mean_negative_similarity = negative_similarity.mean()


                    predicted = similarity_matrix.argmax(dim=1)
                    true = torch.arange(
                        similarity_matrix.size(0)
                    )
                    correct_retrieval += (predicted == true).sum().item()
                    total_samples += similarity_matrix.size(0)


                    similarity_matrix_proj = tissue_proj_shared_norm @ plasma_proj_shared_norm.T


                    positive_similarity_proj = similarity_matrix_proj.diag()
                    mean_positive_similarity_proj = positive_similarity_proj.mean()
                    mask_proj = ~torch.eye(
                        similarity_matrix_proj.size(0),
                        dtype=torch.bool
                    )

                    negative_similarity_proj = similarity_matrix_proj[mask_proj]
                    mean_negative_similarity_proj = negative_similarity_proj.mean()

                    ranks = torch.argsort(
                        torch.argsort(similarity_matrix_proj, dim=1, descending=True),
                        dim=1
                    ) + 1

                    true_ranks = ranks[torch.arange(similarity_matrix_proj.size(0)), torch.arange(similarity_matrix_proj.size(0))]
                    all_ranks_val.extend(true_ranks.cpu().tolist())

                    predicted_proj = similarity_matrix_proj.argmax(dim=1)
                    true_proj= torch.arange(
                        similarity_matrix_proj.size(0)
                    )
                    correct_retrieval_proj += (
                            predicted_proj == true_proj
                    ).sum().item()



                    positive_similarities_emb.append(
                        mean_positive_similarity.item()
                    )
                    negative_similarities_emb.append(
                        mean_negative_similarity.item()
                    )

                    positive_similarities_proj.append(
                        mean_positive_similarity_proj.item()
                    )
                    negative_similarities_proj.append(
                        mean_negative_similarity_proj.item()
                    )

                    inspection = [0, epochs-1]
                    if (correct_retrieval_proj/total_samples)> best_val_retrieval:
                        best_val_retrieval = (correct_retrieval_proj/total_samples)
                        # patient IDs
                        tissue_ids = tissue.patient_id
                        plasma_ids = plasma.patient_id

                        epoch_name = f"epoch{epoch}"
                        torch.save(
                            encoder_plasma.state_dict(),
                            'best_plasma_encoder.pt'
                        )

                        torch.save(
                            encoder_tissue.state_dict(),
                            'best_tissue_encoder.pt'
                        )

                        torch.save(
                            projector_shared.state_dict(),
                            'best_shared_projector.pt'
                        )



                        best_embeddings_proj['epoch'] = epoch_name
                        for i, patient_id in enumerate(tissue_ids):

                            best_embeddings_proj["tissue"][patient_id] = \
                                tissue_proj_shared[i].cpu()



                        for i, patient_id in enumerate(plasma_ids):
                            best_embeddings_proj["plasma"][patient_id] = \
                                plasma_proj_shared[i].cpu()



                    if epoch in inspection:

                        # patient IDs
                        tissue_ids = tissue.patient_id
                        plasma_ids = plasma.patient_id

                        epoch_name = f"epoch{epoch}"

                        for i, patient_id in enumerate(tissue_ids):
                            fold_embeddings[epoch_name]["tissue"][patient_id] = tissue_emb[i]

                        for i, patient_id in enumerate(plasma_ids):
                            fold_embeddings[epoch_name]["plasma"][patient_id] = plasma_emb[i]

                        for i, patient_id in enumerate(tissue_ids):
                            fold_embeddings_proj[epoch_name]["tissue"][patient_id] = \
                                tissue_proj_shared[i].cpu()


                        for i, patient_id in enumerate(plasma_ids):
                            fold_embeddings_proj[epoch_name]["plasma"][patient_id] = \
                                plasma_proj_shared[i].cpu()


            epoch_val_loss = val_loss_Total / val_batches

            val_losses.append(epoch_val_loss)

            epoch_val_recall_5 = np.mean(
                np.array(all_ranks_val) <= 5
            )
            median_rank = np.median(all_ranks_val)

            epoch_val_loss_shared = val_loss_Total_shared / val_batches

            val_losses_shared.append(epoch_val_loss_shared)



            epoch_val_retrieval_accuracy = (
                    correct_retrieval / total_samples
            )

            epoch_val_retrieval_accuracy_proj = (
                    correct_retrieval_proj / total_samples
            )

            retrieval_accuracy_emb_val.append(epoch_val_retrieval_accuracy)
            retrieval_accuracy_proj_val.append(epoch_val_retrieval_accuracy_proj)
            val_recall.append(epoch_val_recall_5)
            val_rank.append(median_rank)


            print(
                f"Val Loss: {epoch_val_loss:.4f} | Val Loss Shared: {epoch_val_loss_shared:.4f} | \n"
                f"retrieval accuracy: embedding: {epoch_val_retrieval_accuracy:.4f}, proj: {epoch_val_retrieval_accuracy_proj:.4f} | \n"
            )

            epochs_range = range(1, epochs + 1)

        retrieval_acc_all_folds.append(retrieval_accuracy_proj_train)
        retrieval_acc_plasma_all_folds.append(retrieval_accuracy_plasma_train)
        retrieval_acc_tissue_all_folds.append(retrieval_accuracy_tissue_train)
        shared_loss_all_folds.append(train_losses_shared)


        retrieval_acc_val_all_folds.append(retrieval_accuracy_proj_val)
        shared_loss_val_all_folds.append(val_losses_shared)


        ############### GNN_EXPLAINER #################
        from torch_geometric.explain import Explainer
        from torch_geometric.explain import GNNExplainer
        from torch_geometric.explain import ModelConfig
        encoder_plasma.load_state_dict(
            torch.load('best_plasma_encoder.pt')
        )

        encoder_tissue.load_state_dict(
            torch.load('best_tissue_encoder.pt')
        )

        projector_shared.load_state_dict(
            torch.load('best_shared_projector.pt')
        )


        plasma_explainer_results ={}
        tissue_explainer_results ={}



        encoder_plasma.eval()
        encoder_tissue.eval()
        projector_shared.eval()
        # explanation loader
        val_explanation_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False
        )

        for tissue, plasma, patient in val_explanation_loader:

            with torch.no_grad():
                emb_tissue = encoder_tissue(tissue.x,
                    tissue.edge_index,
                    tissue.batch)
                emb_plasma = encoder_plasma(plasma.x,
                    plasma.edge_index,
                    plasma.batch)
                z_tissue = projector_shared(emb_tissue)
                z_plasma = projector_shared(emb_plasma)

            # ==================================================
            # 4. Tissue explanation (shared projector)
            # ==================================================

            tissue_masks = []

            for run in range(10):
                tissue_model = Tissue_Graph_Similarity(
                    encoder_tissue,
                    projector_shared,
                    z_plasma
                )

                tissue_explainer = Explainer(
                    model=tissue_model,
                    algorithm=GNNExplainer(
                        epochs=200,
                        lr=0.01
                    ),
                    explanation_type="model",
                    node_mask_type="object",
                    edge_mask_type=None,
                    model_config=ModelConfig(
                        mode="regression",
                        task_level="graph",
                        return_type="raw"
                    )
                )

                tissue_explanation = tissue_explainer(
                    tissue.x,
                    tissue.edge_index,
                    target=None,
                    batch=tissue.batch
                )

                tissue_masks.append(
                    tissue_explanation.node_mask.detach().cpu().numpy()
                )

            tissue_masks = np.array(tissue_masks)

            tissue_mean = tissue_masks.mean(axis=0)
            tissue_std = tissue_masks.std(axis=0)

            # ==================================================
            # 3. Plasma explanation
            # ==================================================

            plasma_masks = []

            for run in range(10):
                plasma_model = Plasma_Graph_Similarity(
                    encoder_plasma,
                    projector_shared,
                    z_tissue
                )

                plasma_explainer = Explainer(
                    model=plasma_model,
                    algorithm=GNNExplainer(
                        epochs=200,
                        lr=0.01
                    ),
                    explanation_type="model",
                    node_mask_type="object",
                    edge_mask_type=None,
                    model_config=ModelConfig(
                        mode="regression",
                        task_level="graph",
                        return_type="raw"
                    )
                )

                plasma_explanation = plasma_explainer(
                    plasma.x,
                    plasma.edge_index,
                    target=None,
                    batch=plasma.batch
                )

                plasma_masks.append(
                    plasma_explanation.node_mask.detach().cpu().numpy()
                )

            plasma_masks = np.array(plasma_masks)

            plasma_mean = plasma_masks.mean(axis=0)
            plasma_std = plasma_masks.std(axis=0)


            # ==================================================
            # 5. Save
            # ==================================================

            plasma_explainer_results[patient] = {
                "mean": plasma_mean,
                "std": plasma_std
            }

            tissue_explainer_results[patient] = {
                "mean": tissue_mean.copy(),
                "std": tissue_std.copy()
            }

            print("PLASMA EXPLAINER")
            print(plasma_explainer_results)

            print("TISSUE EXPLAINER")
            print(tissue_explainer_results)


            explainer_fold_results_tissue[fold] = tissue_explainer_results
            explainer_fold_results_plasma[fold] = plasma_explainer_results

            torch.save(
                explainer_fold_results_plasma,
                "training_plots/baseline/explainer_fold_results_plasma.pt"
            )

            torch.save(
                explainer_fold_results_tissue,
                "training_plots/baseline/explainer_fold_results_tissue.pt"
            )

        torch.save(  fold_embeddings,f"embeddings_baseline/fold_{fold}_embeddings.pt" )
        torch.save(fold_embeddings_proj, f"embeddings_baseline/fold_{fold}_embeddings_proj.pt")
        torch.save(best_embeddings_proj, f"embeddings_baseline/fold_{fold}_best_embeddings_proj.pt")

    # Loss plot
    plt.figure(figsize=(6, 4))

    plt.plot(
        epochs_range,
        np.array(shared_loss_all_folds).mean(axis=0),
        label="Contrastive Train loss"
    )

    plt.plot(
        epochs_range,
        np.array(shared_loss_val_all_folds).mean(axis=0),
        label="Contrastive Validation loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Contrastive Loss")
    plt.legend()
    plt.title(f"Loss over all folds")

    plt.tight_layout()

    plt.savefig(
        f"training_plots/baseline/shared_loss.png",
        dpi=300
    )

    plt.close()


    # Similarity plot proj
    plt.figure(figsize=(6, 4))

    plt.plot(
        epochs_range,
        np.array(retrieval_acc_all_folds).mean(axis=0),
        label="Retrieval accuracy train"
    )

    plt.plot(
        epochs_range,
        np.array(retrieval_acc_val_all_folds).mean(axis=0),
        label="Retrieval accuracy val"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Retrieval accuracy")
    plt.legend()
    plt.title(f"Retrieval accuracy shared embeddings")

    plt.tight_layout()

    plt.savefig(
        f"training_plots/baseline/retrieval_acc.png",
        dpi=300
    )

    plt.close()



    ######## get neat gnn_explainer results
    idx_to_gene_tissue = dict(
        zip(gene_to_idx_tissue["idx"], gene_to_idx_tissue["Gene"])
    )
    idx_to_gene_plasma = dict(
        zip(gene_to_idx_plasma["idx"], gene_to_idx_plasma["Gene"])
    )

    # --------------------------------------------------
    # Shared explanations
    # --------------------------------------------------

    tissue_explanations_df = explanation_dict_to_df(
        explainer_fold_results_tissue,
        idx_to_gene_tissue
    )

    plasma_explanations_df = explanation_dict_to_df(
        explainer_fold_results_plasma,
        idx_to_gene_plasma
    )



    # --------------------------------------------------
    # Save GNNExplainer results
    # --------------------------------------------------

    output_dir = "training_plots/baseline"

    tissue_explanations_df.to_csv(
        f"{output_dir}/tissue_gnn_explainer_results.csv",
        index=False
    )

    plasma_explanations_df.to_csv(
        f"{output_dir}/plasma_gnn_explainer_results.csv",
        index=False
    )



    ##### Create edge masks

    g0_plasma = graphs_plasma[0]
    g0_tissue = graphs_tissue[0]

    edge_index_tissue = g0_tissue.edge_index
    edge_index_plasma = g0_plasma.edge_index

    subnetworks_tissue = create_subnetworks_global(
        tissue_explanations_df,
        idx_to_gene_tissue,
        edge_index_tissue,
        "TISSUE shared"
    )

    subnetworks_plasma = create_subnetworks_global(
        plasma_explanations_df,
        idx_to_gene_plasma,
        edge_index_plasma,
        "PLASMA shared"
    )

    # ==================================================
    # Run ORA
    # ==================================================

    background_genes_tissue = set(
        idx_to_gene_tissue.values()
    )

    background_genes_plasma = set(
        idx_to_gene_plasma.values()
    )


    # Shared tissue
    run_ora(
        subnetworks_tissue,
        background_genes_tissue,
        "training_plots/baseline/tissue_subcommunities_enrichment_results_GLOBAL.xlsx"
    )


    # Shared plasma
    run_ora(
        subnetworks_plasma,
        background_genes_plasma,
        "training_plots/baseline/plasma_subcommunities_enrichment_results_GLOBAL.xlsx"
    )



training_loop()



