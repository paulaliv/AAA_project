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
# gene_edges_model = pd.read_csv('gene_edges_model.csv')

graphs_plasma = torch.load("PR80_30/proteins_graphs_400_plasma.pt", weights_only=False)
graphs_tissue = torch.load("PR80_30/proteins_graphs_400_tissue.pt", weights_only=False)
### graph structure
# 6559 nodes
# 2 node features per node
## node
# 3,581,542 directed edges
# 1 edge weight per edge

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

# tissue_graphs = [
#     g for g in graphs
#     if g.modality == 0
# ]
#
# plasma_graphs = [
#     g for g in graphs
#     if g.modality == 1
# ]
# print(len(tissue_graphs))
# print(len(plasma_graphs))

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

# print(len(patients))
#print(gene_to_idx)
#idx_to_gene = {v: k for k, v in gene_to_idx.items()}

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

        return tissue, plasma

class GraphEncoder_Tissue(nn.Module):

    def __init__(self, hidden_dim=64):

        super().__init__()

        self.conv1 = GCNConv(
            1,
            hidden_dim
        )
        #
        # self.conv2 = SAGEConv(
        #     hidden_dim,
        #     hidden_dim
        # )
        # self.conv1 = GCNConv(
        #     1,
        #     hidden_dim
        # )

        self.conv2 = GCNConv(
            hidden_dim,
            hidden_dim
        )

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

        return x

class GraphEncoder_Plasma(nn.Module):

    def __init__(self, hidden_dim=64):

        super().__init__()

        self.conv1 = GCNConv(
                        1,
                        hidden_dim
                    )

        self.conv2 = GCNConv(
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
        # self.pool = global_mean_pool


    def forward(self,x,edge_index,batch):

        x= self.conv1(
            x,
            edge_index
        )

        x = torch.relu(x)

        #
        x = self.conv2(
            x,
            edge_index
        )

        x = torch.relu(x)
        #
        # x = global_mean_pool(
        #     x,
        #     batch
        # )

        x = self.pool(
            x,
            batch
        )

        return x
# class GraphEncoder(nn.Module):
#
#     def __init__(
#         self,
#         hidden_dim=64
#     ):
#         super().__init__()
#
#         self.conv1 = GCNConv(
#             1,
#             hidden_dim
#         )
#
#         self.conv2 = GCNConv(
#             hidden_dim,
#             hidden_dim
#         )
#         self.dropout = nn.Dropout(0.3)
#
#
#     def forward(self, x, edge_index, batch):
#
#         x = self.conv1(
#             x,
#             edge_index
#         )
#
#         x = torch.relu(x)
#
#
#         x = self.conv2(
#             x,
#             edge_index
#         )
#
#         x = torch.relu(x)
#
#
#         x = global_mean_pool(
#             x,
#             batch
#         )
#         return x

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

def contrastive_loss(
    tissue_z,
    plasma_z,
    temperature=0.1
):

    tissue_z = nn.functional.normalize(
        tissue_z,
        dim=1
    )

    plasma_z = nn.functional.normalize(
        plasma_z,
        dim=1
    )


    logits = (
        tissue_z @ plasma_z.T
    ) / temperature

    print(f'Loss logits: {logits[:3, :3]}')


    labels = torch.arange(
        len(logits),
        device=logits.device
    )


    loss = nn.CrossEntropyLoss()(
        logits,
        labels
    )

    return loss
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

def orthogonal_loss(shared, private):

    shared = F.normalize(shared, dim=1)
    private = F.normalize(private, dim=1)

    return torch.mean(
        torch.sum(shared * private, dim=1)**2
    )

def training_loop():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    for fold, (train_idx, val_idx) in enumerate(kf.split(patients)):
        train_losses = []
        val_losses = []

        positive_similarities_emb = []
        negative_similarities_emb = []
        positive_similarities_proj = []
        negative_similarities_proj = []

        fold_embeddings = {
            "epoch0": {"tissue": {}, "plasma": {}},
            "epoch99": {"tissue": {}, "plasma": {}}
        }
        fold_embeddings_proj = {
            "epoch0": {"tissue": {}, "plasma": {}},
            "epoch99": {"tissue": {}, "plasma": {}}
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
            batch_size=32,
            shuffle=True,
            drop_last=True,
            collate_fn=pair_collate
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
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

        # projector_plasma = ProjectionHead_Plasma(
        # ).to(device)
        #
        # projector_tissue = ProjectionHead_Tissue(
        # ).to(device)

        optimizer = torch.optim.AdamW(
            list(encoder_tissue.parameters())
            +
            list(encoder_plasma.parameters())
            +
            list(projector_shared.parameters()),
            lr=3e-4,
            weight_decay=1e-5
        )


        epochs = 100

        for epoch in range(epochs):

            encoder_tissue.train()
            encoder_plasma.train()
            projector_shared.train()
            # projector_plasma.train()
            # projector_tissue.train()

            total_loss = 0

            for tissue, plasma in train_loader:
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

                print(
                    "Tissue ↔ Tissue:",
                    tissue_sim[mask].mean().item()
                )

                print(
                    "Plasma ↔ Plasma:",
                    plasma_sim[mask].mean().item()
                )

                print(
                    "Tissue ↔ Plasma positive:",
                    cross_sim.diag().mean().item()
                )

                print(
                    "Tissue ↔ Plasma negative:",
                    cross_sim[mask].mean().item()
                )
                # projection space
                #
                # tissue_proj = projector_tissue(
                #     tissue_emb
                # )
                #
                # plasma_proj = projector_tissue(
                #     plasma_emb
                # )

                tissue_proj_shared = projector_shared(
                    tissue_emb
                )

                plasma_proj_shared = projector_shared(
                    plasma_emb
                )


                loss_shared = symmetric_contrastive_loss(
                    tissue_proj_shared,
                    plasma_proj_shared
                )


                optimizer.zero_grad()

                loss_shared.backward()

                optimizer.step()

                total_loss += loss_shared.item()

            epoch_train_loss = total_loss / len(train_loader)
            train_losses.append(epoch_train_loss)

            print(
                f"Epoch {epoch}: "
                f"Loss: {epoch_train_loss:.4f}"
            )

            val_loss_total = 0
            val_batches = 0
            with torch.no_grad():
                encoder_tissue.eval()
                encoder_plasma.eval()
                projector_shared.eval()
                for tissue, plasma in val_loader:

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

                    # projection space

                    tissue_proj = projector_shared(
                        tissue_emb
                    )

                    plasma_proj = projector_shared(
                        plasma_emb
                    )

                    val_loss = symmetric_contrastive_loss(
                        tissue_proj,
                        plasma_proj
                    )


                    val_loss_total += val_loss.item()
                    val_batches += 1

                    tissue_proj_norm = F.normalize(tissue_proj, dim=1)
                    plasma_proj_norm = F.normalize(plasma_proj, dim=1)

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

                    print(
                        f'Mean positive similarity: {mean_positive_similarity:.4f}',
                        f'Mean negative similarity: {mean_negative_similarity:.4f}'
                    )

                    predicted = similarity_matrix.argmax(dim=1)
                    true = torch.arange(
                        similarity_matrix.size(0)
                    )

                    retrieval_accuracy = (
                            predicted == true
                    ).float().mean()
                    print(f'retrieval accuracy: {retrieval_accuracy}')

                    similarity_matrix_proj = tissue_proj_norm @ plasma_proj_norm.T

                    positive_similarity_proj = similarity_matrix_proj.diag()
                    mean_positive_similarity_proj = positive_similarity_proj.mean()
                    mask_proj = ~torch.eye(
                        similarity_matrix_proj.size(0),
                        dtype=torch.bool
                    )

                    negative_similarity_proj = similarity_matrix_proj[mask_proj]
                    mean_negative_similarity_proj = negative_similarity_proj.mean()

                    print(
                        f'Mean positive similarity projection space: {mean_positive_similarity_proj:.4f}',
                        f'Mean negative similarity projection space: {mean_negative_similarity_proj:.4f}'
                    )
                    predicted_proj = similarity_matrix_proj.argmax(dim=1)
                    true_proj= torch.arange(
                        similarity_matrix_proj.size(0)
                    )

                    retrieval_accuracy_proj = (
                            predicted_proj == true_proj
                    ).float().mean()
                    print(f'retrieval accuracy projection space: {retrieval_accuracy_proj}')

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
                    if epoch in inspection:
                        # move back to CPU
                        tissue_emb = tissue_emb.cpu()
                        plasma_emb = plasma_emb.cpu()

                        # patient IDs
                        tissue_ids = tissue.patient_id
                        plasma_ids = plasma.patient_id

                        epoch_name = f"epoch{epoch}"

                        for i, patient_id in enumerate(tissue_ids):
                            fold_embeddings[epoch_name]["tissue"][patient_id] = tissue_emb[i]

                        for i, patient_id in enumerate(plasma_ids):
                            fold_embeddings[epoch_name]["plasma"][patient_id] = plasma_emb[i]

                        for i, patient_id in enumerate(tissue_ids):
                            fold_embeddings_proj[epoch_name]["tissue"][patient_id] = tissue_proj[i]

                        for i, patient_id in enumerate(plasma_ids):
                            fold_embeddings_proj[epoch_name]["plasma"][patient_id] = plasma_proj[i]



            epoch_val_loss = val_loss_total / val_batches
            val_losses.append(epoch_val_loss)

            epochs_range = range(1, epochs + 1)

        # Loss plot
        plt.figure(figsize=(6, 4))

        plt.plot(
            epochs_range,
            train_losses,
            label="Train loss"
        )

        plt.plot(
            epochs_range,
            val_losses,
            label="Validation loss"
        )

        plt.xlabel("Epoch")
        plt.ylabel("InfoNCE loss")
        plt.legend()
        plt.title(f"Fold {fold + 1} Loss")

        plt.tight_layout()

        plt.savefig(
            f"training_plots/GCN_loss/fold_{fold + 1}_loss.png",
            dpi=300
        )

        plt.close()

        # Similarity plot
        plt.figure(figsize=(6, 4))

        plt.plot(
            epochs_range,
            positive_similarities_emb,
            label="Positive similarity"
        )

        plt.plot(
            epochs_range,
            negative_similarities_emb,
            label="Negative similarity"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Cosine similarity")
        plt.legend()
        plt.title(f"Fold {fold + 1} Similarity Embedding Space")

        plt.tight_layout()

        plt.savefig(
            f"training_plots/fold_{fold + 1}_similarity_emb.png",
            dpi=300
        )

        plt.close()

        # Similarity plot proj
        plt.figure(figsize=(6, 4))

        plt.plot(
            epochs_range,
            positive_similarities_proj,
            label="Positive similarity"
        )

        plt.plot(
            epochs_range,
            negative_similarities_proj,
            label="Negative similarity"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Cosine similarity")
        plt.legend()
        plt.title(f"Fold {fold + 1} Similarity Projection Space")

        plt.tight_layout()

        plt.savefig(
            f"training_plots/fold_{fold + 1}_similarity_proj.png",
            dpi=300
        )

        plt.close()

        torch.save(  fold_embeddings,f"embeddings_GCN/fold_{fold}_embeddings.pt" )
        torch.save(fold_embeddings_proj, f"embeddings_GCN/fold_{fold}_embeddings_proj.pt")

training_loop()
