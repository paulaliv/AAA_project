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

tissue_scores = pd.read_csv('PR80_30/tissue_pathway_scores.csv')
plasma_scores = pd.read_csv('PR80_30/plasma_pathway_scores.csv')
pathways_1 = set(tissue_scores.columns).intersection(plasma_scores.columns)

#pathways_1.remove('patient')
print(pathways_1)
tissue_scores_common = tissue_scores[list(pathways_1)]
plasma_scores_common = plasma_scores[list(pathways_1)]
print(tissue_scores_common.shape)
print(plasma_scores.shape)
print(tissue_scores.shape)
print(plasma_scores.head())

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




class Decoder(nn.Module):
    def __init__(self,shared_dim=32, private_dim=32, n_pathways=317):
        super().__init__()
        self.decoder = nn.Linear(
            shared_dim + private_dim,
            n_pathways
        )

    def forward(self, shared, private):
        x = torch.cat([shared, private], dim=1)
        return self.decoder(x)

def reconstruction_loss(pred, target):
    return F.mse_loss(pred, target)

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

# def orthogonal_loss(shared, private):
#
#     shared = F.normalize(shared, dim=1)
#     private = F.normalize(private, dim=1)
#
#     return torch.mean(
#         torch.sum(shared * private, dim=1)**2
#     )

def orthogonal_loss(shared, private):

    cross = shared.T @ private

    return torch.sum(cross ** 2)


def get_pathways(patients, pathway_scores):
    pathway_scores = pathway_scores[pathway_scores['Name'].isin(patients)]

    pathway_scores = pathway_scores.drop(
        columns=["Name"]
    )
    return pathway_scores

def training_loop(lamda_orth =0.1, lamda_recon =0.1):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    for fold, (train_idx, val_idx) in enumerate(kf.split(patients)):
        train_losses = []
        train_losses_shared = []
        train_losses_orth =[]
        train_losses_recon = []
        val_losses = []
        val_losses_shared = []
        val_losses_diff = []
        val_losses_recon = []

        positive_similarities_emb = []
        negative_similarities_emb = []
        positive_similarities_proj = []
        negative_similarities_proj = []
        retrieval_accuracy_emb_train = []
        retrieval_accuracy_proj_train = []
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
                    "shared": {},
                    "private": {}
                },
                "plasma": {
                    "shared": {},
                    "private": {}
                }
            },
            "epoch99": {
                "tissue": {
                    "shared": {},
                    "private": {}
                },
                "plasma": {
                    "shared": {},
                    "private": {}
                }
            }
        }
        best_embeddings_proj = {
            "epoch":0,
            "tissue": {
                "shared": {},
                "private": {}
            },
            "plasma": {
                "shared": {},
                "private": {}
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

        projector_plasma = ProjectionHead_Plasma(
        ).to(device)

        projector_tissue = ProjectionHead_Tissue(
        ).to(device)

        decoder_plasma = Decoder().to(device)

        optimizer = torch.optim.AdamW(
            list(encoder_tissue.parameters())
            +
            list(encoder_plasma.parameters())
            +
            list(projector_shared.parameters())
            +
            list(projector_plasma.parameters())
            +
            list(projector_tissue.parameters())
            +
            list(decoder_plasma.parameters())
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
            projector_plasma.train()
            projector_tissue.train()
            decoder_plasma.train()

            total_loss = 0
            total_loss_shared = 0
            total_loss_diff = 0
            total_loss_recon = 0

            correct_retrieval = 0
            total_samples = 0
            correct_retrieval_proj = 0

            all_ranks = []


            for tissue, plasma, patient in train_loader:
                print(f"Train batch size: {len(patient)}")
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

                # print(
                #     "Tissue ↔ Tissue:",
                #     tissue_sim[mask].mean().item()
                # )
                #
                # print(
                #     "Plasma ↔ Plasma:",
                #     plasma_sim[mask].mean().item()
                # )


                # rank of the true matching plasma sample


                print(
                    f'Mean similarity embedding space: positive similarity: {cross_sim.diag().mean().item():.4f} | negative similarity: {cross_sim[mask].mean().item():.4f} | Difference: {cross_sim.diag().mean().item() - cross_sim[mask_cross].mean().item():.4f}'
                )


                # projection space

                tissue_proj = projector_tissue(
                    tissue_emb
                )

                plasma_proj = projector_plasma(
                    plasma_emb
                )

                tissue_proj_shared = projector_shared(
                    tissue_emb
                )

                plasma_proj_shared = projector_shared(
                    plasma_emb
                )
                ### Compute cosine similarity and retrieval accuracy
                tissue_proj_norm = F.normalize(tissue_proj_shared, dim=1)
                plasma_proj_norm = F.normalize(plasma_proj_shared, dim=1)
                cross_sim_proj = tissue_proj_norm @ plasma_proj_norm.T

                print(
                    f'Mean similarity projection space: positive similarity: {cross_sim_proj.diag().mean().item():.4f} | negative similarity: {cross_sim_proj[mask].mean().item():.4f} | Difference: {cross_sim_proj.diag().mean().item() - cross_sim_proj[mask].mean().item():.4f}'
                )

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



                #### reconstruction loss
                ## standardize pathway scores

                plasma_pathways = get_pathways(patient, plasma_scores)
                scaler = StandardScaler()
                plasma_pathways_scaled = scaler.fit_transform(
                    plasma_pathways
                )
                pathway_scores_scaled = torch.tensor(
                    plasma_pathways_scaled,
                    dtype=torch.float32,
                    device=device
                )

                decoded_plasma = decoder_plasma(plasma_proj_shared,plasma_proj)
                loss_recon_plasma = reconstruction_loss(decoded_plasma, pathway_scores_scaled)




                ### Difference Loss
                #Difference between modality-invariant and specific representations
                loss_diff = orthogonal_loss(plasma_proj_shared, plasma_proj) + orthogonal_loss(tissue_proj_shared, tissue_proj)
                #Difference between the two modality specific representations
                #loss_diff_2 = orthogonal_loss(plasma_proj, tissue_proj)

                #loss_diff = loss_diff_1 + loss_diff_2

                ### Similarity Loss: currently contrastive loss. Could add CMD loss for general distribution alignment(not patient specifc)
                loss_shared = symmetric_contrastive_loss(
                    tissue_proj_shared,
                    plasma_proj_shared
                )

                loss_total = loss_shared + lamda_orth * loss_diff + lamda_recon * loss_recon_plasma


                optimizer.zero_grad()

                loss_total.backward()
                # print(
                #     "Tissue encoder gradient:",
                #     encoder_tissue.conv1.lin_l.weight.grad.norm().item()
                # )
                #
                # print(
                #     "Plasma encoder gradient:",
                #     encoder_plasma.conv1.lin_l.weight.grad.norm().item()
                # )

                optimizer.step()

                total_loss_shared += loss_shared.item()
                total_loss_recon += loss_recon_plasma.item()
                total_loss_diff += loss_diff.item()
                total_loss += loss_total.item()

            epoch_recall_5 = np.mean(
                np.array(all_ranks) <= 5
            )
            train_recall.append(epoch_recall_5)

            epoch_train_loss = total_loss / len(train_loader)
            epoch_train_loss_shared = total_loss_shared / len(train_loader)
            epoch_train_loss_diff = total_loss_diff / len(train_loader)
            epoch_train_loss_recon = total_loss_recon/len(train_loader)

            train_losses.append(epoch_train_loss)
            train_losses_shared.append(epoch_train_loss_shared)
            train_losses_orth.append(lamda_orth*epoch_train_loss_diff)
            train_losses_recon.append(lamda_recon*epoch_train_loss_recon)

            epoch_retrieval_accuracy = (correct_retrieval / total_samples)
            epoch_retrieval_accuracy_proj = (
                    correct_retrieval_proj / total_samples
            )

            median_rank = np.median(np.array(all_ranks))
            train_rank.append(median_rank)

            retrieval_accuracy_emb_train.append(epoch_retrieval_accuracy)
            retrieval_accuracy_proj_train.append(epoch_retrieval_accuracy_proj)

            print(
                f"Epoch {epoch}: "
                f"Train Loss: {epoch_train_loss:.4f} |Train Loss Shared: {epoch_train_loss_shared:.4f} | Train Loss Orth: {lamda_orth*epoch_train_loss_diff:.4f} | | Train Loss Recon: {lamda_orth*epoch_train_loss_recon:.4f} | retrieval accuracy embedding: {epoch_retrieval_accuracy:.4f} | retrieval accuracy proj: {epoch_retrieval_accuracy_proj:.4f} | Median Rank: {median_rank} | Recall@5: {epoch_recall_5:.4f}"
            )

            val_loss_Total = 0
            val_loss_Total_shared = 0
            val_loss_Total_diff = 0
            val_loss_Total_recon = 0
            val_batches = 0
            correct_retrieval = 0
            correct_retrieval_proj = 0
            total_samples = 0
            all_ranks_val = []
            with torch.no_grad():
                encoder_tissue.eval()
                encoder_plasma.eval()
                projector_shared.eval()
                projector_tissue.eval()
                projector_plasma.eval()
                for tissue, plasma, patient in val_loader:
                    print(f'Val batch size: {len(patient)}')
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
                    tissue_proj = projector_tissue(
                        tissue_emb
                    )

                    plasma_proj = projector_plasma(
                        plasma_emb
                    )

                    tissue_proj_shared = projector_shared(
                        tissue_emb
                    )

                    plasma_proj_shared = projector_shared(
                        plasma_emb
                    )

                    plasma_pathways = get_pathways(patient, plasma_scores)
                    plasma_pathways_scaled = scaler.transform(plasma_pathways)
                    plasma_pathways_scaled = torch.tensor(
                        plasma_pathways_scaled,
                        dtype=torch.float32,
                        device=device
                    )

                    decoded_plasma = decoder_plasma(plasma_proj_shared,plasma_proj)
                    val_loss_recon = reconstruction_loss(decoded_plasma, plasma_pathways_scaled)

                    val_loss_diff = orthogonal_loss(plasma_proj_shared, plasma_proj) + orthogonal_loss(tissue_proj_shared,
                                                                                                     tissue_proj)
                    # Difference between the two modality specific representations
                    #val_loss_diff_2 = orthogonal_loss(plasma_proj, tissue_proj)
                    #
                    # val_loss_diff = val_loss_diff_1 + val_loss_diff_2

                    ### Similarity Loss: currently contrastive loss. Could add CMD loss for general distribution alignment(not patient specifc)
                    val_loss_shared = symmetric_contrastive_loss(
                        tissue_proj_shared,
                        plasma_proj_shared
                    )


                    val_loss_total = val_loss_shared + lamda_orth * val_loss_diff + lamda_recon*val_loss_recon


                    val_loss_Total += val_loss_total.item()
                    val_loss_Total_shared += val_loss_shared.item()
                    val_loss_Total_diff += val_loss_diff.item()
                    val_loss_Total_recon += val_loss_recon.item()
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

                    print(
                        f'Mean similarity embedding space: positive similarity: {mean_positive_similarity:.4f} | negative similarity: {mean_negative_similarity:.4f} | Difference: {mean_negative_similarity - mean_positive_similarity:.4f}'
                    )


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

                    print(
                        f'Mean similarity projection space: positive similarity: {mean_positive_similarity_proj:.4f} | negative similarity: {mean_negative_similarity_proj:.4f} | Difference: {mean_positive_similarity_proj - mean_negative_similarity_proj:.4f}'
                    )

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
                        best_epoch = epoch
                        # move back to CPU
                        # tissue_emb = tissue_emb.cpu()
                        # plasma_emb = plasma_emb.cpu()

                        # patient IDs
                        tissue_ids = tissue.patient_id
                        plasma_ids = plasma.patient_id

                        epoch_name = f"epoch{epoch}"
                        best_embeddings_proj['epoch'] = epoch_name
                        for i, patient_id in enumerate(tissue_ids):

                            best_embeddings_proj["tissue"]["shared"][patient_id] = \
                                tissue_proj_shared[i].cpu()

                            best_embeddings_proj["tissue"]["private"][patient_id] = \
                                tissue_proj[i].cpu()

                        for i, patient_id in enumerate(plasma_ids):
                            best_embeddings_proj["plasma"]["shared"][patient_id] = \
                                plasma_proj_shared[i].cpu()

                            best_embeddings_proj["plasma"]["private"][patient_id] = \
                                plasma_proj[i].cpu()


                    if epoch in inspection:
                        # move back to CPU
                        # tissue_emb = tissue_emb.cpu()
                        # plasma_emb = plasma_emb.cpu()

                        # patient IDs
                        tissue_ids = tissue.patient_id
                        plasma_ids = plasma.patient_id

                        epoch_name = f"epoch{epoch}"

                        for i, patient_id in enumerate(tissue_ids):
                            fold_embeddings[epoch_name]["tissue"][patient_id] = tissue_emb[i]

                        for i, patient_id in enumerate(plasma_ids):
                            fold_embeddings[epoch_name]["plasma"][patient_id] = plasma_emb[i]

                        for i, patient_id in enumerate(tissue_ids):
                            fold_embeddings_proj[epoch_name]["tissue"]["shared"][patient_id] = \
                                tissue_proj_shared[i].cpu()

                            fold_embeddings_proj[epoch_name]["tissue"]["private"][patient_id] = \
                                tissue_proj[i].cpu()

                        for i, patient_id in enumerate(plasma_ids):
                            fold_embeddings_proj[epoch_name]["plasma"]["shared"][patient_id] = \
                                plasma_proj_shared[i].cpu()

                            fold_embeddings_proj[epoch_name]["plasma"]["private"][patient_id] = \
                                plasma_proj[i].cpu()




            epoch_val_loss = val_loss_Total / val_batches

            val_losses.append(epoch_val_loss)

            epoch_val_recall_5 = np.mean(
                np.array(all_ranks_val) <= 5
            )
            median_rank = np.median(all_ranks_val)

            epoch_val_loss_shared = val_loss_Total_shared / val_batches
            epoch_val_loss_diff = val_loss_Total_diff / val_batches
            epoch_val_loss_recon = val_loss_Total_recon / val_batches
            val_losses_shared.append(epoch_val_loss_shared)
            val_losses_diff.append(lamda_orth*epoch_val_loss_diff)
            val_losses_recon.append(lamda_recon*epoch_val_loss_recon)

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
                f"Val Loss: {epoch_val_loss:.4f} | Val Loss Shared: {epoch_val_loss_shared:.4f} | Val Loss Orth: {lamda_orth*epoch_val_loss_diff:.4f} | Val Loss Recon: {lamda_recon*epoch_val_loss_recon:.4f} | retrieval accuracy embedding: {epoch_val_retrieval_accuracy:.4f} |  retrieval accuracy proj: {epoch_val_retrieval_accuracy_proj:.4f} | rank: {median_rank} | recall@5: {epoch_val_recall_5:.4f}"
            )




            epochs_range = range(1, epochs + 1)

        # Loss plot
        plt.figure(figsize=(6, 4))

        plt.plot(
            epochs_range,
            train_losses_shared,
            label="Contrastive Train loss"
        )

        plt.plot(
            epochs_range,
            val_losses_shared,
            label="Contrastive Validation loss"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"Fold {fold + 1} Loss")

        plt.tight_layout()

        plt.savefig(
            f"training_plots/orth_recon/fold_{fold + 1}_shared_loss.png",
            dpi=300
        )

        plt.close()

        # Loss plot
        plt.figure(figsize=(6, 4))


        plt.plot(
            epochs_range,
            train_losses_orth,
            label="Lamda * Orthogonality Train loss")


        plt.plot(
            epochs_range,
            val_losses_diff,
            label="Lamda * Orthogonality Validation loss"
        )


        plt.xlabel("Epoch")
        plt.ylabel("Orthogonal Loss")
        plt.legend()
        plt.title(f"Fold {fold + 1} Loss")

        plt.tight_layout()

        plt.savefig(
            f"training_plots/orth_recon/fold_{fold + 1}_orth_loss.png",
            dpi=300
        )

        plt.close()

        # Loss plot
        plt.figure(figsize=(6, 4))


        plt.plot(
            epochs_range,
            train_losses_recon,
            label="Lamda * Plasma Reconstruction Train loss"
        )

        plt.plot(
            epochs_range,
            val_losses_recon,
            label="Lamda * Plasma Reconstruction Validation loss"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Reconstruction Loss")
        plt.legend()
        plt.title(f"Fold {fold + 1} Loss")

        plt.tight_layout()

        plt.savefig(
            f"training_plots/orth_recon/fold_{fold + 1}_recon_loss.png",
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
            f"training_plots/orth_recon/fold_{fold + 1}_similarity_proj.png",
            dpi=300
        )

        plt.close()

        # Similarity plot proj
        plt.figure(figsize=(6, 4))

        plt.plot(
            epochs_range,
            retrieval_accuracy_proj_train,
            label="Recall@1 train"
        )
        plt.plot(
            epochs_range,
            train_recall,
            label="Recall@5 train"
        )


        plt.plot(
            epochs_range,
            retrieval_accuracy_proj_val,
            label="Recall@1 val"
        )
        plt.plot(
            epochs_range,
            val_recall,
            label="Recall@5 Val"
        )


        plt.xlabel("Epoch")
        plt.ylabel("Retrieval accuracy")
        plt.legend()
        plt.title(f"Fold {fold + 1} Retrieval Accuracy")

        plt.tight_layout()

        plt.savefig(
            f"training_plots/orth_recon/fold_{fold + 1}_retrieval_acc.png",
            dpi=300
        )

        plt.close()

        # Median rank plot proj
        plt.figure(figsize=(6, 4))

        plt.plot(
            epochs_range,
            train_rank,
            label="Median rank train"
        )

        plt.plot(
            epochs_range,
            val_rank,
            label="Median rank val"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Median rank")
        plt.legend()
        plt.title(f"Fold {fold + 1} Median Rank Similarity")

        plt.tight_layout()

        plt.savefig(
            f"training_plots/orth_recon/fold_{fold + 1}_median_rank.png",
            dpi=300
        )

        plt.close()


        torch.save(  fold_embeddings,f"embeddings_orth_recon/fold_{fold}_embeddings.pt" )
        torch.save(fold_embeddings_proj, f"embeddings_orth_recon/fold_{fold}_embeddings_proj.pt")
        torch.save(best_embeddings_proj, f"embeddings_orth_recon/fold_{fold}_best_embeddings_proj.pt")


training_loop()
