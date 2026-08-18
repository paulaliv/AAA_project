import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import KFold
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

tissue_data = pd.read_excel('PR80_30/tissue_data_imputed_30.xlsx')
plasma_data = pd.read_excel('PR80_30/plasma_data_imputed_30.xlsx')

plasma_expression = plasma_data.set_index("Genes")
plasma_z = (
    plasma_expression
    .sub(plasma_expression.mean(axis=1), axis=0)
    .div(plasma_expression.std(axis=1), axis=0)
)

tissue_expression = tissue_data.set_index("Genes")
tissue_z = (
    tissue_expression
    .sub(tissue_expression.mean(axis=1), axis=0)
    .div(tissue_expression.std(axis=1), axis=0)
)
print(f'Plasma shape: {plasma_data.shape}')
print(f'Tissue shape: {tissue_data.shape}')

print(f'Plasma shape: {plasma_z.shape}')
print(f'Tissue shape: {tissue_z.shape}')
patients = plasma_data.columns[1:]
print(patients)
tissue_dict = {
    patient: torch.tensor(
        tissue_z[patient].values,
        dtype=torch.float32
    )
    for patient in patients
}

plasma_dict = {
    patient: torch.tensor(
        plasma_z[patient].values,
        dtype=torch.float32
    )
    for patient in patients
}

class CreatePairs(Dataset):
    def __init__(
        self,
        patients,
        tissue_data,
        plasma_data
    ):

        self.patients = patients
        self.tissue_data = tissue_data
        self.plasma_data = plasma_data


    def __len__(self):

        return len(self.patients)


    def __getitem__(self, idx):

        patient = self.patients[idx]

        tissue = self.tissue_data[patient]

        plasma = self.plasma_data[patient]

        return tissue, plasma, patient

class LinearEncoder(nn.Module):
    def __init__(self, input_dim=1400, hidden_dim=64):
        super().__init__()
        self.net = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return self.net(x)
class MLPEncoder(nn.Module):
    def __init__(self, input_dim=1400, hidden_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )

    def forward(self, x):
        return self.net(x)



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
        epochs = 20
        fold_embeddings = {
            "epoch0": {"tissue": {}, "plasma": {}},
           f"epoch{epochs-1}": {"tissue": {}, "plasma": {}}
        }
        fold_embeddings_proj = {
            "epoch0": {"tissue": {}, "plasma": {}},
            f"epoch{epochs-1}": {"tissue": {}, "plasma": {}}
        }

        print(
            f"Fold {fold + 1}"
        )


        train_patients = patients[train_idx]
        val_patients = patients[val_idx]

        train_dataset = CreatePairs(train_patients, tissue_dict, plasma_dict)
        val_dataset = CreatePairs(val_patients, tissue_dict, plasma_dict)

        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            drop_last=True,

        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            drop_last=False,

        )
        encoder_tissue = MLPEncoder(
            input_dim=6630
        ).to(device)

        encoder_plasma = MLPEncoder(input_dim=1531).to(device)

        projector_shared = ProjectionHead_Shared(
        ).to(device)


        optimizer = torch.optim.AdamW(
            list(encoder_tissue.parameters())
            +
            list(encoder_plasma.parameters())
            +
            list(projector_shared.parameters()),
            lr=3e-4,
            weight_decay=1e-5
        )




        for epoch in range(epochs):

            encoder_tissue.train()
            encoder_plasma.train()
            projector_shared.train()

            total_loss = 0
            retrieval_accuracies =[]
            retrieval_accuracies_proj =[]

            for tissue, plasma, patient in train_loader:
                tissue = tissue.to(device)
                plasma = plasma.to(device)

                print(
                    "Tissue patient-to-patient std:",
                    tissue.std(dim=0).mean().item()
                )

                print(
                    "Plasma patient-to-patient std:",
                    plasma.std(dim=0).mean().item()
                )
                tissue_raw_norm = F.normalize(tissue, dim=1)
                plasma_raw_norm = F.normalize(plasma, dim=1)

                tissue_raw_sim = tissue_raw_norm @ tissue_raw_norm.T
                plasma_raw_sim = plasma_raw_norm @ plasma_raw_norm.T
                mask_tissue= ~torch.eye(
                    tissue_raw_sim.size(0),
                    dtype=torch.bool,
                    device=tissue_raw_sim.device
                )
                mask_plasma = ~torch.eye(
                    plasma_raw_sim.size(0),
                    dtype=torch.bool,
                    device=plasma_raw_sim.device
                )


                print(
                    "RAW tissue off:",
                    tissue_raw_sim[mask_tissue].mean().item()
                )

                print(
                    "RAW plasma off:",
                    plasma_raw_sim[mask_plasma].mean().item()
                )



                tissue_emb = encoder_tissue(
                    tissue
                )
                # x = encoder_tissue.net[0](tissue)
                #
                # print("After Linear 1:", x.std(dim=0).mean().item())
                #
                # x = encoder_tissue.net[1](x)
                #
                # print("After ReLU:", x.std(dim=0).mean().item())
                #
                # x = encoder_tissue.net[2](x)
                #
                # print("After Linear 2:", x.std(dim=0).mean().item())

                plasma_emb = encoder_plasma(
                    plasma
                )

                print(
                    "Tissue embedding std across patients:",
                    tissue_emb.std(dim=0).mean().item()
                )

                print(
                    "Plasma embedding std across patients:",
                    plasma_emb.std(dim=0).mean().item()
                )


                tissue_emb_1 = F.normalize(tissue_emb, dim=1)
                plasma_emb_1 = F.normalize(plasma_emb, dim=1)

                # tissue_sim = tissue_emb_1 @ tissue_emb_1.T
                # plasma_sim = plasma_emb_1 @ plasma_emb_1.T
                cross_sim = tissue_emb_1 @ plasma_emb_1.T
                predicted = cross_sim.argmax(dim=1)
                true = torch.arange(
                    cross_sim.size(0)
                )

                retrieval_accuracy = (
                        predicted == true
                ).float().mean()
                retrieval_accuracies.append(retrieval_accuracy.item())

                mask = ~torch.eye(
                    cross_sim.size(0),
                    dtype=torch.bool,
                    device=cross_sim.device
                )


                print(
                    f'Mean similarity embedding space: positive similarity: {cross_sim.diag().mean().item():.4f} | negative similarity: {cross_sim[mask].mean().item():.4f} | Difference: {cross_sim.diag().mean().item() - cross_sim[mask].mean().item():.4f}'
                )

                tissue_proj_shared = projector_shared(
                    tissue_emb
                )

                plasma_proj_shared = projector_shared(
                    plasma_emb
                )

                tissue_proj_norm = F.normalize(tissue_proj_shared, dim=1)
                plasma_proj_norm = F.normalize(plasma_proj_shared, dim=1)
                cross_sim_proj = tissue_proj_norm @ plasma_proj_norm.T

                print(
                    f'Mean similarity projection space: positive similarity: {cross_sim_proj.diag().mean().item():.4f} | negative similarity: {cross_sim_proj[mask].mean().item():.4f} | Difference: {cross_sim_proj.diag().mean().item() - cross_sim_proj[mask].mean().item():.4f}'
                )

                predicted = cross_sim_proj.argmax(dim=1)
                true_proj = torch.arange(
                    cross_sim_proj.size(0)
                )

                retrieval_accuracy_proj = (
                        predicted == true_proj
                ).float().mean()
                retrieval_accuracies_proj.append(retrieval_accuracy_proj.item())




                loss_shared = symmetric_contrastive_loss(
                    tissue_proj_shared,
                    plasma_proj_shared
                )

                #old_weight = encoder_tissue.net[0].weight.detach().clone()

                optimizer.zero_grad()

                loss_shared.backward()

                optimizer.step()
                # weight_change = (
                #         encoder_tissue.net[0].weight.detach() - old_weight
                # ).abs().mean()

                #print("Weight change:", weight_change.item())

                total_loss += loss_shared.item()

            epoch_train_loss = total_loss / len(train_loader)
            train_losses.append(epoch_train_loss)


            print(
                f"Epoch {epoch}: "
                f"Train Loss: {epoch_train_loss:.4f} | retrieval accuracy embedding: {np.mean(retrieval_accuracies)} | | retrieval accuracy proj: {np.mean(retrieval_accuracies)}"
            )

            val_loss_total = 0
            val_batches = 0
            with torch.no_grad():
                encoder_tissue.eval()
                encoder_plasma.eval()
                projector_shared.eval()
                for tissue, plasma, patient in val_loader:

                    tissue = tissue.to(device)
                    plasma = plasma.to(device)

                    # graph encoder

                    tissue_emb = encoder_tissue(
                        tissue
                    )

                    plasma_emb = encoder_plasma(
                        plasma
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
                        f'Mean similarity embedding space: positive similarity: {mean_positive_similarity:.4f} | negative similarity: {mean_negative_similarity:.4f} | Difference: {mean_negative_similarity - mean_positive_similarity:.4f}'
                    )

                    predicted = similarity_matrix.argmax(dim=1)
                    true = torch.arange(
                        similarity_matrix.size(0)
                    )

                    retrieval_accuracy = (
                            predicted == true
                    ).float().mean()

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
                        f'Mean similarity projection space: positive similarity: {mean_positive_similarity_proj:.4f} | negative similarity: {mean_negative_similarity_proj:.4f} | Difference: {mean_positive_similarity_proj - mean_negative_similarity_proj:.4f}'
                    )

                    predicted_proj = similarity_matrix_proj.argmax(dim=1)
                    true_proj= torch.arange(
                        similarity_matrix_proj.size(0)
                    )

                    retrieval_accuracy_proj = (
                            predicted_proj == true_proj
                    ).float().mean()




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
                        ids = patient


                        epoch_name = f"epoch{epoch}"

                        for i, patient_id in enumerate(ids):
                            fold_embeddings[epoch_name]["tissue"][patient_id] = tissue_emb[i]

                        for i, patient_id in enumerate(ids):
                            fold_embeddings[epoch_name]["plasma"][patient_id] = plasma_emb[i]

                        for i, patient_id in enumerate(ids):
                            fold_embeddings_proj[epoch_name]["tissue"][patient_id] = tissue_proj[i]

                        for i, patient_id in enumerate(ids):
                            fold_embeddings_proj[epoch_name]["plasma"][patient_id] = plasma_proj[i]



            epoch_val_loss = val_loss_total / val_batches
            val_losses.append(epoch_val_loss)
            print(
                f"Val Loss: {epoch_val_loss:.4f} | retrieval accuracy embedding: {retrieval_accuracy} | retrieval accuracy proj: {retrieval_accuracy_proj}"
                )

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
            f"training_plots/baseline/fold_{fold + 1}_loss.png",
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
            f"training_plots/baseline/fold_{fold + 1}_similarity_emb.png",
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
            f"training_plots/baseline/fold_{fold + 1}_similarity_proj.png",
            dpi=300
        )

        plt.close()

        torch.save(  fold_embeddings,f"embeddings_baseline/fold_{fold}_embeddings.pt" )
        torch.save(fold_embeddings_proj, f"embeddings_baseline/fold_{fold}_embeddings_proj.pt")

training_loop()

