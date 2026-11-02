import copy

import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import Dataset

from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GlobalAttention

from torch_geometric.loader import DataLoader
import numpy as np
import json

with open("pretraining_fold_splits.json", "r") as f:
    fold_splits = json.load(f)

gene_to_idx_plasma = pd.read_csv('PR80_30/gene_to_idx_400_plasma.csv')
gene_to_idx_tissue = pd.read_csv('PR80_30/gene_to_idx_400_tissue.csv')


graphs_plasma = torch.load("PR80_30/proteins_graphs_400_plasma.pt", weights_only=False)

plasma_dict = {
    g.patient_id:g
    for g in graphs_plasma
}


#graphs_tissue = torch.load("PR80_30/proteins_graphs_400_tissue.pt", weights_only=False)

##Load tissue pathway scores and select targets
tissue_pathway_scores = pd.read_csv('PR80_30/tissue_pathway_scores.csv')
target_pathway_names = ['REACTOME_EXTRACELLULAR_MATRIX_ORGANIZATION', 'REACTOME_DEGRADATION_OF_THE_EXTRACELLULAR_MATRIX', 'REACTOME_INTEGRIN_CELL_SURFACE_INTERACTIONS',
                   'REACTOME_NON_INTEGRIN_MEMBRANE_ECM_INTERACTIONS']
tissue_pathway_scores = tissue_pathway_scores.set_index('Name')
target_pathway_scores = tissue_pathway_scores[target_pathway_names]

patients = np.array(sorted(
    set(plasma_dict.keys()) &
    set(target_pathway_scores.index)
))
print("Number of patients:", len(patients))
print("Unique patients:", len(set(patients)))

print(
    "Missing target values:",
    target_pathway_scores.loc[
        patients, target_pathway_names
    ].isna().sum()
)
print(target_pathway_scores.head())
print(target_pathway_scores.shape)

dataframe = []
print(len(patients))
for patient in patients:

    graph = plasma_dict[patient]

    target = tissue_pathway_scores.loc[patient, target_pathway_names].values
    dataframe.append({
        'patient': patient,
        'plasma': graph,
        'target': target,
    })

print(dataframe)

# for pathway in target_pathway_scores:
#     plt.hist(target_pathway_scores[pathway])
#     plt.title(pathway)
#     plt.xlabel('Pathway score')
#     plt.ylabel('Frequency')
#     plt.show()



class GetData(Dataset):
    def __init__(
        self,
        patients,
        plasma_graphs,
        pathways
    ):

        self.patients = list(patients)
        self.plasma_graphs = plasma_graphs
        self.pathways = pathways


    def __len__(self):

        return len(self.patients)


    def __getitem__(self, idx):

        patient = self.patients[idx]

        plasma = self.plasma_graphs[patient]

        target = self.pathways.loc[
            patient, target_pathway_names
        ].to_numpy(dtype=np.float32)

        return plasma, patient, torch.tensor(target)



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


class Tissue_Pathway_Predictor(nn.Module):
    def __init__(self,
                 encoder):
        super().__init__()

        self.encoder = encoder
        # self.projector = projector

        # Freeze pretrained modules
        self.encoder.requires_grad_(False)
        self.encoder.conv2.requires_grad_(True)

        # self.projector.requires_grad_(False)

        self.predictor = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 4)
        )

    def forward(self,plasma):
        # with torch.no_grad():
        x = self.encoder(plasma.x, plasma.edge_index,plasma.batch)
        # z = self.projector(x)
        return self.predictor(x)

    def train(self, mode=True):
        # Prevent parent train() calls from enabling training mode
        # in the frozen pretrained modules.
        super().train(mode)
        # Keep frozen layers in evaluation mode
        self.encoder.conv1.eval()
        self.encoder.pool.eval()

        # Allow conv2 to follow training mode
        self.encoder.conv2.train(mode)
        # self.encoder.eval()
        # self.projector.eval()
        return self

def training_loop():
    device = "cuda" if torch.cuda.is_available() else "cpu"


    # Collect predictions once per patient, out of fold
    oof_predictions = {}
    oof_actuals = {}
    fold_metrics = []
    history = []

    for fold in range(5):
        split = fold_splits[str(fold)]

        print(f"\nFold {fold}/5")

        train_losses = []
        val_losses = []
        train_patients = [
            p for p in split["train_patients"]
            if p in patients
        ]

        val_patients = [
            p for p in split["val_patients"]
            if p in patients
        ]
        print(
            f"Fold {fold}: train={len(train_patients)}, "
            f"val={len(val_patients)}"
        )
        #### Scale target

        # from sklearn.preprocessing import StandardScaler
        #
        # # Extract targets for this fold, preserving patient ID strings
        # y_train = tissue_pathway_scores.loc[
        #     train_patients, target_pathway_names
        # ].to_numpy(dtype=np.float32)
        #
        # y_val = tissue_pathway_scores.loc[
        #     val_patients, target_pathway_names
        # ].to_numpy(dtype=np.float32)
        #
        # # Fit only on training targets
        # y_scaler = StandardScaler()
        # y_train_scaled = y_scaler.fit_transform(y_train)
        # y_val_scaled = y_scaler.transform(y_val)
        #
        # # Create a fold-specific mapping of scaled targets
        # scaled_pathway_scores = pd.DataFrame(
        #     np.concatenate([y_train_scaled, y_val_scaled], axis=0),
        #     index=train_patients + val_patients,
        #     columns=target_pathway_names
        # )

        # print("Train unique:", len(set(train_patients)))
        # print("Val unique:", len(set(val_patients)))
        # print("Overlap:", len(set(train_patients) & set(val_patients)))

        train_dataset = GetData(
            train_patients,
            plasma_dict,
            target_pathway_scores
        )

        val_dataset = GetData(
            val_patients,
            plasma_dict,
           target_pathway_scores
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=True,
            drop_last=False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=False,
            drop_last=False
        )
        plasma_encoder = GraphEncoder_Plasma()
        plasma_encoder.load_state_dict(
            torch.load(
                f'best_plasma_encoder_{fold}.pt',
                map_location=device,
                weights_only=True
            )
        )

        # projector = ProjectionHead_Shared()
        # projector.load_state_dict(
        #     torch.load(
        #         f'best_shared_projector_{fold}.pt',
        #         map_location=device,
        #         weights_only=True
        #     )
        # )

        predictor = Tissue_Pathway_Predictor(plasma_encoder).to(device)

        criterion = nn.MSELoss()

        # optimizer = torch.optim.Adam(predictor.predictor.parameters(), lr=1e-4)
        optimizer = torch.optim.AdamW([
            {
                "params": predictor.encoder.conv2.parameters(),
                "lr": 1e-5
            },
            {
                "params": predictor.predictor.parameters(),
                "lr": 1e-4
            }
        ], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3
        )

        epochs = 100
        best_val_loss = float("inf")
        best_val_preds = None
        best_val_targets = None
        best_val_ids = None

        for epoch in range(epochs):
            train_batch_losses = []
            predictor.train()
            for plasma, patient, target in train_loader:
                optimizer.zero_grad()

                pred = predictor(plasma)

                pred = pred.reshape_as(target)

                loss = criterion(pred, target)
                loss.backward()

                optimizer.step()
                train_batch_losses.append(loss.item())

            epoch_train_loss = float(np.mean(train_batch_losses))
            train_losses.append(epoch_train_loss)


            ##### Validation

            predictor.eval()
            val_batch_losses = []
            val_preds = []
            val_targets = []
            val_ids = []
            with torch.no_grad():
                for plasma, patient_ids, target in val_loader:
                    pred = predictor(plasma).reshape_as(target)

                    val_batch_losses.append(
                        criterion(pred, target).item()
                    )

                    val_preds.append(pred.cpu().numpy())
                    val_targets.append(target.cpu().numpy())
                    val_ids.extend(list(patient_ids))

                epoch_val_loss = float(np.mean(val_batch_losses))

                scheduler.step(epoch_val_loss)

            # Save best model using validation loss
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss

                # Save validation predictions from this best epoch
                best_val_preds = np.concatenate(val_preds, axis=0)
                best_val_targets = np.concatenate(val_targets, axis=0)
                best_val_ids = list(val_ids)

            history.append({
                "fold": fold,
                "epoch": epoch + 1,
                "train_loss": epoch_train_loss,
                "val_loss": epoch_val_loss,
                "lr": optimizer.param_groups[0]["lr"]
            })

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1:03d}/{epochs} | "
                    f"Train MSE: {epoch_train_loss:.4f} | "
                    f"Val MSE: {epoch_val_loss:.4f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                )
        print(f"Best validation MSE: {best_val_loss:.4f}")
        #
        # best_val_preds_original = y_scaler.inverse_transform(
        #     best_val_preds
        # )
        #
        # best_val_targets_original = y_scaler.inverse_transform(
        #     best_val_targets
        # )
        # Store this fold's out-of-fold predictions
        for patient, actual, prediction in zip(
                best_val_ids, best_val_targets, best_val_preds
        ):
            key = str(patient)
            oof_actuals[key] = actual
            oof_predictions[key] = prediction

        # Fold-level metrics, per pathway
        for j, pathway in enumerate(target_pathway_names):
            fold_metrics.append({
                "fold": fold,
                "pathway": pathway,
                "MAE": mean_absolute_error(
                    best_val_targets[:, j], best_val_preds[:, j]
                ),
                "RMSE": np.sqrt(mean_squared_error(
                    best_val_targets[:, j], best_val_preds[:, j]
                )),
                "R2": r2_score(best_val_targets[:, j], best_val_preds[:, j]),
                "MSE": mean_squared_error(
                    best_val_targets[:, j], best_val_preds[:, j]
                )
            })
        print(fold_metrics)



        # ---------------- Overall out-of-fold metrics ----------------
    metrics = []

    for j, pathway in enumerate(target_pathway_names):

        actual = np.array([
            oof_actuals[str(p)][j] for p in patients
        ])
        predicted = np.array([
            oof_predictions[str(p)][j] for p in patients
        ])

        metrics.append({
            "pathway": pathway,
            "MAE": mean_absolute_error(actual, predicted),
            "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
            "MSE": mean_squared_error(actual, predicted),
            "R2": r2_score(actual, predicted)
        })

    metrics_df = pd.DataFrame(metrics)
    fold_metrics_df = pd.DataFrame(fold_metrics)
    history_df = pd.DataFrame(history)

    predictions_df = pd.DataFrame({
        "patient": patients
    })
    for j, pathway in enumerate(target_pathway_names):
        predictions_df[f"{pathway}_actual"] = [
            oof_actuals[str(p)][j] for p in patients
        ]
        predictions_df[f"{pathway}_predicted"] = [
            oof_predictions[str(p)][j] for p in patients
        ]
    print("\nOverall out-of-fold metrics:")
    print(metrics_df.to_string(index=False))

    print("\nMetrics by fold:")
    print(fold_metrics_df.to_string(index=False))

    # ---------------- Prediction vs actual plots ----------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for j, pathway in enumerate(target_pathway_names):
        actual = np.array([
            oof_actuals[str(p)][j] for p in patients
        ])
        predicted = np.array([
            oof_predictions[str(p)][j] for p in patients
        ])
        axes[j].scatter(actual, predicted, alpha=0.8)

        low = min(actual.min(), predicted.min())
        high = max(actual.max(), predicted.max())

        axes[j].plot(
            [low, high], [low, high],
            linestyle="--"
        )

        axes[j].set_title(
            pathway.replace("REACTOME_", "").replace("_", " ")
        )
        axes[j].set_xlabel("Actual tissue pathway score")
        axes[j].set_ylabel("Predicted tissue pathway score")

        axes[j].text(
            0.05, 0.95,
            f"MAE = {metrics_df.loc[j, 'MAE']:.3f}\n"
            f"RMSE = {metrics_df.loc[j, 'RMSE']:.3f}\n"
            f"R² = {metrics_df.loc[j, 'R2']:.3f}",
            transform=axes[j].transAxes,
            va="top"
        )

    plt.tight_layout()
    plt.show()

    for j, pathway in enumerate(target_pathway_names):
        actual = np.array([
            oof_actuals[str(p)][j] for p in patients
        ])
        predicted = np.array([
            oof_predictions[str(p)][j] for p in patients
        ])

        print(f"\n{pathway}")
        print(f"Actual mean: {actual.mean():.4f}")
        print(f"Actual SD:   {actual.std(ddof=1):.4f}")
        print(f"Actual range: {actual.min():.4f} to {actual.max():.4f}")
        print(f"Prediction SD: {predicted.std(ddof=1):.4f}")
        print(f"MAE:         {mean_absolute_error(actual, predicted):.4f}")
        print(f"R2:          {r2_score(actual, predicted):.4f}")

    return metrics_df, fold_metrics_df, predictions_df, history_df


metrics_df, fold_metrics_df, predictions_df, history_df = training_loop()

