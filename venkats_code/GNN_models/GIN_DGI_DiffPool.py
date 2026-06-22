import torch
from torch_geometric.utils import to_undirected
import os

os.chdir('/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/Proteomics/')

graphs_dir = './PlasmaAAA/PreprocessingScripts/CohortSpecificImpute/PatientGraphs/Static/'

patient_graphs = {}

for file_name in os.listdir(graphs_dir):
    if file_name.endswith('_graph.pt'):
        # Extract patient ID and convert to int
        patient_id_str = file_name.replace('_graph.pt', '')
        try:
            patient_id = int(patient_id_str)
        except ValueError:
            print(f"Could not convert '{patient_id_str}' to int. Skipping.")
            continue

        # Load the graph
        graph = torch.load(os.path.join(graphs_dir, file_name), weights_only=False)

        if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
            # If edge attributes are present, use the to_undirected function with edge_attr
            graph.edge_index, graph.edge_attr = to_undirected(
                graph.edge_index, edge_attr=graph.edge_attr, num_nodes=graph.num_nodes, reduce='mean'
            )
        else:
            # If no edge attributes are present (which won't happen here), just make the graph undirected
            graph.edge_index = to_undirected(graph.edge_index, num_nodes=graph.num_nodes)

        # Store the graph in the dictionary
        patient_graphs[patient_id] = graph

        # Optionally, print some information about the graph
        print(f"Loaded graph for patient ID: {patient_id}")
        print(f"Edge index shape: {graph.edge_index.shape}")
        print(f"Edge attributes shape: {graph.edge_attr.shape}")

###############################################################################################################################################
#bad_sample_ids = [52, 48, 44, 42]

#bad_sample_ids = [52, 48, 44, 42]

#filtered_graphs = {
#    pid: g for pid, g in patient_graphs.items()
#    if pid not in bad_sample_ids
#}

###############################################################################################################################################
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.dense.diff_pool import dense_diff_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn import GINEConv
from torch_geometric.nn import BatchNorm
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

class Discriminator(nn.Module):
    def __init__(self, feat_dim, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.hidden_layer = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.Dropout(0.3),
        )
        self.activation = nn.ReLU()
        self.f_k = nn.Bilinear(feat_dim, feat_dim, 1)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Bilinear)):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, c, h):
        h = self.activation(self.hidden_layer(h))
        return torch.sigmoid(self.f_k(h, c) / self.temperature).squeeze(1)

class DiffPoolReadout(nn.Module):
    def __init__(self, feat_dim, num_nodes=100):
        super().__init__()
        self.pool_gnn = GATv2Conv(feat_dim, num_nodes, heads=1)

    def forward(self, x, edge_index, batch):
        s = F.softmax(self.pool_gnn(x, edge_index), dim=-1)
        adj = to_dense_adj(edge_index, batch=batch)
        x_dense = to_dense_batch(x, batch)[0]
        s_dense = to_dense_batch(s, batch)[0]
        x_pool, adj_pool, link_loss, ent_loss = dense_diff_pool(x_dense, adj, s_dense)
        return x_pool.mean(dim=1), link_loss + ent_loss

class GINEConvEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=1):
        super().__init__()
        self.dropout = 0.5
        
        self.gine1 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            ),
            edge_dim=edge_dim
        )
        self.bn1 = BatchNorm(hidden_channels)
        
        self.gine2 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            ),
            edge_dim=edge_dim
        )
        self.bn2 = BatchNorm(hidden_channels)
        
        self.gine3 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(hidden_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels)
            ),
            edge_dim=edge_dim
        )
        self.bn3 = BatchNorm(out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.elu(self.gine1(x, edge_index, edge_attr))
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = F.elu(self.gine2(x, edge_index, edge_attr))
        x = self.bn2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.gine3(x, edge_index, edge_attr)
        x = self.bn3(x)
        return x


class GINE_DGI(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=1):
        super().__init__()
#
        self.encoder = GINEConvEncoder(in_channels, hidden_channels, out_channels, edge_dim)
        self.readout = DiffPoolReadout(out_channels)
        self.discriminator = Discriminator(out_channels)
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, edge_index, edge_attr=None, corrupt_x=None):
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        
        x, edge_index, edge_attr = x.to(device), edge_index.to(device), edge_attr.to(device) if edge_attr is not None else None
        node_emb = self.encoder(x, edge_index, edge_attr)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        graph_emb, aux_loss = self.readout(node_emb, edge_index, batch=batch)

        if corrupt_x is not None:
            corrupt_x = corrupt_x.to(device)
            corrupt_emb = self.encoder(corrupt_x, edge_index, edge_attr)
            return node_emb, graph_emb, corrupt_emb, aux_loss

        return node_emb, graph_emb, aux_loss

def dgi_loss(logits):
    pos_logits = logits[:, 0]
    neg_logits = logits[:, 1]
    pos_loss = F.binary_cross_entropy(pos_logits, torch.ones_like(pos_logits))
    neg_loss = F.binary_cross_entropy(neg_logits, torch.zeros_like(neg_logits))
    return (pos_loss + neg_loss) / 2

def robust_graph_corruption(x, corruption_rate=0.5, noise_scale=0.15, feature_corruption_rate=0.3, shuffle_prob=0.3):
    corrupted = x.clone()
    device = x.device
    if feature_corruption_rate > 0:
        feat_mask = torch.rand_like(x) < feature_corruption_rate
        node_mask = torch.rand(x.size(0), device=device) < corruption_rate
        corrupted[node_mask] *= feat_mask[node_mask].float()
    if noise_scale > 0:
        feature_std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
        corrupted += noise_scale * feature_std * torch.randn_like(x)
    if shuffle_prob > 0 and torch.rand(1, device=device) < shuffle_prob:
        corrupted = corrupted[torch.randperm(x.size(0), device=device)]
    return corrupted

def train_model(patient_graphs, in_channels, hidden_channels, out_channels, epochs=100, lr=5e-4):
    model = GINE_DGI(in_channels, hidden_channels, out_channels).to(device)  # Removed num_heads
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    best_loss = float('inf')
    patience = 20
    cnt_wait = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for graph in patient_graphs.values():
            graph = graph.to(device)
            x, edge_index, edge_attr = graph.x.to(device), graph.edge_index.to(device), graph.edge_attr.to(device) if graph.edge_attr is not None else None
            if edge_attr is not None and edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)

            corrupt_x = robust_graph_corruption(x)

            optimizer.zero_grad()
            node_emb, graph_emb, corrupt_emb, aux_loss = model(x, edge_index, edge_attr, corrupt_x)
            graph_emb_exp = graph_emb.unsqueeze(0).expand(node_emb.size(0), -1)
            sc_pos = model.discriminator(graph_emb_exp, node_emb)
            sc_neg = model.discriminator(graph_emb_exp, corrupt_emb)
            logits = torch.stack([sc_pos, sc_neg], dim=1)
            dgi = dgi_loss(logits)
            loss = dgi + 0.5 * aux_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(patient_graphs)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')
        if avg_loss < best_loss:
            best_loss = avg_loss
            cnt_wait = 0
            torch.save(model.state_dict(), 'best_model_gin_dgi_diffpool.pth')
        else:
            cnt_wait += 1
            if cnt_wait >= patience:
                print("Early stopping!")
                break
        scheduler.step()

    model.load_state_dict(torch.load('/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/Proteomics/PlasmaAAA/GNN_models/DGI_RESULTS/GIN/DiffPool/best_model_gin_dgi_diffpool.pth'))
    model.eval()
    embeddings = []
    with torch.no_grad():
        for graph in patient_graphs.values():
            edge_attr = graph.edge_attr
            if edge_attr is not None and edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            _, graph_emb, _ = model(graph.x, graph.edge_index, edge_attr)
            embeddings.append(graph_emb.cpu().numpy())
    embeddings = [F.normalize(torch.tensor(emb), p=2, dim=0).numpy() for emb in embeddings]
    return list(patient_graphs.keys()), embeddings

def run_multiple_dgi_trials(patient_graphs, in_channels, hidden_channels, out_channels, num_heads, num_runs=10):
    patient_ids = list(patient_graphs.keys())
    all_embeddings = {pid: [] for pid in patient_ids}
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        _, embeddings = train_model(patient_graphs, in_channels, hidden_channels, out_channels, num_heads)
        for pid, emb in zip(patient_ids, embeddings):
            all_embeddings[pid].append(emb)
    avg_embeddings = {}
    std_embeddings = {}
    for pid in patient_ids:
        embs = np.stack(all_embeddings[pid])
        avg_embeddings[pid] = np.mean(embs, axis=0)
        std_embeddings[pid] = np.std(embs, axis=0)
    return patient_ids, avg_embeddings, std_embeddings

in_channels = patient_graphs[list(patient_graphs.keys())[0]].x.shape[1]
hidden_channels = 128
out_channels = 128
num_heads = 8
######################################################################################################################################################

patient_ids, avg_embs, std_embs = run_multiple_dgi_trials(
    patient_graphs, 
    in_channels, 
    hidden_channels, 
    out_channels, 
    num_heads, 
    num_runs=10
)
######################################################################################################################################################

import os
import numpy as np

def save_embeddings_as_dict_and_arrays(patient_ids, avg_embeddings, std_embeddings, 
                                        dict_file='embeddings_dict.npy', 
                                        array_file='embeddings_array.npy', 
                                        save_dir='./'):
    embeddings_dict = {
        'patient_ids': patient_ids, 
        'avg_embeddings': avg_embeddings,  
        'std_embeddings': std_embeddings  
    }

    np.save(f'{save_dir}/{dict_file}', embeddings_dict)
    print(f"Embeddings saved as dictionaries in {dict_file}")

    avg_array = np.stack([avg_embeddings[pid] for pid in patient_ids])  
    std_array = np.stack([std_embeddings[pid] for pid in patient_ids])  

    embeddings_array = {
        'patient_ids': patient_ids,  
        'avg_embeddings': avg_array,  
        'std_embeddings': std_array  
    }

    np.save(f'{save_dir}/{array_file}', embeddings_array)
    print(f"Embeddings saved as arrays in {array_file}")


save_dir = '/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/Proteomics/PlasmaAAA/GNN_models/DGI_RESULTS/GIN/DiffPool/'

save_embeddings_as_dict_and_arrays(patient_ids, avg_embs, std_embs, 
                                   dict_file='embeddings_dict.npy', 
                                   array_file='embeddings_array.npy', 
                                   save_dir=save_dir)
