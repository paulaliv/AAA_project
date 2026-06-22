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
bad_sample_ids = [46, 52, 57, 90, 31, 67, 71, 35, 68, 50, 95]

filtered_graphs = {
    pid: g for pid, g in patient_graphs.items()
    if pid not in bad_sample_ids
}
#

###############################################################################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TransformerConv
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphDataset(Dataset):
    """Custom dataset for efficient batching"""
    def __init__(self, graph_dict):
        self.graphs = list(graph_dict.values())
        self.pids = list(graph_dict.keys())
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.pids[idx]

def custom_collate_fn(batch):
    """Custom collate function for PyTorch Geometric Data objects"""
    graphs, pids = zip(*batch)
    return list(graphs), list(pids)

class GraphCLAugmentation:
    """Graph augmentation strategies for GraphCL"""
    
    @staticmethod
    def node_dropping(x, edge_index, edge_attr=None, drop_ratio=0.1):
        """Randomly drop nodes"""
        num_nodes = x.size(0)
        drop_num = int(num_nodes * drop_ratio)
        
        if drop_num == 0 or num_nodes <= 2:
            return x, edge_index, edge_attr
        
        # Randomly select nodes to keep
        keep_indices = torch.randperm(num_nodes)[:num_nodes - drop_num]
        keep_indices = torch.sort(keep_indices)[0]
        
        # Create node mapping
        node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(keep_indices)}
        
        # Filter edges
        mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
        new_edges = []
        new_edge_attrs = []
        
        for i in range(edge_index.size(1)):
            src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
            if src in node_map and tgt in node_map:
                new_edges.append([node_map[src], node_map[tgt]])
                if edge_attr is not None:
                    new_edge_attrs.append(edge_attr[i])
        
        new_x = x[keep_indices]
        
        if new_edges:
            new_edge_index = torch.tensor(new_edges, dtype=torch.long, device=x.device).t()
            new_edge_attr = torch.stack(new_edge_attrs) if new_edge_attrs else None
        else:
            new_edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
            new_edge_attr = None
            
        return new_x, new_edge_index, new_edge_attr
    
    @staticmethod
    def edge_perturbation(x, edge_index, edge_attr=None, drop_ratio=0.1, add_ratio=0.1):
        """Randomly drop and add edges"""
        num_edges = edge_index.size(1)
        num_nodes = x.size(0)
        
        # Edge dropping
        drop_num = int(num_edges * drop_ratio)
        if drop_num > 0 and num_edges > 1:
            keep_indices = torch.randperm(num_edges)[drop_num:]
            edge_index = edge_index[:, keep_indices]
            if edge_attr is not None:
                edge_attr = edge_attr[keep_indices]
        
        # Edge adding
        add_num = int(num_nodes * (num_nodes - 1) * add_ratio / 2)
        if add_num > 0:
            # Generate random edges
            new_edges = []
            for _ in range(add_num):
                src = torch.randint(0, num_nodes, (1,)).item()
                tgt = torch.randint(0, num_nodes, (1,)).item()
                if src != tgt:  # Avoid self-loops
                    new_edges.append([src, tgt])
            
            if new_edges:
                new_edge_tensor = torch.tensor(new_edges, dtype=torch.long, device=x.device).t()
                edge_index = torch.cat([edge_index, new_edge_tensor], dim=1)
                
                if edge_attr is not None:
                    # Add random edge attributes for new edges
                    new_edge_attrs = torch.randn(len(new_edges), edge_attr.size(1), device=x.device)
                    edge_attr = torch.cat([edge_attr, new_edge_attrs], dim=0)
        
        return x, edge_index, edge_attr
    
    @staticmethod
    def attribute_masking(x, edge_index, edge_attr=None, mask_ratio=0.1):
        """Randomly mask node attributes"""
        num_nodes, num_features = x.size()
        mask_num = int(num_nodes * num_features * mask_ratio)
        
        if mask_num == 0:
            return x, edge_index, edge_attr
        
        new_x = x.clone()
        
        # Random masking
        mask_indices = torch.randperm(num_nodes * num_features)[:mask_num]
        for idx in mask_indices:
            node_idx = idx // num_features
            feat_idx = idx % num_features
            new_x[node_idx, feat_idx] = 0  
        
        return new_x, edge_index, edge_attr
    
    @staticmethod
    def subgraph_sampling(x, edge_index, edge_attr=None, sample_ratio=0.8):
        """Sample a connected subgraph"""
        num_nodes = x.size(0)
        sample_size = max(2, int(num_nodes * sample_ratio))
        
        if sample_size >= num_nodes:
            return x, edge_index, edge_attr
        
        # Start from a random node and do BFS
        start_node = torch.randint(0, num_nodes, (1,)).item()
        visited = {start_node}
        queue = [start_node]
        
        adj_list = {i: [] for i in range(num_nodes)}
        for i in range(edge_index.size(1)):
            src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
            adj_list[src].append(tgt)
            adj_list[tgt].append(src)
        
        # BFS to get connected subgraph
        while len(visited) < sample_size and queue:
            current = queue.pop(0)
            neighbors = [n for n in adj_list[current] if n not in visited]
            
            # Add some neighbors
            add_count = min(len(neighbors), sample_size - len(visited))
            if add_count > 0:
                selected = random.sample(neighbors, add_count)
                for neighbor in selected:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # If still not enough nodes, add random ones
        if len(visited) < sample_size:
            remaining = [i for i in range(num_nodes) if i not in visited]
            additional = random.sample(remaining, min(len(remaining), sample_size - len(visited)))
            visited.update(additional)
        
        sampled_nodes = sorted(list(visited))
        node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sampled_nodes)}
        
        new_x = x[sampled_nodes]
        
        new_edges = []
        new_edge_attrs = []
        node_set = set(sampled_nodes)
        
        for i in range(edge_index.size(1)):
            src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
            if src in node_set and tgt in node_set:
                new_edges.append([node_map[src], node_map[tgt]])
                if edge_attr is not None:
                    new_edge_attrs.append(edge_attr[i])
        
        if new_edges:
            new_edge_index = torch.tensor(new_edges, dtype=torch.long, device=x.device).t()
            new_edge_attr = torch.stack(new_edge_attrs) if new_edge_attrs else None
        else:
            new_edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
            new_edge_attr = None
        
        return new_x, new_edge_index, new_edge_attr

class OptimizedAttentionReadout(nn.Module):
    """More efficient attention readout"""
    def __init__(self, feat_dim):
        super().__init__()
        self.att = nn.Linear(feat_dim, 1)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, batch=None):
        if batch is None:
            # Single graph case
            att_logits = self.att(x)
            att_weights = torch.softmax(att_logits, dim=0)
            return torch.sum(att_weights * x, dim=0)
        else:
            # Batched case - handled by global pooling functions
            from torch_geometric.nn import global_add_pool
            return global_add_pool(x, batch)

class OptimizedGATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, edge_dim=1):
        super().__init__()
        self.dropout = 0.4  
        
        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=num_heads, 
                             dropout=self.dropout, edge_dim=edge_dim, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_channels * num_heads)
        self.gat2 = GATv2Conv(hidden_channels * num_heads, out_channels, heads=1, 
                             concat=False, dropout=self.dropout, edge_dim=edge_dim)
        
    def forward(self, x, edge_index, edge_attr=None):
        x = F.elu(self.gat1(x, edge_index, edge_attr))
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index, edge_attr)
        return x

class GraphCL_GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, edge_dim=1, temperature=0.3):
        super().__init__()
        self.encoder = OptimizedGATEncoder(in_channels, hidden_channels, out_channels, num_heads, edge_dim)
        self.readout = OptimizedAttentionReadout(out_channels)
        self.temperature = temperature
        
        self.projection_head = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels // 2)
        )
        
        self.augmentation = GraphCLAugmentation()

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        
        node_emb = self.encoder(x, edge_index, edge_attr)
        graph_emb = self.readout(node_emb, batch)
        
        return node_emb, graph_emb
    
    def augment_graph(self, x, edge_index, edge_attr=None, aug_type='random'):
        """Apply graph augmentation"""
        if aug_type == 'random':
            aug_type = random.choice(['node_drop', 'edge_pert', 'attr_mask', 'subgraph'])
        
        if aug_type == 'node_drop':
            return self.augmentation.node_dropping(x, edge_index, edge_attr, drop_ratio=0.15)
        elif aug_type == 'edge_pert':
            return self.augmentation.edge_perturbation(x, edge_index, edge_attr, drop_ratio=0.2, add_ratio=0.1)
        elif aug_type == 'attr_mask':
            return self.augmentation.attribute_masking(x, edge_index, edge_attr, mask_ratio=0.15)
        elif aug_type == 'subgraph':
            return self.augmentation.subgraph_sampling(x, edge_index, edge_attr, sample_ratio=0.75)
        else:
            return x, edge_index, edge_attr
    
    def contrastive_loss(self, z1, z2):
        """Compute InfoNCE loss for contrastive learning"""
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        sim_matrix = torch.mm(z1, z2.t()) / self.temperature
        
        batch_size = z1.size(0)
        labels = torch.arange(batch_size, device=z1.device)
        
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss

def optimized_train_graphcl_model(patient_graphs, filtered_graphs, in_channels, hidden_channels, 
                                 out_channels, num_heads, epochs=100, lr=1e-4, batch_size=8):
    """Optimized training with GraphCL"""
    
    model = GraphCL_GAT(in_channels, hidden_channels, out_channels, num_heads)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_loss = float('inf')
    patience = 30
    cnt_wait = 0
    
    dataset = GraphDataset(filtered_graphs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           collate_fn=custom_collate_fn, num_workers=0)
    
    print(f"Training GraphCL on {len(filtered_graphs)} graphs with batch size {batch_size}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0

        for batch_graphs, batch_pids in dataloader:
            optimizer.zero_grad()
            
            # Collect all graph embeddings for the batch
            original_embeddings = []
            augmented_embeddings = []
            
            for graph in batch_graphs:
                graph = graph.to(device)
                x = graph.x.to(device)
                edge_index = graph.edge_index.to(device)
                edge_attr = graph.edge_attr.to(device) if graph.edge_attr is not None else None

                if edge_attr is not None and edge_attr.dim() == 1:
                    edge_attr = edge_attr.unsqueeze(-1)

                _, graph_emb = model(x, edge_index, edge_attr)
                proj_emb = model.projection_head(graph_emb)
                original_embeddings.append(proj_emb)
                
                # Augmented graph embedding
                try:
                    aug_x, aug_edge_index, aug_edge_attr = model.augment_graph(x, edge_index, edge_attr)
                    if aug_x.size(0) > 0:  
                        _, aug_graph_emb = model(aug_x, aug_edge_index, aug_edge_attr)
                        aug_proj_emb = model.projection_head(aug_graph_emb)
                        augmented_embeddings.append(aug_proj_emb)
                    else:
                        # Fallback: use original embedding with noise
                        noise = torch.randn_like(proj_emb) * 0.1
                        augmented_embeddings.append(proj_emb + noise)
                except Exception as e:
                    print(f"Warning: Augmentation failed, using noisy original - {e}")
                    noise = torch.randn_like(proj_emb) * 0.1
                    augmented_embeddings.append(proj_emb + noise)
            
            if len(original_embeddings) > 0 and len(augmented_embeddings) > 0:
                # Stack embeddings
                z1 = torch.stack(original_embeddings)
                z2 = torch.stack(augmented_embeddings)
                
                loss = model.contrastive_loss(z1, z2)
                
                # Backward pass
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1

        if batch_count > 0:
            avg_loss = total_loss / batch_count
            if avg_loss < best_loss:
                best_loss = avg_loss
                cnt_wait = 0
                # Save model
                torch.save(model.state_dict(), '/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/Proteomics/PlasmaAAA/GNN_models/GraphCL/GATv2/AttenPool/best_model_optimized_d2.pth')
            else:
                cnt_wait += 1
                if cnt_wait >= patience:
                    print("Early stopping!")
                    break

            if epoch % 5 == 0:
                print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')

    # Generate embeddings
    model.load_state_dict(torch.load('/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/Proteomics/PlasmaAAA/GNN_models/GraphCL/GATv2/AttenPool/best_model_optimized_d2.pth'))
    model.eval()
    embeddings = []

    with torch.no_grad():
        for graph in patient_graphs.values():
            graph = graph.to(device)
            edge_attr = graph.edge_attr
            if edge_attr is not None and edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            _, graph_emb = model(graph.x, graph.edge_index, edge_attr)
            embeddings.append(F.normalize(graph_emb, p=2, dim=0).cpu().numpy())

    return list(patient_graphs.keys()), embeddings

def run_graphcl_trials(patient_graphs, filtered_graphs, in_channels, hidden_channels, 
                      out_channels, num_heads, num_runs=3):
    """Run multiple GraphCL training trials"""
    patient_ids = list(patient_graphs.keys())
    all_embeddings = {pid: [] for pid in patient_ids}

    for run in range(num_runs):
        print(f"\nGraphCL Run {run + 1}/{num_runs}")
        _, embeddings = optimized_train_graphcl_model(
            patient_graphs, filtered_graphs, in_channels, 
            hidden_channels, out_channels, num_heads,
            epochs=100,
            lr=1e-4,
            batch_size=8
        )
        for pid, emb in zip(patient_ids, embeddings):
            all_embeddings[pid].append(emb)

    # Compute statistics
    avg_embeddings = {}
    std_embeddings = {}
    for pid in patient_ids:
        embs = np.stack(all_embeddings[pid])
        avg_embeddings[pid] = np.mean(embs, axis=0)
        std_embeddings[pid] = np.std(embs, axis=0)

    return patient_ids, avg_embeddings, std_embeddings

# Example usage (assuming you have patient_graphs and filtered_graphs defined):

in_channels = patient_graphs[list(patient_graphs.keys())[0]].x.shape[1]
hidden_channels = 128
out_channels = 128
num_heads = 8

patient_ids, avg_embs, std_embs = run_graphcl_trials(
    patient_graphs=patient_graphs,
    filtered_graphs=filtered_graphs,
    in_channels=in_channels,
    hidden_channels=hidden_channels,
    out_channels=out_channels,
    num_heads=num_heads,
    num_runs=2
)

######################################################################################################################################################

import os
import numpy as np

def save_embeddings_as_dict_and_arrays(patient_ids, avg_embeddings, std_embeddings, 
                                        dict_file='embeddings_dict_d2.npy', 
                                        array_file='embeddings_array_d2.npy', 
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


save_dir = '/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/Proteomics/PlasmaAAA/GNN_models/GraphCL/GATv2/AttenPool/'

save_embeddings_as_dict_and_arrays(patient_ids, avg_embs, std_embs, 
                                   dict_file='embeddings_dict_d2.npy', 
                                   array_file='embeddings_array_d2.npy', 
                                   save_dir=save_dir)
