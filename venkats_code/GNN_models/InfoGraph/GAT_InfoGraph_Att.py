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

class InfoGraphDiscriminator(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feat_dim // 2, 1)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, graph_emb, subgraph_emb):
        combined = torch.cat([graph_emb, subgraph_emb], dim=-1)
        return torch.sigmoid(self.layers(combined))

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
        self.dropout = 0.3  
        
        # Simplified architecture
        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=num_heads, 
                             dropout=self.dropout, edge_dim=edge_dim, concat=True)
        self.gat2 = GATv2Conv(hidden_channels * num_heads, out_channels, heads=1, 
                             concat=False, dropout=self.dropout, edge_dim=edge_dim)
        
    def forward(self, x, edge_index, edge_attr=None):
        x = F.elu(self.gat1(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index, edge_attr)
        return x

class OptimizedGAT_InfoGraph(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, edge_dim=1):
        super().__init__()
        self.encoder = OptimizedGATEncoder(in_channels, hidden_channels, out_channels, num_heads, edge_dim)
        self.readout = OptimizedAttentionReadout(out_channels)
        self.discriminator = InfoGraphDiscriminator(out_channels)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        
        node_emb = self.encoder(x, edge_index, edge_attr)
        graph_emb = self.readout(node_emb, batch)
        
        return node_emb, graph_emb

def efficient_subgraph_sampling(x, edge_index, edge_attr=None, num_subgraphs=3, subgraph_ratio=0.4):
    """Enhanced subgraph generation with robust disconnected graph handling"""
    device = x.device
    num_nodes = x.size(0)
    subgraph_size = max(2, int(num_nodes * subgraph_ratio))
    
    # Handle very small graphs
    if num_nodes < 3:
        # Create multiple valid subgraphs even from tiny graphs
        subgraphs = []
        for i in range(min(num_subgraphs, num_nodes)):
            if num_nodes == 1:
                sub_x = x
            else:
                # Take different subsets of nodes
                end_idx = min(i + 1, num_nodes)
                sub_x = x[:end_idx] if end_idx > 0 else x[:1]
            empty_edges = torch.empty((2, 0), dtype=torch.long, device=device)
            subgraphs.append((sub_x, empty_edges, None))
        return subgraphs if subgraphs else [(x[:1], torch.empty((2, 0), dtype=torch.long, device=device), None)]
    
    # Pre-compute adjacency for faster lookup and connectivity checking
    adj_dict = {}
    edge_dict = {}  # Store edge indices for faster retrieval
    
    for i in range(edge_index.size(1)):
        src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
        if src not in adj_dict:
            adj_dict[src] = []
        if src not in edge_dict:
            edge_dict[src] = []
        adj_dict[src].append(tgt)
        edge_dict[src].append(i)
        
        # Add reverse for undirected graphs
        if tgt not in adj_dict:
            adj_dict[tgt] = []
        if tgt not in edge_dict:
            edge_dict[tgt] = []
        if src not in adj_dict[tgt]:  # Avoid duplicates for undirected
            adj_dict[tgt].append(src)
            edge_dict[tgt].append(i)
    
    subgraphs = []
    max_attempts = num_subgraphs * 3
    attempts = 0
    
    while len(subgraphs) < num_subgraphs and attempts < max_attempts:
        attempts += 1
        
        # Strategy 1: Try connected subgraph via random walk (70% of time)
        if random.random() < 0.7 and adj_dict:
            sampled_nodes = set()
            # Start from a node with edges
            start_candidates = [n for n in adj_dict.keys() if adj_dict[n]]
            if start_candidates:
                start_node = random.choice(start_candidates)
                sampled_nodes.add(start_node)
                current_frontier = [start_node]
                
                # Breadth-first expansion for better connectivity
                while len(sampled_nodes) < subgraph_size and current_frontier:
                    current = current_frontier.pop(0)
                    if current in adj_dict:
                        neighbors = [n for n in adj_dict[current] if n not in sampled_nodes]
                        if neighbors:
                            # Add 1-2 neighbors to maintain connectivity
                            num_to_add = min(len(neighbors), 2, subgraph_size - len(sampled_nodes))
                            selected_neighbors = random.sample(neighbors, num_to_add)
                            for neighbor in selected_neighbors:
                                sampled_nodes.add(neighbor)
                                current_frontier.append(neighbor)
                
                # Fill remaining with random nodes if needed
                remaining_nodes = [i for i in range(num_nodes) if i not in sampled_nodes]
                if remaining_nodes and len(sampled_nodes) < subgraph_size:
                    additional_needed = min(len(remaining_nodes), subgraph_size - len(sampled_nodes))
                    additional_nodes = random.sample(remaining_nodes, additional_needed)
                    sampled_nodes.update(additional_nodes)
            else:
                # Fallback to random sampling if no edges exist
                sampled_nodes = set(random.sample(range(num_nodes), min(subgraph_size, num_nodes)))
        
        # Strategy 2: Pure random sampling (30% of time or fallback)
        else:
            sampled_nodes = set(random.sample(range(num_nodes), min(subgraph_size, num_nodes)))
        
        if not sampled_nodes:
            sampled_nodes = {0}  # Ensure at least one node
        
        sampled_nodes = sorted(list(sampled_nodes))
        sampled_tensor = torch.tensor(sampled_nodes, device=device)
        
        # Extract subgraph edges with better error handling
        node_set = set(sampled_nodes)
        valid_edges = []
        valid_edge_attrs = []
        
        for i in range(edge_index.size(1)):
            src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
            if src in node_set and tgt in node_set:
                valid_edges.append([src, tgt])
                if edge_attr is not None:
                    valid_edge_attrs.append(edge_attr[i])
        
        if valid_edges:
            # Create node mapping for renumbering
            node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sampled_nodes)}
            
            # Remap edge indices
            remapped_edges = [[node_map[src], node_map[tgt]] for src, tgt in valid_edges]
            sub_edge_index = torch.tensor(remapped_edges, dtype=torch.long, device=device).t()
            
            if valid_edge_attrs:
                sub_edge_attr = torch.stack(valid_edge_attrs) if len(valid_edge_attrs) > 1 else valid_edge_attrs[0].unsqueeze(0)
            else:
                sub_edge_attr = None
        else:
            # No edges case - create empty edge tensor
            sub_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            sub_edge_attr = None
        
        # Validate subgraph before adding
        sub_x = x[sampled_tensor]
        if sub_x.size(0) > 0:  # Ensure we have nodes
            subgraphs.append((sub_x, sub_edge_index, sub_edge_attr))
    
    # Ensure we always return at least one subgraph
    if not subgraphs:
        # Emergency fallback - single node subgraph
        single_node_idx = torch.randint(0, num_nodes, (1,), device=device)
        sub_x = x[single_node_idx]
        empty_edges = torch.empty((2, 0), dtype=torch.long, device=device)
        subgraphs = [(sub_x, empty_edges, None)]
    
    return subgraphs

def optimized_train_model(patient_graphs, filtered_graphs, in_channels, hidden_channels, 
                         out_channels, num_heads, epochs=50, lr=1e-3, batch_size=4):
    """Optimized training with batching and reduced complexity"""
    
    model = OptimizedGAT_InfoGraph(in_channels, hidden_channels, out_channels, num_heads)
    model.to(device)
    
    # More aggressive optimizer settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    patience = 20  # Reduced patience
    cnt_wait = 0
    
    # Create dataset and dataloader with custom collate function
    dataset = GraphDataset(filtered_graphs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           collate_fn=custom_collate_fn, num_workers=0)
    
    graph_list = list(filtered_graphs.values())
    
    print(f"Training on {len(graph_list)} graphs with batch size {batch_size}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0

        for batch_graphs, batch_pids in dataloader:
            optimizer.zero_grad()
            batch_loss = 0
            
            for i, graph in enumerate(batch_graphs):
                graph = graph.to(device)
                x = graph.x.to(device)
                edge_index = graph.edge_index.to(device)
                edge_attr = graph.edge_attr.to(device) if graph.edge_attr is not None else None

                if edge_attr is not None and edge_attr.dim() == 1:
                    edge_attr = edge_attr.unsqueeze(-1)

                node_emb, graph_emb = model(x, edge_index, edge_attr)
                
                # Reduced subgraph sampling
                subgraphs = efficient_subgraph_sampling(x, edge_index, edge_attr, num_subgraphs=2)
                
                pos_scores = []
                neg_scores = []
                
                for sub_x, sub_edge_index, sub_edge_attr in subgraphs:
                    if sub_x.size(0) > 0:
                        try:
                            _, sub_graph_emb = model(sub_x, sub_edge_index, sub_edge_attr)
                            # Ensure embeddings are valid
                            if not torch.isnan(sub_graph_emb).any() and not torch.isinf(sub_graph_emb).any():
                                pos_score = model.discriminator(graph_emb.unsqueeze(0), sub_graph_emb.unsqueeze(0))
                                pos_scores.append(pos_score)
                        except Exception as e:
                            print(f"Warning: Skipping problematic subgraph - {e}")
                            continue
                
                # Reduced negative sampling - only from current batch
                neg_candidates = [g for j, g in enumerate(batch_graphs) if j != i]
                if neg_candidates:
                    # Sample only 1 negative graph per positive
                    neg_graph = random.choice(neg_candidates).to(device)
                    neg_x = neg_graph.x.to(device)
                    neg_edge_index = neg_graph.edge_index.to(device)
                    neg_edge_attr = neg_graph.edge_attr.to(device) if neg_graph.edge_attr is not None else None
                    
                    if neg_edge_attr is not None and neg_edge_attr.dim() == 1:
                        neg_edge_attr = neg_edge_attr.unsqueeze(-1)
                    
                    # Single negative subgraph
                    neg_subgraphs = efficient_subgraph_sampling(neg_x, neg_edge_index, neg_edge_attr, num_subgraphs=1)
                    
                    for neg_sub_x, neg_sub_edge_index, neg_sub_edge_attr in neg_subgraphs:
                        if neg_sub_x.size(0) > 0:
                            try:
                                _, neg_sub_graph_emb = model(neg_sub_x, neg_sub_edge_index, neg_sub_edge_attr)
                                # Validate negative embeddings too
                                if not torch.isnan(neg_sub_graph_emb).any() and not torch.isinf(neg_sub_graph_emb).any():
                                    neg_score = model.discriminator(graph_emb.unsqueeze(0), neg_sub_graph_emb.unsqueeze(0))
                                    neg_scores.append(neg_score)
                            except Exception as e:
                                print(f"Warning: Skipping problematic negative subgraph - {e}")
                                continue
                
                if pos_scores and neg_scores:
                    pos_scores = torch.cat(pos_scores)
                    neg_scores = torch.cat(neg_scores)
                    
                    # Simplified loss
                    pos_loss = -torch.log(pos_scores + 1e-8).mean()
                    neg_loss = -torch.log(1 - neg_scores + 1e-8).mean()
                    loss = pos_loss + neg_loss
                    
                    batch_loss += loss
            
            if batch_loss > 0:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Reduced clipping
                optimizer.step()
                total_loss += batch_loss.item()
                batch_count += 1

        if batch_count > 0:
            avg_loss = total_loss / batch_count
            if avg_loss < best_loss:
                best_loss = avg_loss
                cnt_wait = 0
                # Save model
                torch.save(model.state_dict(), '/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/Proteomics/PlasmaAAA/GNN_models/InfoGraph/GATv2/AttenPool/best_model_optimized_d2.pth')
            else:
                cnt_wait += 1
                if cnt_wait >= patience:
                    print("Early stopping!")
                    break

            if epoch % 5 == 0:  # Print less frequently
                print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        scheduler.step()

    # Generate embeddings
    model.load_state_dict(torch.load('/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/Proteomics/PlasmaAAA/GNN_models/InfoGraph/GATv2/AttenPool/best_model_optimized_d2.pth'))
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

def run_optimized_trials(patient_graphs, filtered_graphs, in_channels, hidden_channels, 
                        out_channels, num_heads, num_runs=3):  # Reduced runs
    """Optimized version with fewer runs and faster training"""
    patient_ids = list(patient_graphs.keys())
    all_embeddings = {pid: [] for pid in patient_ids}

    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        _, embeddings = optimized_train_model(
            patient_graphs, filtered_graphs, in_channels, 
            hidden_channels, out_channels, num_heads,
            epochs=30,  # Reduced epochs
            lr=2e-3,    # Higher learning rate
            batch_size=4  # Larger batch size
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

in_channels = patient_graphs[list(patient_graphs.keys())[0]].x.shape[1]
hidden_channels = 128  # Reduced from 128
out_channels = 128     # Reduced from 128
num_heads = 8         # Reduced from 8

patient_ids, avg_embs, std_embs = run_optimized_trials(
    patient_graphs=patient_graphs,
    filtered_graphs=filtered_graphs,
    in_channels=in_channels,
    hidden_channels=hidden_channels,
    out_channels=out_channels,
    num_heads=num_heads,
    num_runs=2  # Reduced from 10
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


save_dir = '/home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/Proteomics/PlasmaAAA/GNN_models/InfoGraph/GATv2/AttenPool/'

save_embeddings_as_dict_and_arrays(patient_ids, avg_embs, std_embs, 
                                   dict_file='embeddings_dict_d2.npy', 
                                   array_file='embeddings_array_d2.npy', 
                                   save_dir=save_dir)
