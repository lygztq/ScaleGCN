import os, sys
import json
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Reddit
from math import isnan
from time import time
import torch_geometric.transforms as T
from torch_geometric.data import ClusterData, ClusterLoader, NeighborSampler

sys.path.append(os.path.abspath(os.curdir))
from models.scale_gcn import BiScaleGCN
from utils.data_utils import get_stats

### hyper-parameters ###
num_layers = 2
hidden_dim = 128
lr = 1e-2
alpha = 0.1
beta_base = 0.5
dropout = 0.2
weight_decay = 0
patience = 20
epochs = 200
final_update = 1
print_every = 1
output_path = "./output/reddit/nvp"
if not os.path.exists(output_path):
    os.makedirs(output_path)
hyper_params = {
    "num_layers": num_layers,
    "hidden_dim": hidden_dim,
    "lr": lr, "alpha": alpha, "beta_base": beta_base,
    "dropout": dropout, "weight_decay": weight_decay,
    "patience": patience, "epochs": epochs
}
###

dataset = Reddit("~/data/datasets/Reddit", transform=T.NormalizeFeatures())
data = dataset[0]

print('Partioning the graph... (this may take a while)')
cluster_data = ClusterData(data, num_parts=1500, recursive=False,
                           save_dir=dataset.processed_dir)
train_loader = ClusterLoader(cluster_data, batch_size=40, shuffle=True,
                             num_workers=12)
subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1], batch_size=1024,
                                  shuffle=False, num_workers=12)
print('Done!')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BiScaleGCN(dataset.num_features, dataset.num_classes, hidden_channels=hidden_dim,
                   num_layers=num_layers, num_final_update=final_update, dropout=dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def train():
    model.train()
    total_loss = total_nodes = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(data)
        loss = F.nll_loss(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        # print(loss)
        optimizer.step()

        nodes = data.train_mask.sum().item()
        total_loss += loss.item() * nodes
        total_nodes += nodes
    loss = total_loss / total_nodes
    return loss

@torch.no_grad()
def test():
    model.eval()
    
    out = model.inference(data.x, subgraph_loader=subgraph_loader, device=device)
    y_pred = out.argmax(dim=-1)

    out = F.log_softmax(out, dim=-1)
    val_loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask]).item()

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = y_pred[mask].eq(data.y[mask]).sum().item()
        accs.append(correct / mask.sum().item())

    return accs, val_loss

best_f1_val = 0.
f1_test = 0.
best_epoch = 0
bad_count = 0
times = []
train_loss_list = []
val_loss_list = []
train_accs_list = []
val_accs_list = []
print("Start training...")
for e in range(epochs):
    s_time = time()
    loss = train()
    times.append(time() - s_time)
    accs, val_loss = test()
    f1_train, f1_val, tmp_f1_test = accs
    if f1_val > best_f1_val:
        best_f1_val = f1_val
        f1_test = tmp_f1_test
        best_epoch = e
        bad_count = 0
    else:
        bad_count += 1
    if bad_count >= patience:
        break
    if (e+1) % print_every == 0:
        print('Epoch: {:02d}, Loss: {:.4f}, Train: {:.4f}, Val: {:.4f}, '
          'Test(best): {:.4f}'.format(e, loss, f1_train, f1_val, f1_test))
    train_loss_list.append(loss)
    val_loss_list.append(val_loss)
    train_accs_list.append(f1_train)
    val_accs_list.append(f1_val)

mean_train_time, err_bound_train_time = get_stats(times, conf_interval=True)
memory_usage = torch.cuda.max_memory_allocated(device)/ (1024*1024)
print("Best Epoch: {:02d}, Best Test F1: {:.4f}".format(best_epoch, f1_test))
print("Memory usage: {:.4f}(M)".format(memory_usage))

out_dict = {
    "hyper-params": hyper_params,
    "best_epoch": best_epoch,
    "f1_test": f1_test,
    "mean_train_time": mean_train_time,
    "err_bound_train_time": err_bound_train_time,
    "train_losses": train_loss_list,
    "val_losses": val_loss_list,
    "train_accs": train_accs_list,
    "val_accs": val_accs_list,
    "memory_usage": memory_usage
}
with open(os.path.join(output_path, "nvp-L{}-H{}-U{}".format(num_layers, hidden_dim, final_update)), "w") as f:
    json.dump(out_dict, f, indent=4)
