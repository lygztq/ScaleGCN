import os.path as osp
import os
import sys
import json
import torch
import torch.nn.functional as F
from torch_geometric.datasets import PPI
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from sklearn.metrics import f1_score
from time import time

sys.path.append(os.path.abspath(os.curdir))
from models import BiScaleGCN

### hyper-parameters
lr = 0.0005
weight_decay = 0
hidden_channels = 2048
num_layers = 4
dropout = 0.2
fix_beta = False
beta = 1#0.5
alpha = 0.1
patience = 1000
num_epochs = 8000
num_final_update = 3
###

path = "~/data/datasets/PPI"
train_dataset = PPI(path, split='train')
val_dataset = PPI(path, split='val')
test_dataset = PPI(path, split='test')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiScaleGCN(train_dataset.num_features, train_dataset.num_classes, 
                   hidden_channels=hidden_channels, num_layers=num_layers, dropout=dropout,
                   num_final_update=num_final_update).to(device)
loss_op = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=int(0.2*patience), verbose=True, min_lr=lr/100)

def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = loss_op(model(data, use_softmax=False), data.y)
        total_loss += loss.item() * data.num_graphs
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    ys, preds = [], []
    total_loss = 0
    for data in loader:
        ys.append(data.y)
        data = data.to(device)
        out = model(data, use_softmax=False)
        total_loss += loss_op(out, data.y)
        preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0, total_loss / len(loader.dataset)

bad_count = 0
best_val_loss = float("inf")
final_test_f1 = 0.
best_epoch = 0
train_times = []
for epoch in range(1, num_epochs + 1):
    s_time = time()
    loss = train()
    train_times.append(time() - s_time)
    val_f1, val_loss = test(val_loader)
    test_f1, _ = test(test_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        final_test_f1 = test_f1
        bad_count = 0
        best_epoch = epoch
    else:
        bad_count += 1

    if bad_count >= patience:
        break

    scheduler.step(loss)

    print('Epoch: {:02d}, Loss: {:.4f}, Val: {:.4f}, Test(Best): {:.4f}'.format(
        epoch, loss, val_f1, final_test_f1))

print("Best Epoch: {:02d}, final test f1: {:.4f}".format(best_epoch, final_test_f1))
print("Mem usage: {}".format(torch.cuda.max_memory_allocated(device)/(1024*1024)))
print("Train time: {:.4f}".format( sum(train_times) / len(train_times) ))
