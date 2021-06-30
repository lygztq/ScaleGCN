import os.path as osp
import os
import sys
import json
import time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from pytorch_memlab import MemReporter

sys.path.append(os.path.abspath(os.curdir))
from models import ScaleGCN
from utils.data_utils import get_stats
from utils import Profiler
prof = Profiler()

######### set random seed if necessary #########
# torch.manual_seed(128)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
################################################

# dataset_name = 'PubMed'
dataset_name = "Cora"
# dataset_name = "CiteSeer"

###### hyper-parameters ######
output_path = "./output/acc/scale_gcn"
if not osp.exists(output_path):
    os.makedirs(output_path)

num_layers = 16
hidden_dim = 64
dropout = 0.5
dropedge = 0.0
alpha = 0.1
beta_base = 0.1
fix_beta = False
scale_conv_type = "normal"
# scale_conv_type = "star"
weight_decay1 = 1e-2
weight_decay2 = 5e-4

if len(sys.argv) > 1:
    num_layers = int(sys.argv[1])
    fix_beta = bool(int(sys.argv[2]))
    scale_conv_type = sys.argv[3]
    if fix_beta:
        beta_base = 0.1
    dataset_name = sys.argv[4]

lr = 0.01
patience = 100
trial_times = 2

# data_split = "random"
# data_split = "full"
data_split = "public"
path = "~/data/datasets/{}_random".format(dataset_name) if data_split == "random" else "~/data/datasets/{}".format(dataset_name)
##############################

result_dict = {
    "hyper-parameters": {
        "dataset": dataset_name,
        "num_layer": num_layers,
        "dropout_rate": dropout,
        "hidden_dim": hidden_dim,
        "dropedge": dropedge,
        "weight_decay_reg": weight_decay1,
        "weight_decay_non_reg": weight_decay2,
        "lr": lr,
        "early_stop_patience": patience,
        "data_split_method": data_split,
    }
}

def one_trial():
    global dataset_name
    dataset = Planetoid(path, dataset_name, split=data_split, transform=T.NormalizeFeatures())
    data = dataset[0]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ScaleGCN(dataset.num_features, dataset.num_classes, hidden_channels=hidden_dim,
                   num_layers=num_layers, dropedge=dropedge, dropout=dropout, alpha=alpha, 
                   beta_base=beta_base, fix_beta=fix_beta, gconv_type=scale_conv_type).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.reg_params, weight_decay=weight_decay1),
        dict(params=model.non_reg_params, weight_decay=weight_decay2)
    ], lr=lr)
    reporter = MemReporter(model)

    def train():
        model.train()
        optimizer.zero_grad()
        loss_train = F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask])
        loss_train.backward()
        optimizer.step()
        return loss_train.item()


    @torch.no_grad()
    def test():
        model.eval()
        logits = model(data)
        loss_val = F.nll_loss(logits[data.val_mask], data.y[data.val_mask]).item()
        for _, mask in data('test_mask'):
            pred = logits[mask].max(1)[1]
            accs = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        for _, mask in data('val_mask'):
            pred = logits[mask].max(1)[1]
            val_accs = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        return loss_val, accs, val_accs
    
    best_val_loss = 9999999
    test_acc = 0
    bad_counter = 0
    best_epoch = 0
    for epoch in range(1, 1500):
        t1 = time.time()
        loss_tra = train()
        prof["train_time_per_epoch"].add_time(time.time() - t1)
        loss_val,acc_test_tmp, val_acc = test()
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            test_acc = acc_test_tmp
            bad_counter = 0
            best_epoch = epoch
        else:
            bad_counter += 1
        if epoch % 20 == 0: 
            log = 'Epoch: {:03d}, Train loss: {:.4f}, Val loss: {:.4f}, Val acc: {:.4f}, Test acc: {:.4f}'
            print(log.format(epoch, loss_tra, loss_val, val_acc, test_acc))
        if bad_counter == patience:
            break

    # reporter.report()
    return test_acc

result_dict["mean"], result_dict["err_bound"] = 0, 0
result_dict["num_trials"] = trial_times
result_dict["detail"] = []
result_dict["time"] = time.asctime( time.localtime(time.time()) )

for i in range(trial_times):
    print("trial {}/{}".format(i + 1, trial_times))
    result_dict["detail"].append(one_trial())

result_dict["profile"] = prof.dump()
result_dict["mean"], result_dict["err_bound"] = get_stats(result_dict["detail"], conf_interval=True)

out_name = "{}-{}-NVP-L{}-{}".format(dataset_name, data_split, num_layers, scale_conv_type)
if fix_beta: out_name += "-no_decay"

with open(osp.join(output_path, out_name), 'w') as f:
    json.dump(result_dict, f, indent=4)
