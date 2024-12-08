import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import to_dense_adj
import pandas as pd
import numpy as np

# -------------------------------
# Step 1: 数据加载和Union Representation
# -------------------------------
# 读取数据
edges = pd.read_csv("edge.py", sep=",", header=None, names=["source", "target"])
features = pd.read_csv("feature.py", sep=",", header=None)
labels = pd.read_csv("label.py", sep=",", header=None, names=["label"])

# 图数据初始化
edge_index = torch.tensor(edges.values.T, dtype=torch.long)
node_features = torch.tensor(features.values, dtype=torch.float)
labels = torch.tensor(labels.values, dtype=torch.long).squeeze()

# Union Representation: 将用户特征和交互特征联合表示
def build_union_representation(edge_index, node_features):
    union_features = []
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        union_feat = torch.cat((node_features[src], node_features[dst], node_features[src] - node_features[dst]))
        union_features.append(union_feat)
    return torch.stack(union_features)

union_features = build_union_representation(edge_index, node_features)

# -------------------------------
# Step 2: Interaction Graph Building
# -------------------------------
# 构建交互图：以Union为节点
def build_interaction_graph(edge_index):
    adj = to_dense_adj(edge_index).squeeze(0)
    interaction_adj = torch.zeros_like(adj)
    for i in range(adj.size(0)):
        for j in range(adj.size(1)):
            if adj[i, j] > 0:
                interaction_adj[i, j] = 1  # 保留有交互的关系
    return interaction_adj

interaction_adj = build_interaction_graph(edge_index)

# -------------------------------
# Step 3: PRM-GNN模型定义
# -------------------------------
class PRMGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super(PRMGNN, self).__init__()
        # 两层图注意力网络
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=0.6)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)
        self.fc = nn.Linear(out_channels, 2)  # 输出2个类别

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return F.log_softmax(self.fc(x), dim=1)

# -------------------------------
# Step 4: 训练和评估
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PRMGNN(in_channels=union_features.shape[1], hidden_channels=32, out_channels=16).to(device)

# 数据准备
data = Data(x=union_features, edge_index=edge_index, y=labels)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# 训练函数
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()

# 测试函数
def test():
    model.eval()
    logits = model(data.x, data.edge_index)
    pred = logits.argmax(dim=1)
    acc = (pred == data.y).sum().item() / data.y.size(0)
    return acc

# 训练循环
for epoch in range(1, 201):
    loss = train()
    if epoch % 10 == 0:
        acc = test()
        print(f"Epoch: {epoch}, Loss: {loss:.4f}, Test Acc: {acc:.4f}")

# -------------------------------
# Step 5: 保存结果
# -------------------------------
logits = model(data.x, data.edge_index)
pred = logits.argmax(dim=1)
result = pd.DataFrame({"Edge": range(len(pred)), "Prediction": pred.cpu().numpy()})
result.to_csv("prm_gnn_predictions.csv", index=False)
print("Results saved to prm_gnn_predictions.csv")
