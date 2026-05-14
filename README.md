下面给你一个**适合课程大作业 + 可复现实验 + 后续扩展**的 ASTGCN 项目结构。核心原则是：**不要把数据处理、图构建、模型、训练、评估全部写在一个脚本里**。ASTGCN 本身模块较多，如果结构不拆清楚，后面调维度会非常痛苦。

论文原文中，ASTGCN 的输入是交通网络图上的历史测量值：交通网络定义为图 (G=(V,E,A))，每个节点在每个时间片产生 (F) 维特征，历史输入为 (\mathcal{X}\in \mathbb{R}^{N\times F\times \tau})，预测目标是所有节点未来 (T_p) 个时间片的交通流 (Y\in \mathbb{R}^{N\times T_p})。 论文整体模型由 recent、daily-periodic、weekly-periodic 三个独立但结构相同的组件组成，每个组件由多个 ST block 和全连接层构成，最后通过参数矩阵加权融合。

---

# 一、推荐项目总结构

建议项目命名为：

```text
ASTGCN-PEMS04/
├── configs/
│   ├── pems04.yaml
│   └── pems08.yaml
│
├── data/
│   ├── raw/
│   │   └── PEMS04/
│   │       ├── pems04.npz
│   │       └── distance.csv
│   ├── processed/
│   │   └── PEMS04/
│   └── README.md
│
├── src/
│   └── astgcn/
│       ├── __init__.py
│       │
│       ├── data/
│       │   ├── dataset.py
│       │   ├── window.py
│       │   ├── scaler.py
│       │   └── graph.py
│       │
│       ├── models/
│       │   ├── astgcn.py
│       │   ├── component.py
│       │   ├── st_block.py
│       │   ├── attention.py
│       │   ├── cheb_conv.py
│       │   ├── temporal_conv.py
│       │   └── fusion.py
│       │
│       ├── baselines/
│       │   ├── historical_average.py
│       │   ├── var.py
│       │   ├── svr.py
│       │   ├── lstm.py
│       │   └── gru.py
│       │
│       ├── engine/
│       │   ├── trainer.py
│       │   ├── evaluator.py
│       │   ├── predictor.py
│       │   └── checkpoint.py
│       │
│       ├── metrics.py
│       ├── losses.py
│       ├── logger.py
│       └── utils.py
│
├── scripts/
│   ├── prepare_data.py
│   ├── train_astgcn.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── run_baselines.py
│   └── ablation.py
│
├── tests/
│   ├── test_dataset_shape.py
│   ├── test_graph.py
│   ├── test_attention.py
│   └── test_model_forward.py
│
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   ├── predictions/
│   └── figures/
│
├── notebooks/
│   ├── 01_data_check.ipynb
│   ├── 02_graph_check.ipynb
│   └── 03_result_visualization.ipynb
│
├── docs/
│   ├── model_design.md
│   ├── data_preprocessing.md
│   └── experiment_report.md
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

# 二、数据层：`src/astgcn/data/`

PEMS04 在你们大作业材料中属于交通状态公开数据集，数据类型包括速度、流量、占有率，检测器数量为 307，并且有邻接矩阵；相关论文正是 ASTGCN。 所以数据层至少要处理两个东西：

```text
pems04.npz       时间序列数据
distance.csv     图结构/距离边表
```

## 1. `graph.py`

负责把 `distance.csv` 转换成模型需要的图结构。

建议函数：

```python
def load_distance_csv(path):
    """读取 from, to, cost 三列边表。"""

def build_adjacency_matrix(edges, num_nodes, directed=False, weighted=False):
    """构建 A ∈ R^{N×N}。"""

def scaled_laplacian(adj):
    """计算缩放拉普拉斯矩阵。"""

def cheb_polynomials(scaled_lap, K):
    """生成 K 阶 Chebyshev 多项式。"""
```

注意：论文中明确把交通网络定义为**无向图** (G=(V,E,A))。 但 PEMS04 的 `distance.csv` 是 `from-to-cost` 边表形式，看起来像有向边。工程上你要做一个选择：

```text
复现论文：默认 directed=False，把邻接矩阵对称化
做消融实验：可以额外提供 directed=True
```

不要在代码里偷偷默认。要在 `configs/pems04.yaml` 里写清楚：

```yaml
graph:
  directed: false
  weighted: false
  cheb_order: 3
```

## 2. `window.py`

负责构造 ASTGCN 的三个输入片段。

论文中，模型沿时间轴截取三类时间序列片段：

```text
recent segment          近期片段
daily-periodic segment  日周期片段
weekly-periodic segment 周周期片段
```

三者分别输入到 recent、daily、weekly 三个组件中。论文设定采样频率为每天 (q) 次，预测窗口大小为 (T_p)，并截取长度为 (T_h,T_d,T_w) 的三类输入片段。

建议函数：

```python
def get_recent_segment(data, t0, Th):
    """取 [t0-Th+1, ..., t0]。"""

def get_daily_segment(data, t0, Td, Tp, q):
    """取过去若干天中与预测窗口相同时间段的数据。"""

def get_weekly_segment(data, t0, Tw, Tp, q):
    """取过去若干周中与预测窗口相同星期属性、相同时间段的数据。"""
```

PEMS04 通常是 5 分钟一个时间片，所以：

```text
q = 24 × 60 / 5 = 288
```

也就是一天 288 个采样点。

## 3. `dataset.py`

负责封装 PyTorch Dataset。

每个样本应该返回：

```python
{
    "x_h": Tensor,  # [N, F, Th]
    "x_d": Tensor,  # [N, F, Td]
    "x_w": Tensor,  # [N, F, Tw]
    "y":   Tensor,  # [N, Tp]
}
```

加上 batch 后：

```text
x_h: [B, N, F, Th]
x_d: [B, N, F, Td]
x_w: [B, N, F, Tw]
y:   [B, N, Tp]
```

这里建议你坚持论文维度顺序：

```text
N × F × T
```

也就是：

```text
节点 × 特征 × 时间
```

但是 PyTorch 的 `Conv2d` 更喜欢：

```text
B × C × H × W
```

所以在模型内部再转换成：

```text
[B, F, N, T]
```

不要让 Dataset 一会儿输出 `[B,N,F,T]`，一会儿输出 `[B,F,N,T]`。这是 ASTGCN 复现中最容易出错的地方。

## 4. `scaler.py`

负责标准化。

建议实现：

```python
class StandardScaler:
    def fit(self, train_data):
        pass

    def transform(self, data):
        pass

    def inverse_transform(self, data):
        pass
```

注意：**只能用训练集 fit scaler**。不能拿全部数据算均值和方差，否则会发生数据泄漏。

---

# 三、模型层：`src/astgcn/models/`

论文说每个 ASTGCN 组件包含两个核心部分：时空注意力机制和时空卷积；其中时空注意力捕获动态时空相关性，时空卷积同时使用图卷积提取空间模式、使用普通卷积描述时间特征。 所以模型层应严格按论文模块拆分。

---

## 1. `attention.py`

放两个类：

```python
class SpatialAttention(nn.Module):
    """空间注意力 S ∈ R^{N×N}。"""

class TemporalAttention(nn.Module):
    """时间注意力 E ∈ R^{T×T}。"""
```

### 空间注意力

论文中空间注意力矩阵 (S) 是根据当前层输入动态计算的，元素 (S_{i,j}) 表示节点 (i) 和节点 (j) 的相关强度，之后通过 softmax 归一化，并在图卷积时与邻接矩阵一起用于动态调整节点间影响权重。

所以代码接口建议是：

```python
S = spatial_attention(x)
```

输入：

```text
x: [B, N, C, T]
```

输出：

```text
S: [B, N, N]
```

### 时间注意力

论文中时间注意力矩阵 (E) 表示不同时间片之间的依赖强度，归一化后直接作用于输入，实现对输入时间维度的信息融合。

接口建议：

```python
E = temporal_attention(x)
x_hat = apply_temporal_attention(x, E)
```

输入：

```text
x: [B, N, C, T]
```

输出：

```text
E: [B, T, T]
x_hat: [B, N, C, T]
```

---

## 2. `cheb_conv.py`

放 Chebyshev 图卷积。

论文采用谱图卷积，并使用 Chebyshev 多项式近似以避免大规模图上直接特征分解的高代价；K 阶 Chebyshev 卷积相当于聚合中心节点 0 到 (K-1) 阶邻居的信息。

建议类：

```python
class ChebGraphConv(nn.Module):
    """普通 Chebyshev 图卷积。"""

class ChebGraphConvWithSAtt(nn.Module):
    """带空间注意力的 Chebyshev 图卷积。"""
```

核心输入输出：

```text
输入 x:        [B, N, C_in, T]
Cheb 多项式:  [K, N, N]
空间注意力 S: [B, N, N]
输出:         [B, N, C_out, T]
```

关键逻辑：

```text
T_k(L) 与 S 做 Hadamard product
再对节点特征进行图卷积
```

对应论文中的思想是：

```text
用静态图结构 A / L 描述道路拓扑
用动态空间注意力 S 调整节点影响强弱
```

---

## 3. `temporal_conv.py`

放普通时间卷积。

论文中，在图卷积捕获空间邻域信息之后，会在时间维度堆叠标准卷积，用于融合相邻时间片信息。

建议类：

```python
class TemporalConv(nn.Module):
    """沿时间维度做标准卷积。"""
```

输入输出：

```text
输入: [B, N, C_in, T]
内部转为: [B, C_in, N, T]
Conv2d kernel 可以使用 (1, 3)
输出再转回: [B, N, C_out, T_out]
```

---

## 4. `st_block.py`

这是 ASTGCN 的核心模块。

论文明确说明：每个 ST block 包含时空注意力模块和时空卷积模块；多个 ST block 堆叠后用于提取更大范围的动态时空相关性。

建议结构：

```python
class STBlock(nn.Module):
    def __init__(...):
        self.temporal_attention = TemporalAttention(...)
        self.spatial_attention = SpatialAttention(...)
        self.cheb_conv_satt = ChebGraphConvWithSAtt(...)
        self.temporal_conv = TemporalConv(...)
        self.residual_conv = ...
        self.layer_norm = ...

    def forward(self, x):
        E = self.temporal_attention(x)
        x_tatt = apply_temporal_attention(x, E)

        S = self.spatial_attention(x_tatt)
        x_gcn = self.cheb_conv_satt(x_tatt, S)

        x_time = self.temporal_conv(x_gcn)

        out = residual + x_time
        out = norm(out)
        return out
```

一个 ST block 的数据流可以写成：

```text
输入 X
  ↓
Temporal Attention
  ↓
Spatial Attention
  ↓
Chebyshev Graph Convolution with Spatial Attention
  ↓
Temporal Convolution
  ↓
Residual Connection + Normalization
  ↓
输出 X'
```

---

## 5. `component.py`

表示一个完整的 ASTGCN 分支。

论文中三个组件共享相同网络结构，每个组件由若干 ST block 和一个全连接层组成。

建议类：

```python
class ASTGCNComponent(nn.Module):
    def __init__(self, num_blocks, ...):
        self.blocks = nn.ModuleList([
            STBlock(...) for _ in range(num_blocks)
        ])
        self.final_conv = ...
        self.fc = ...

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        y_hat = self.fc_or_conv(x)
        return y_hat
```

输入：

```text
x_h: [B, N, F, Th]
```

输出：

```text
y_h: [B, N, Tp]
```

同理：

```text
x_d → y_d
x_w → y_w
```

---

## 6. `fusion.py`

负责融合三个组件输出。

论文说三个组件输出最终通过参数矩阵进一步融合，得到最终预测结果。

建议类：

```python
class FusionLayer(nn.Module):
    def __init__(self, num_nodes, pred_len):
        self.W_h = nn.Parameter(torch.ones(num_nodes, pred_len))
        self.W_d = nn.Parameter(torch.ones(num_nodes, pred_len))
        self.W_w = nn.Parameter(torch.ones(num_nodes, pred_len))

    def forward(self, y_h, y_d, y_w):
        return self.W_h * y_h + self.W_d * y_d + self.W_w * y_w
```

输入输出：

```text
y_h: [B, N, Tp]
y_d: [B, N, Tp]
y_w: [B, N, Tp]
y:   [B, N, Tp]
```

---

## 7. `astgcn.py`

顶层模型。

```python
class ASTGCN(nn.Module):
    def __init__(...):
        self.recent_component = ASTGCNComponent(...)
        self.daily_component = ASTGCNComponent(...)
        self.weekly_component = ASTGCNComponent(...)
        self.fusion = FusionLayer(...)

    def forward(self, x_h, x_d, x_w):
        y_h = self.recent_component(x_h)
        y_d = self.daily_component(x_d)
        y_w = self.weekly_component(x_w)
        y = self.fusion(y_h, y_d, y_w)
        return y
```

整体数据流：

```text
x_h ── Recent Component ── y_h ┐
                               │
x_d ── Daily Component ─── y_d ├── Fusion ── y_hat
                               │
x_w ── Weekly Component ── y_w ┘
```

---

# 四、训练与评估层：`src/astgcn/engine/`

## 1. `trainer.py`

负责训练循环：

```python
class Trainer:
    def train_one_epoch(self):
        pass

    def validate(self):
        pass

    def fit(self):
        pass
```

训练时主要流程：

```text
读取 batch
  ↓
x_h, x_d, x_w, y
  ↓
model(x_h, x_d, x_w)
  ↓
loss(pred, y)
  ↓
backward
  ↓
optimizer.step
  ↓
记录 MAE / RMSE / MAPE
```

论文实验中使用 RMSE 和 MAE 作为评价指标。 你也可以加 MAPE，但要注意真实值接近 0 时 MAPE 会爆炸，需要做 mask。

## 2. `evaluator.py`

负责最终测试：

```python
class Evaluator:
    def evaluate(self, model, test_loader):
        pass
```

注意：最终评估一定要在 `inverse_transform` 之后做。否则你的 MAE/RMSE 是标准化尺度下的误差，不是实际交通流量误差。

## 3. `checkpoint.py`

负责保存和加载模型：

```python
def save_checkpoint(model, optimizer, epoch, metrics, path):
    pass

def load_checkpoint(model, optimizer, path):
    pass
```

保存内容建议包括：

```text
model_state_dict
optimizer_state_dict
epoch
best_val_mae
config
scaler_mean
scaler_std
```

---

# 五、基础机器学习 baseline：`src/astgcn/baselines/`

你这个不是纯论文复现，还是机器学习大作业。大作业材料里明确写到要包含课程中涉及到的基础机器学习算法。 所以不能只做 ASTGCN。至少要做几个 baseline。

论文原文比较了 HA、ARIMA、VAR、LSTM、GRU、STGCN、GLU-STGCN、GeoMAN 等方法。 对你的课程项目，建议分三层 baseline：

```text
第一层：简单统计方法
- Historical Average, HA

第二层：传统机器学习 / 时间序列方法
- SVR
- Random Forest Regressor
- VAR 或 ARIMA

第三层：深度学习方法
- LSTM
- GRU
```

其中最适合你们报告展示的是：

```text
HA
SVR / RandomForest
LSTM
GRU
ASTGCN
```

这样可以体现从传统机器学习到深度学习、再到图神经网络的递进关系。

---

# 六、配置文件：`configs/pems04.yaml`

建议把所有超参数放到 YAML，不要写死在 Python 脚本里。

```yaml
dataset:
  name: PEMS04
  data_path: data/raw/PEMS04/pems04.npz
  distance_path: data/raw/PEMS04/distance.csv
  num_nodes: 307
  input_dim: 3
  target_dim: 0
  points_per_day: 288

time_window:
  pred_len: 12
  recent_len: 12
  daily_len: 12
  weekly_len: 12

split:
  train_ratio: 0.6
  val_ratio: 0.2
  test_ratio: 0.2

graph:
  directed: false
  weighted: false
  cheb_order: 3

model:
  num_blocks: 2
  in_channels: 3
  hidden_channels: 64
  cheb_channels: 64
  time_channels: 64
  dropout: 0.0

train:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: Adam
  loss: MAE
  device: cuda

log:
  save_dir: outputs/
  early_stop_patience: 10
```

这里 `pred_len=12` 通常表示预测未来 12 个 5 分钟时间片，也就是未来 1 小时。

---

# 七、脚本层：`scripts/`

## 1. `prepare_data.py`

职责：

```text
读取原始 npz
读取 distance.csv
划分 train / val / test
fit scaler
构造样本索引
保存 processed 文件
```

命令示例：

```bash
python scripts/prepare_data.py --config configs/pems04.yaml
```

## 2. `train_astgcn.py`

职责：

```text
读取配置
加载数据
构建图结构
初始化 ASTGCN
训练
保存最优模型
```

命令示例：

```bash
python scripts/train_astgcn.py --config configs/pems04.yaml
```

## 3. `evaluate.py`

职责：

```text
加载 checkpoint
在 test set 上评估
输出 MAE / RMSE / MAPE
保存预测结果
```

## 4. `run_baselines.py`

职责：

```text
运行 HA / SVR / RF / LSTM / GRU
生成对比表格
```

## 5. `ablation.py`

职责：

做消融实验：

```text
ASTGCN
去掉 Spatial Attention
去掉 Temporal Attention
去掉 Daily Component
去掉 Weekly Component
只保留 Recent Component
```

论文中也设计了去掉时空注意力的退化版本 MSTGCN，用来验证注意力机制的作用。 所以你报告里做这个消融会比较有说服力。

---

# 八、你应该优先实现的顺序

不要一上来就写完整 ASTGCN。建议按下面顺序推进：

```text
1. 读取 PEMS04.npz，确认数据 shape
2. 读取 distance.csv，构建 A、L、Cheb 多项式
3. 构造 Dataset，确保 x_h/x_d/x_w/y 维度正确
4. 实现 HA baseline，先跑通评估流程
5. 实现单个 ChebGraphConv
6. 实现 SpatialAttention 和 TemporalAttention
7. 实现一个 STBlock
8. 实现一个 ASTGCNComponent
9. 实现完整 ASTGCN 三分支融合
10. 加入训练、验证、测试、保存模型
11. 加入 baseline 对比和消融实验
```

最重要的是第 3 步和第 8 步。ASTGCN 出错最多的地方不是公式，而是维度：

```text
[B, N, F, T]
[B, F, N, T]
[B, N, C, T]
[B, N, T]
```

你要在每个模块的 `forward()` 里写清楚输入输出 shape。

---

# 九、最终报告可以按这个软件结构写

你的报告中可以这样描述实现架构：

```text
本项目按照数据处理层、图结构构建层、模型层、训练评估层和实验对比层进行模块化设计。

数据处理层负责读取 PEMS04 时序数据和 distance.csv 图结构文件，并按照 ASTGCN 论文中的 recent、daily-periodic、weekly-periodic 三种时间依赖构造输入样本。

图结构构建层负责根据传感器连接关系构造邻接矩阵、归一化拉普拉斯矩阵和 Chebyshev 多项式，为图卷积提供空间拓扑基础。

模型层严格对应 ASTGCN 论文结构，由 Spatial Attention、Temporal Attention、Chebyshev Graph Convolution、Temporal Convolution、ST Block、ASTGCN Component 和 Fusion Layer 组成。

训练评估层负责模型训练、验证、测试、指标计算和权重保存。实验对比层实现 HA、SVR、LSTM、GRU 等基础模型，用于和 ASTGCN 进行性能比较。
```

---

# 十、最关键的工程判断

你这个项目不要追求“代码最短”，要追求：

```text
模块边界清楚
输入输出维度清楚
论文结构对应清楚
实验流程可复现
baseline 和 ablation 能支撑报告
```

ASTGCN 的代码包核心不是一个 `model.py`，而是下面这条主线：

```text
PEMS04 时序数据 + distance.csv
        ↓
图结构构建：A / L / Cheb Polynomials
        ↓
时间片构造：recent / daily / weekly
        ↓
三个 ASTGCNComponent
        ↓
STBlock：TAtt + SAtt + ChebGCN + TemporalConv
        ↓
Fusion
        ↓
未来 Tp 个时间片所有节点交通流预测
```

你后面真正开始写代码时，先不要写完整训练。第一步应该是写 `test_model_forward.py`，只用随机张量验证：

```text
输入:
x_h: [B, N, F, Th]
x_d: [B, N, F, Td]
x_w: [B, N, F, Tw]

输出:
y_hat: [B, N, Tp]
```

只要这个 shape 测试没过，就不要继续写训练。ASTGCN 项目里，维度错误比优化器、学习率、损失函数更致命。
