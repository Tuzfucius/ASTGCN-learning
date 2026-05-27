# 第1章 模型实现与实验设计

本文档依据当前仓库中的 `src/astgcn/`、`scripts/`、`configs/` 与 `outputs/` 目录整理，目标是说明 ASTGCN 项目的实现方式、数据流、训练流程与实验设计。文档内容尽量与代码保持一致，并采用统一的张量约定与数学符号。

## 1.1 项目代码架构

项目采用“配置驱动 + 模块化实现 + 脚本入口”的组织方式。核心逻辑集中在 `src/astgcn/`，训练、推理和基线对比通过 `scripts/` 下的入口脚本调用；实验配置统一由 `configs/pems04.yaml` 管理；训练结果、预测结果和对比结果写入 `outputs/`。

### 1.1.1 目录职责

| 目录 / 文件 | 作用 |
|---|---|
| `src/astgcn/data/` | 数据读取、时间窗口构造、图结构生成、标准化与 DataLoader 封装 |
| `src/astgcn/models/` | 注意力、Chebyshev 图卷积、时空块、单分支组件、三分支融合与消融配置 |
| `src/astgcn/engine/` | 训练器、评估器、预测器与 checkpoint 保存 / 加载 |
| `src/astgcn/baselines/` | Historical Average、SVR、LSTM、GRU 等基线模型 |
| `scripts/train.py` | ASTGCN 训练入口 |
| `scripts/infer.py` | checkpoint 加载、测试集评估与预测保存 |
| `scripts/compare_baselines.py` | ASTGCN 与常用基线的统一对比 |
| `configs/pems04.yaml` | 数据集、窗口、图结构、模型、训练和输出目录配置 |
| `outputs/` | 训练 checkpoint、日志、预测结果、基线对比结果与消融实验结果 |

### 1.1.2 数据流总览

项目主流程可以概括为：

1. 从 `configs/pems04.yaml` 读取实验配置。
2. 从 `data/raw/PEMS04/pems04.npz` 读取原始时序数据，从 `distance.csv` 构造图结构。
3. 依据 recent / daily / weekly 三种时间尺度构造样本，并按时间顺序划分训练、验证和测试集。
4. 在训练集时间范围上计算标准化统计量，并对全量数据做标准化。
5. 将标准化后的三分支张量输入 ASTGCN，训练时以损失函数优化，验证时以 MAE 选择最优 checkpoint。
6. 在测试集上对预测结果反标准化并计算指标，同时保存预测数组、日志和图像结果。

### 1.1.3 模块间关系

ASTGCN 的前向计算由三层结构组成：

1. `ASTGCN`：三分支总模型，负责 recent / daily / weekly 的实例化和融合。
2. `ASTGCNComponent`：单分支时序建模单元，由多个 `STBlock` 堆叠而成。
3. `STBlock`：单个时空块，依次执行时间注意力、空间注意力、图卷积、时间卷积、残差连接与归一化。

其关系可以形式化表示为：

$$
\hat{Y}
=
\mathcal{F}\Big(
\mathcal{C}_r(X_r),
\mathcal{C}_d(X_d),
\mathcal{C}_w(X_w)
\Big),
$$

其中 $X_r, X_d, X_w$ 分别表示 recent、daily、weekly 分支输入，$\mathcal{C}_\cdot$ 表示单分支 ASTGCNComponent，$\mathcal{F}$ 表示三分支融合层。

## 1.2 数据处理与张量约定

### 1.2.1 原始数据形式

PEMS04 数据在代码中统一表示为三维张量：

$$
X \in \mathbb{R}^{T \times N \times F},
$$

其中：

1. $T$ 为时间步数；
2. $N$ 为传感器节点数；
3. $F$ 为每个节点的特征数。

配置文件中默认 `num_nodes = 307`，`input_dim = 3`，`points_per_day = 288`，对应 5 分钟采样粒度。

### 1.2.2 标准化

代码中的标准化器位于 `src/astgcn/data/scaler.py`。标准化统计量只在训练集覆盖的时间范围内计算，以避免验证集和测试集信息泄漏。设训练集均值与标准差分别为 $\mu$ 与 $\sigma$，则标准化定义为：

$$
\tilde{X} = \frac{X - \mu}{\sigma},
\qquad
\sigma_{f} \leftarrow \max(\sigma_f, 1).
$$

其中 $\sigma_f$ 为第 $f$ 个特征通道的标准差。标准化后，训练、验证和测试数据都沿用同一组参数。

### 1.2.3 时间窗口构造

项目为每个样本构造三类历史片段和一段未来预测目标：

1. recent：紧邻当前时刻的短期历史；
2. daily：与当前时刻在前若干天相同时间段对齐的日周期片段；
3. weekly：与当前时刻在前若干周相同星期和时间段对齐的周周期片段；
4. target：未来 $P$ 步预测目标。

设当前时刻为 $t_0$，预测长度为 $P$，recent 长度为 $H_r$，每天采样点数为 $S$，daily 使用天数为 $D$，weekly 使用周数为 $W$，则：

$$
X^{(r)}_{t_0} = X_{t_0-H_r+1:t_0+1},
$$

$$
X^{(d)}_{t_0}
=
\operatorname{Concat}\Big(
X_{t_0+1-iS:t_0+1-iS+P}
\Big)_{i=1}^{D},
$$

$$
X^{(w)}_{t_0}
=
\operatorname{Concat}\Big(
X_{t_0+1-7iS:t_0+1-7iS+P}
\Big)_{i=1}^{W},
$$

$$
Y_{t_0} = X_{t_0+1:t_0+P}.
$$

其中 `ASTGCNDataset` 会将输入从时间优先的 `[T, N, F]` 转换为模型使用的 `[N, F, T]`，即：

$$
\Phi\left([T, N, F]\right) = [N, F, T].
$$

最终，`DataLoader` 输出的 batch 张量约定为：

$$
X_r, X_d, X_w \in \mathbb{R}^{B \times N \times F \times T},
\qquad
Y \in \mathbb{R}^{B \times N \times P}.
$$

### 1.2.4 样本切分原则

候选时间点集合记为 $\mathcal{T}$，其构造方式要求所有窗口片段都在数据范围内。训练、验证和测试集合按时间顺序切分：

$$
\mathcal{T}
=
\mathcal{T}_{train} \cup \mathcal{T}_{val} \cup \mathcal{T}_{test},
\qquad
\mathcal{T}_{train} \prec \mathcal{T}_{val} \prec \mathcal{T}_{test}.
$$

这样可以保证后验时间信息不会进入前序训练阶段。

#### 伪代码：样本构造

```text
Input: 原始数据 X[T, N, F], recent长度 Hr, daily天数 D, weekly周数 W, 预测长度 P
Output: 三分支样本集合 S

for each合法时刻 t0 do
    Xr <- X[t0-Hr+1 : t0+1]
    Xd <- concatenate_{i=1..D} X[t0+1-i*S : t0+1-i*S+P]
    Xw <- concatenate_{i=1..W} X[t0+1-7*i*S : t0+1-7*i*S+P]
    Y  <- X[t0+1 : t0+P]
    add (Xr, Xd, Xw, Y) to S
end for
```

## 1.3 ASTGCN 模型实现

### 1.3.1 三分支输入结构

`src/astgcn/models/astgcn.py` 中的 `ASTGCN` 将输入分为 recent、daily、weekly 三个时间分支。每个分支单独进入 `ASTGCNComponent`，得到分支级预测：

$$
\hat{Y}_r = \mathcal{C}_r(X_r), \qquad
\hat{Y}_d = \mathcal{C}_d(X_d), \qquad
\hat{Y}_w = \mathcal{C}_w(X_w),
$$

其中每个分支输出均为：

$$
\hat{Y}_\ast \in \mathbb{R}^{B \times N \times P}.
$$

消融配置 `AblationConfig` 可以按需关闭任意分支，因此模型并不强制依赖三个分支同时启用。

### 1.3.2 单分支 ASTGCNComponent

单分支组件由多个 `STBlock` 堆叠构成。设输入为 $X^{(0)}$，第 $\ell$ 个时空块输出为 $X^{(\ell)}$，则：

$$
X^{(\ell)} = \mathrm{STBlock}^{(\ell)}\left(X^{(\ell-1)}\right),
 \qquad \ell = 1, \dots, L.
$$

其中 $L = \texttt{num\_blocks}$。堆叠完成后，代码将最后一层输出按节点和时间维展平：

$$
\operatorname{vec}\left(X^{(L)}\right) \in \mathbb{R}^{B \times N \times (C \cdot T)},
$$

再通过线性层映射到预测长度：

$$
\hat{Y} = \operatorname{Linear}\left(\operatorname{vec}\left(X^{(L)}\right)\right),
\qquad
\hat{Y} \in \mathbb{R}^{B \times N \times P}.
$$

该设计使每个时间分支都能独立学习“时空表征 -> 预测窗口”的映射。

### 1.3.3 STBlock 实现

`src/astgcn/models/st_block.py` 中的 `STBlock` 是模型的核心单元。其数据流可以概括为：

$$
X
\xrightarrow{\text{Temporal Attention}}
\tilde{X}
\xrightarrow{\text{Spatial Attention + Graph Conv}}
Z
\xrightarrow{\text{Temporal Conv}}
U
\xrightarrow{\text{Residual + ReLU + LayerNorm}}
X',
$$

其中：

1. 时间注意力用于建模不同时间片之间的相关性；
2. 空间注意力用于建模节点间动态关联；
3. 图卷积用于提取图结构上的局部传播模式；
4. 时间卷积用于进一步聚合时间轴上的局部模式；
5. 残差连接和层归一化用于稳定训练。

#### 时间注意力

时间注意力层在代码中输出形状为 `[B, T, T]` 的注意力矩阵，可写为：

$$
A_t = \operatorname{Softmax}\left(
V_t \, \sigma\!\left(\Psi_t(X)\right)
\right),
$$

其中 $\Psi_t(\cdot)$ 表示由可学习参数与输入张量组合得到的打分函数，$\sigma$ 为 Sigmoid。应用时间注意力后：

$$
\tilde{X} = X \otimes A_t,
$$

其中 $\otimes$ 表示沿时间维的加权求和。

#### 空间注意力

空间注意力输出形状为 `[B, N, N]`，用于刻画每个样本内部的节点依赖关系：

$$
A_s = \operatorname{Softmax}\left(
V_s \, \sigma\!\left(\Psi_s(X)\right)
\right).
$$

当空间注意力启用时，图卷积的传播矩阵将与 $A_s$ 逐元素相乘，从而得到样本自适应的动态图结构。

#### Chebyshev 图卷积

图结构由距离矩阵 `distance.csv` 构造。先得到邻接矩阵 $A$，再构造归一化拉普拉斯矩阵：

$$
L = I - D^{-1/2} A D^{-1/2},
$$

然后进行缩放：

$$
\tilde{L} = \frac{2L}{\lambda_{\max}} - I.
$$

Chebyshev 多项式递推为：

$$
T_0(\tilde{L}) = I,\qquad
T_1(\tilde{L}) = \tilde{L},\qquad
T_k(\tilde{L}) = 2\tilde{L}T_{k-1}(\tilde{L}) - T_{k-2}(\tilde{L}).
$$

因此，带空间注意力的图卷积可表示为：

$$
Z
=
\sum_{k=0}^{K-1}
 \left(
 T_k(\tilde{L}) \odot A_s
 \right)
 X \Theta_k,
$$

其中 $\odot$ 为逐元素乘法，$\Theta_k$ 为第 $k$ 阶可学习参数。

#### 时间卷积与残差

图卷积后接时间卷积，记为：

$$
U = \mathrm{TC}(Z).
$$

随后加入残差分支：

$$
R = W_r * X,
$$

其中 $W_r$ 是 $1 \times 1$ 卷积核。最终输出为：

$$
X' = \operatorname{LayerNorm}\left(\operatorname{ReLU}(U + R)\right).
$$

#### 伪代码：STBlock

```text
Input: X[B, N, F, T]
if temporal attention enabled then
    At <- TemporalAttention(X)
    X <- X with At applied along time dimension
end if

if graph convolution enabled then
    As <- SpatialAttention(X) or all-ones matrix
    Z  <- ChebyshevConv(X, As)
else
    Z  <- linear bypass(X)
end if

U <- TemporalConv(Z) or Identity
R <- ResidualConv(X)
X' <- LayerNorm(ReLU(U + R))
return X'
```

### 1.3.4 三分支融合

`src/astgcn/models/fusion.py` 中实现了可配置融合层。对于三分支输出：

$$
\hat{Y}_r,\ \hat{Y}_d,\ \hat{Y}_w \in \mathbb{R}^{B \times N \times P},
$$

默认 `matrix` 融合模式对应按节点和预测步长加权：

$$
\hat{Y}
=
W_r \odot \hat{Y}_r
+
W_d \odot \hat{Y}_d
+
W_w \odot \hat{Y}_w,
$$

其中：

$$
W_r, W_d, W_w \in \mathbb{R}^{N \times P}.
$$

若启用其他模式，代码还支持：

1. `average`：直接求均值；
2. `scalar`：使用分支级标量权重；
3. `concat_mlp`：将分支拼接后送入线性层。

这些选项主要用于消融实验与结构分析。

## 1.4 模型训练流程

### 1.4.1 训练入口与配置文件

训练入口位于 `scripts/train.py`。其运行逻辑为：

1. 读取 YAML 配置；
2. 固定随机种子；
3. 初始化日志目录与 checkpoint 目录；
4. 构建训练、验证和测试 DataLoader；
5. 依据图配置构造 Chebyshev 多项式；
6. 实例化 ASTGCN；
7. 绑定优化器与训练器；
8. 运行训练并保存最优模型。

训练中用到的关键配置字段包括：

1. `dataset`：数据路径、节点数、输入通道数、目标通道、采样粒度；
2. `time_window`：`pred_len`、`recent_len`、`daily_days`、`weekly_weeks`；
3. `graph`：是否有向、是否加权、Chebyshev 阶数；
4. `model`：块数、隐藏维度、卷积核大小；
5. `train`：batch size、epoch、学习率、权重衰减、设备、早停轮数；
6. `log`：checkpoint、日志和预测结果目录。

### 1.4.2 训练、验证与 checkpoint 保存

训练过程由 `src/astgcn/engine/trainer.py` 中的 `Trainer` 完成。其核心步骤是：

1. 模型切换到训练模式；
2. 取出 batch 中的 `recent`、`daily`、`weekly` 和 `target`；
3. 前向传播得到预测值；
4. 计算损失并反向传播；
5. 更新参数；
6. 计算每个 batch 的误差指标并做平均。

训练目标在代码中默认采用 MAE：

$$
\mathcal{L}_{\text{MAE}}
=
\frac{1}{B N P}
\sum_{b=1}^{B}
\sum_{n=1}^{N}
\sum_{p=1}^{P}
\left|
\hat{Y}_{bnp} - Y_{bnp}
\right|.
$$

验证阶段使用相同的前向接口，但不更新参数；当验证指标 `monitor` 改善时，保存 `outputs/checkpoints/best.pt`。

#### 伪代码：训练循环

```text
Input: model, train_loader, val_loader, optimizer, epochs, patience
best_metric <- +inf
bad_epochs <- 0

for epoch in 1..epochs do
    model.train()
    for batch in train_loader do
        pred <- model(batch)
        loss <- MAE(pred, target)
        loss.backward()
        optimizer.step()
    end for

    model.eval()
    val_metric <- evaluate(val_loader)
    if val_metric < best_metric then
        best_metric <- val_metric
        save_checkpoint(best.pt)
        bad_epochs <- 0
    else
        bad_epochs <- bad_epochs + 1
    end if

    if bad_epochs >= patience then
        break
    end if
end for
```

#### checkpoint 保存内容

`src/astgcn/engine/checkpoint.py` 保存的内容包括：

1. `model_state_dict`
2. `optimizer_state_dict`
3. `epoch`
4. `best_metric`
5. `config`
6. 标准化器参数 `scaler_mean` 与 `scaler_std`

这使得模型恢复时可以同时恢复参数、优化器状态和数据尺度信息。

### 1.4.3 主要超参数设置

`configs/pems04.yaml` 中的默认训练参数如下：

| 参数 | 值 |
|---|---|
| `batch_size` | 32 |
| `epochs` | 50 |
| `learning_rate` | 0.001 |
| `weight_decay` | 0.0001 |
| `optimizer` | Adam |
| `loss` | mae |
| `early_stop_patience` | 10 |
| `seed` | 42 |
| `num_blocks` | 2 |
| `hidden_channels` | 64 |
| `cheb_order` | 3 |
| `time_kernel_size` | 3 |
| `recent_len` | 12 |
| `daily_days` | 1 |
| `weekly_weeks` | 1 |
| `pred_len` | 12 |

说明：配置文件中保留了 `grad_clip` 字段，但当前训练器实现未显式调用梯度裁剪逻辑。若后续需要严格启用，可在 `Trainer.train_one_epoch()` 中插入 `torch.nn.utils.clip_grad_norm_`。

## 1.5 评价指标与基线模型设置

### 1.5.1 评价指标

代码层面在 `src/astgcn/metrics.py` 中实现了 MAE、RMSE 和 MAPE 三类指标。考虑到论文正文的常规写法，本文档重点给出 MAE 与 MSE 的数学表达，同时说明当前工程在 baseline 对比中额外保留了 RMSE 和 MAPE。

#### MAE

$$
\mathrm{MAE}
=
\frac{1}{B N P}
\sum_{b=1}^{B}
\sum_{n=1}^{N}
\sum_{p=1}^{P}
\left|
\hat{Y}_{bnp} - Y_{bnp}
\right|.
$$

#### MSE

$$
\mathrm{MSE}
=
\frac{1}{B N P}
\sum_{b=1}^{B}
\sum_{n=1}^{N}
\sum_{p=1}^{P}
\left(
\hat{Y}_{bnp} - Y_{bnp}
\right)^2.
$$

#### 代码中的补充指标

1. RMSE：

$$
\mathrm{RMSE} = \sqrt{\mathrm{MSE}}.
$$

2. MAPE：

$$
\mathrm{MAPE}
=
\frac{100\%}{B N P}
\sum_{b,n,p}
\left|
\frac{\hat{Y}_{bnp} - Y_{bnp}}{Y_{bnp}}
\right|.
$$

评估阶段统一在反标准化后的真实交通流量尺度上计算指标；这由 `Evaluator` 和 `compare_baselines.py` 中的 `inverse_transform_target()` 保证。

### 1.5.2 基线模型设置

`src/astgcn/baselines/` 与 `scripts/compare_baselines.py` 中实现并统一对比了以下模型：

1. HA（Historical Average）
2. SVR
3. LSTM
4. GRU
5. ASTGCN

所有可直接前向的基线模型均使用同一数据切分、同一预测长度和同一评估指标。对于 PyTorch 模型，训练采用与 ASTGCN 类似的 batch 级 MAE 优化；对于 SVR，则先将节点级样本展开后再进行拟合。这样可以保证横向比较的输入输出条件尽量一致。

### 1.5.3 当前对比结果

`outputs/comparison/baseline_metrics.csv` 记录了当前云端对比结果。该结果受训练轮数与 smoke test 参数影响，适合用于流程验证和相对趋势分析，不宜直接作为最终论文定稿数值。

| 模型 | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| HA | 25.0742 | 37.5859 | 296498.9502 |
| LSTM | 132.4166 | 171.5795 | 500368.6523 |
| GRU | 129.9815 | 168.7458 | 506249.1699 |
| ASTGCN | 139.5850 | 178.6584 | 498276.7090 |
| SVR | 39.3568 | 63.4045 | 198388.6230 |

`outputs/kaggle-astgcn-pems04-training.ipynb` 中还包含了按预测步长的误差曲线、单节点预测曲线、误差分布与预测-真实散点图，用于辅助分析模型在不同预测步长上的稳定性。

## 1.6 消融实验设计

`outputs/ablation-cloud-training.ipynb` 用于组织消融实验。其核心思想是在保持其余条件尽量一致的前提下，逐项关闭或替换模型中的关键结构，从而观察各模块对结果的影响。

### 1.6.1 消融项定义

当前 notebook 中定义的实验项如下：

| 实验名称 | 配置修改 | 目的 |
|---|---|---|
| `full_astgcn` | 无修改 | 完整模型基线 |
| `recent_only` | `use_daily = False`, `use_weekly = False` | 验证周期分支的作用 |
| `no_temporal_attention` | `use_temporal_attention = False` | 验证时间注意力的作用 |
| `no_spatial_attention` | `use_spatial_attention = False` | 验证空间注意力的作用 |
| `identity_graph` | `graph_mode = "identity"` | 验证图结构传播的作用 |

其中 `AblationConfig` 通过布尔开关和图结构模式控制模型行为。消融实验的核心变量可概括为：

$$
\theta_{ablation}
=
\big(
u_r, u_d, u_w,
u_t, u_s, u_g, u_c,
g, f
\big),
$$

其中分别表示三分支启用状态、时间注意力、空间注意力、图卷积、时间卷积、图模式和融合模式。

### 1.6.2 实验设计原则

1. 除被考察模块外，其余训练超参数保持一致。
2. 训练、验证和测试划分保持一致。
3. 每个实验都保存独立的 checkpoint 与预测文件。
4. 通过 `outputs/ablation_cloud/ablation_results.csv` 汇总测试指标。
5. 附加保存预测曲线与相对变化率图，便于比较模块增减带来的影响。

### 1.6.3 当前消融结果

根据 notebook 输出，当前完整实验中的测试指标如下：

| 实验 | best_val_mae | test_mae | test_rmse | test_mape |
|---|---:|---:|---:|---:|
| full_astgcn | 0.135544 | 22.044945 | 35.672867 | 1905319.375 |
| recent_only | 0.144014 | 22.723455 | 35.022690 | 1313689.500 |
| no_temporal_attention | 0.135867 | 22.023161 | 35.615143 | 1832686.875 |
| no_spatial_attention | 0.132057 | 21.268803 | 33.801922 | 1610077.500 |
| identity_graph | 0.135601 | 22.059681 | 35.675758 | 1911414.000 |

从当前结果看：

1. recent_only 的测试误差高于完整模型，说明日周期和周周期分支仍提供了有效信息；
2. no_temporal_attention 与 full_astgcn 接近，说明在当前训练设置下，时间注意力的边际增益较弱；
3. no_spatial_attention 的 MAE 反而更低，这表明该实验结果可能受到训练轮数、随机性或当前数据切分的影响，后续需要在更充分训练下重复验证；
4. identity_graph 与完整模型接近，说明在当前设置下，图结构本身并未表现出稳定、显著的优势。

因此，消融实验更适合作为结构贡献分析，而不应仅依据单次结果直接给出定论。

### 1.6.4 结论性说明

消融实验说明了当前实现具备如下可解释性：

1. 模型结构是可拆解的，便于逐项验证；
2. 三分支机制、注意力机制与图传播机制均可通过配置开关控制；
3. 结果输出已经形成统一的日志、预测文件和表格，便于后续写作和复现实验。

---

## 附：与本文档直接对应的关键文件

1. `src/astgcn/data/dataset.py`
2. `src/astgcn/data/window.py`
3. `src/astgcn/data/dataloader.py`
4. `src/astgcn/data/graph.py`
5. `src/astgcn/models/astgcn.py`
6. `src/astgcn/models/component.py`
7. `src/astgcn/models/st_block.py`
8. `src/astgcn/models/fusion.py`
9. `src/astgcn/engine/trainer.py`
10. `src/astgcn/engine/evaluator.py`
11. `src/astgcn/engine/checkpoint.py`
12. `scripts/train.py`
13. `scripts/infer.py`
14. `scripts/compare_baselines.py`
15. `configs/pems04.yaml`
16. `outputs/kaggle-astgcn-pems04-training.ipynb`
17. `outputs/ablation-cloud-training.ipynb`
