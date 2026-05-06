# 脚本入口目录说明

本目录保存命令行入口脚本。

脚本层只负责：

- 解析命令行参数。
- 读取配置。
- 调用 `src/` 中的模块。

脚本层不应包含核心算法实现。

## 文件说明

| 文件 | 职责 |
| --- | --- |
| `prepare_data.py` | 数据预处理入口。 |
| `train_astgcn.py` | ASTGCN 训练入口。 |
| `evaluate_astgcn.py` | ASTGCN 测试评估入口。 |

## 推荐命令

数据预处理：

```powershell
python scripts/prepare_data.py --config configurations/PEMS04_astgcn.yaml
```

训练：

```powershell
python scripts/train_astgcn.py --config configurations/PEMS04_astgcn.yaml
```

评估：

```powershell
python scripts/evaluate_astgcn.py --config configurations/PEMS04_astgcn.yaml --model experiments/PEMS04/astgcn_recent/best.pt
```

## 参数约定

| 参数 | 说明 |
| --- | --- |
| `--config` | 配置文件路径。 |
| `--model` | 评估阶段加载的模型权重路径。 |
| `--device` | 可选，覆盖配置中的设备。 |

## 代码组织要求

脚本中建议只保留：

```python
def parse_args():
    ...

def main():
    ...

if __name__ == "__main__":
    main()
```

不要在脚本文件顶部直接加载数据或构造模型。
