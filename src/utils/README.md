# `src/utils` 工具模块说明

本目录保存通用工具函数。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `config.py` | 读取和校验配置文件。 |
| `seed.py` | 固定随机种子。 |
| `logging.py` | 日志和实验目录管理。 |

## `config.py`

建议函数：

| 函数 | 职责 |
| --- | --- |
| `load_config` | 读取 yaml 配置。 |
| `validate_config` | 检查必要字段。 |
| `resolve_paths` | 将相对路径转换为项目内路径。 |

配置对象应至少包含：

- `data`
- `task`
- `model`
- `training`
- `output`

## `seed.py`

建议函数：

```python
set_seed(seed)
```

应固定：

- Python random。
- NumPy。
- PyTorch CPU。
- PyTorch CUDA。

## `logging.py`

建议函数：

| 函数 | 职责 |
| --- | --- |
| `create_experiment_dir` | 创建实验输出目录。 |
| `save_config_copy` | 保存配置副本。 |
| `setup_logger` | 创建日志器。 |

## 工具层边界

工具层可以被其他模块调用。

工具层不应反向依赖：

- `models`
- `engine`
- `scripts`

## 注意事项

- 路径统一相对项目根目录。
- 日志目录应包含数据集名和模型名。
- 配置校验应尽早失败，避免训练中途才发现字段缺失。
