# 项目协作说明

本项目用于手工复现 ASTGCN，并对官方 PyTorch 项目进行结构化重构。

## 基本原则

- 使用简体中文撰写文档和说明。
- 代码实现应优先保持简单、可解释、方便扩展。
- 不直接照搬官方项目的文件组织方式，但保留 ASTGCN 的核心算法结构。
- 每个模块应明确输入、输出和职责边界。
- 数据集文件不提交到 git。

## 当前复现边界

第一阶段只复现：

- 数据集：PEMS04。
- 模型：ASTGCN recent component。
- 输入：过去 12 个时间步的第 0 个特征。
- 输出：未来 12 个时间步的第 0 个特征。

暂不实现：

- MSTGCN 对照模型。
- daily component。
- weekly component。
- 多特征联合预测。

## 提交习惯

每完成一个小任务后提交一次。

PowerShell 示例：

```powershell
git add -A; git commit -m "docs: 补充数据说明"
```

不要使用 `&&` 作为命令连接符。
