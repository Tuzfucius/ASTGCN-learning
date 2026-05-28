# pdoc 文档

本目录存放 `astgcn` 包的 `pdoc` 静态文档输出。

## 入口

- `index.html`: 文档入口，会自动跳转到 `astgcn.html`
- `astgcn.html`: 包级总览文档
- `astgcn/`: 各子模块文档页面

## 生成方式

使用 `low_numpy` 环境中的 `pdoc` 生成，命令大致如下：

```powershell
$env:PYTHONPATH='src'
conda run -n low_numpy python -m pdoc -o "docs\pdoc文档" astgcn
```

## 说明

当前源码中部分类型注解仍使用 Python 3.10 风格的联合类型写法，`pdoc` 在 Python 3.9 环境下会输出警告，但不会阻止文档生成。
