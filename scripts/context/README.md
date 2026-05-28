# scripts/context 目录说明

这个目录保存提交前自动生成项目上下文包的脚本。

## 脚本职责

- `generate_file_index.py`：扫描项目目录，生成可靠的文件索引，不解析代码语义。
- `generate_public_api.py`：解析 `src/` 下的 Python 文件，提取公开类和函数签名。
- `build_context_pack.py`：把文件索引、公开 API 和关键文档拼接成单一上下文文本。
- `update_context_pack.py`：给 `pre-commit` 调用的入口，负责生成并把产物重新加入暂存区。

## 产物

- `docs/generated/FILE_INDEX.md`
- `docs/generated/PUBLIC_API.md`
- `docs/generated/CONTEXT_PACK.md`

## 约定

- 这些文件由脚本自动维护，不建议手工编辑。
- 提交前 hook 会自动更新并暂存它们，保证每次 commit 都携带最新上下文。
