# 协作约定（ISML）

## 目录约定

- `Code/`：脚本与配置（不要放数据与运行产物）
- `Data/`：输入数据（尽量保留原始文件名；新增数据请建子目录）
- `Output/`：运行产物（图像/CSV/日志等）
- `Ref/`：参考文献与资料（不参与运行）
- `test/`：Demo/示例（用于验证或对照）
- `docs/`：仓库级 agent 上下文、机器可读索引、最近状态
- `AGENTS.md`：仓库级协作入口；新会话优先读这里

## 路径约定（可移植）

- 配置文件（YAML）里的 `data.path`、`output.output_dir` **优先写相对路径**。
- 代码会按 **YAML 所在目录** 解析相对路径；避免写机器相关的绝对路径（例如 `D:\...`）。

## 运行与依赖

- 推荐使用 `uv`：在 `Code/` 下 `uv sync`，再用 `uv run python ...` 运行脚本。
- 或使用 `pip`：在仓库根目录按 `requirements.txt` 创建虚拟环境（见 `README.md`）。

## 输出约定

- 所有新输出默认写到 `Output/` 下的新子目录。
- 不要把生成物写回 `Data/`（方便区分输入与产出）。

## Agent 协作文件

- 新会话或新协作者优先阅读：
  - `AGENTS.md`
  - `docs/AGENT_CONTEXT.md`
  - `docs/repo_index.yaml`
  - `docs/LAST_STATE.md`
- 如果新增或修改了脚本入口、配置结构、输出文件名、目录约定，请在同一次改动里同步更新：
  - `README.md`
  - `Code/README.md`
  - `Code/PROJECT_IO.md`
  - `docs/repo_index.yaml`
  - `docs/LAST_STATE.md`
- 如需补充仓库专属 agent 能力，请优先在 `.agents/skills/` 下新增或更新 skill，而不是把大量细节散落到多个无约束文档里。
