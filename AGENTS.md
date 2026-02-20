# 仓库指南

  ## 项目结构与模块组织

  核心代码位于 mini_agent/：

- mini_agent/agent.py 和 mini_agent/cli.py 负责运行时行为。
- mini_agent/tools/ 包含内置工具（bash、files、MCP、skills、notes）。
- mini_agent/llm/ 存放各模型提供方客户端及封装。
- mini_agent/config/ 包含配置模板（例如 config-example.yaml）。
- mini_agent/skills/ 存放内置技能资源与加载器。

  测试位于 tests/（单元 + 集成），文档位于 docs/，可运行示例位于 examples/，初始化辅助脚本位于 scripts/。

  ## 构建、测试与开发命令

- uv sync：根据 pyproject.toml/uv.lock 安装项目及开发依赖。
- uv run python -m mini_agent.cli：在本地开发模式运行 CLI。
- uv run pytest tests/ -v：运行完整测试套件。
- uv run pytest tests/test_agent.py tests/test_note_tool.py -v：快速运行核心冒烟测试。
- git submodule update --init --recursive：克隆后初始化技能子模块内容。

  ## 编码风格与命名规范

  使用 Python 3.10+，遵循 PEP 8 默认规范、4 空格缩进，并为新增/修改代码添加类型注解。遵循现有命名模式：

- 模块/函数/变量：snake_case
- 类：PascalCase
- 常量：UPPER_SNAKE_CASE

  保持工具 API 明确，并与 mini_agent/tools/base.py 一致。为公共类/函数编写简洁 docstring。仓库未强制统一格式化工具，请与邻近文件风格保持一致并尽量减少 diff。

  ## 测试指南

  使用 pytest（已启用 pytest-asyncio）。测试放在 tests/ 下，文件/函数命名为 test_*.py 和 test_*。行为变更时需新增或更新测试，尤其是工具执行、MCP 加载和会话流程相关部分。建议先运行与改动模块相邻的聚焦测试，再跑全量
  测试。
