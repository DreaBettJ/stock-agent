"""Core Agent implementation.

这个模块是 Mini-Agent 框架的核心实现了 Agent 的完整执行循环:
1. 与 LLM 交互,获取模型响应
2. 解析模型响应的工具调用
3. 执行工具并返回结果
4. 管理消息历史和上下文
5. 支持上下文自动摘要,防止超出 token 限制
6. 支持取消执行
"""

import asyncio
import json
from pathlib import Path
from time import perf_counter
from typing import Optional

import tiktoken

from .llm import LLMClient
from .logger import AgentLogger
from .schema import Message
from .tools.base import Tool, ToolResult
from .utils import calculate_display_width


# ============================================================
# ANSI 终端颜色代码
# 用于在终端输出文本,中显示不同颜色的提升可读性
# ============================================================
class Colors:
    """Terminal color definitions"""

    RESET = "\033[0m"      # 重置所有颜色/样式
    BOLD = "\033[1m"       # 粗体
    DIM = "\033[2m"        # 暗淡

    # 前景色 (文字颜色)
    RED = "\033[31m"       # 红色 - 通常用于错误
    GREEN = "\033[32m"     # 绿色 - 通常用于成功
    YELLOW = "\033[33m"    # 黄色 - 通常用于警告
    BLUE = "\033[34m"      # 蓝色 - 通常用于信息
    MAGENTA = "\033[35m"   # 品红
    CYAN = "\033[36m"      # 青色

    # 亮色 (更鲜艳的颜色)
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


# ============================================================
# Agent 核心类
# 负责管理 Agent 的完整生命周期:
# - 初始化 LLM 客户端和工具
# - 维护消息历史
# - 执行主循环 (run 方法)
# - 处理工具调用和结果
# - 上下文自动摘要
# ============================================================
class Agent:
    """Single agent with basic tools and MCP support."""

    def __init__(
        self,
        llm_client: LLMClient,          # LLM 客户端实例,用于调用大模型
        system_prompt: str,             # 系统提示词,定义 Agent 的行为
        tools: list[Tool],              # 可用工具列表
        max_steps: int = 50,            # 最大执行步数,防止无限循环
        workspace_dir: str = "./workspace",  # 工作目录,Agent 在此目录下操作文件
        token_limit: int = 80000,      # Token 限制,超过时触发摘要
        enable_intercept_log: bool = True,  # 是否启用日志记录
    ):
        """初始化 Agent 实例

        参数:
            llm_client: 用于调用 LLM 的客户端
            system_prompt: 系统提示词,定义 Agent 角色和行为
            tools: Agent 可以使用的工具列表
            max_steps: 最大执行步数,默认 50 步
            workspace_dir: 工作目录路径
            token_limit: token 数量限制,超过时触发自动摘要
            enable_intercept_log: 是否启用拦截日志
        """
        # 保存 LLM 客户端
        self.llm = llm_client

        # 将工具列表转换为字典,方便通过名称查找
        # 格式: {"tool_name": Tool实例, ...}
        self.tools = {tool.name: tool for tool in tools}

        # 保存配置参数
        self.max_steps = max_steps
        self.token_limit = token_limit
        self.workspace_dir = Path(workspace_dir)

        # 取消事件:用于外部信号(如按 Esc 键)取消 Agent 执行
        # 这是一个 asyncio.Event,可以通过 set() 方法设置为"已取消"状态
        self.cancel_event: Optional[asyncio.Event] = None

        # 确保工作目录存在,不存在则创建
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # ============================================================
        # 注入工作目录信息到系统提示词
        # 如果系统提示词中没有 "Current Workspace",则追加相关信息
        # 这样 Agent 就知道在哪个目录下工作
        # ============================================================
        if "Current Workspace" not in system_prompt:
            workspace_info = f"\n\n## Current Workspace\nYou are currently working in: `{self.workspace_dir.absolute()}`\nAll relative paths will be resolved relative to this directory."
            system_prompt = system_prompt + workspace_info

        self.system_prompt = system_prompt

        # ============================================================
        # 初始化消息历史
        # 消息历史是对话的核心,包含:
        # - system: 系统提示词 (第一条消息)
        # - user: 用户消息
        # - assistant: LLM 响应
        # - tool: 工具执行结果
        # ============================================================
        self.messages: list[Message] = [Message(role="system", content=system_prompt)]

        # 初始化日志记录器
        self.logger = AgentLogger(enabled=enable_intercept_log)

        # ============================================================
        # Token 使用统计
        # api_total_tokens: LLM API 返回的 token 使用量
        # _skip_next_token_check: 标志位,避免连续触发摘要
        #   - 刚完成摘要后,需要跳过下一次 token 检查
        #   - 因为 api_total_tokens 需要在下次 LLM 调用后才能更新
        # ============================================================
        self.api_total_tokens: int = 0  # API 返回的 token 总数
        self._skip_next_token_check: bool = False  # 跳过下次 token 检查

    # ============================================================
    # 公共方法
    # ============================================================

    def add_user_message(self, content: str):
        """添加用户消息到历史记录

        参数:
            content: 用户输入的内容
        """
        self.messages.append(Message(role="user", content=content))

    # ============================================================
    # 取消机制相关方法
    # ============================================================

    def _check_cancelled(self) -> bool:
        """检查 Agent 是否已被取消执行

        当用户按下 Esc 键或其他取消信号时,cancel_event 会被设置为"已触发"状态,
        此时 Agent 会在下一个安全检查点停止执行。

        返回:
            True 表示已取消, False 表示继续执行
        """
        # 检查 cancel_event 是否存在且已被设置
        if self.cancel_event is not None and self.cancel_event.is_set():
            return True
        return False

    def _cleanup_incomplete_messages(self):
        """清理未完成的消息,确保消息历史一致性

        当 Agent 被取消时,可能会有部分工具调用结果还未返回,
        此时需要清理这些不完整的消息,保持消息历史的完整性。

        清理策略:
        - 只删除当前步骤中未完成的消息
        - 保留已完成步骤的所有消息
        """
        # 找到最后一个 assistant 消息的索引
        last_assistant_idx = -1
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].role == "assistant":
                last_assistant_idx = i
                break

        # 如果没有 assistant 消息,无需清理
        if last_assistant_idx == -1:
            return

        # 删除最后一个 assistant 消息及其之后的所有消息
        removed_count = len(self.messages) - last_assistant_idx
        if removed_count > 0:
            self.messages = self.messages[:last_assistant_idx]
            print(f"{Colors.DIM}   Cleaned up {removed_count} incomplete message(s){Colors.RESET}")

    # ============================================================
    # Token 估算方法
    # 用于计算消息历史的 token 数量,以便在超过限制时触发摘要
    # ============================================================

    def _estimate_tokens(self) -> int:
        """使用 tiktoken 估算消息历史的 token 数量

        使用 cl100k_base 编码器 (GPT-4/Claude/MiniMax M2 兼容)

        估算内容包括:
        - 消息文本内容
        - thinking (思考过程)
        - tool_calls (工具调用)
        - 每条消息的元数据开销 (约 4 tokens)

        返回:
            估算的 token 总数
        """
        try:
            # 使用 cl100k_base 编码器 (GPT-4 和大多数现代模型使用)
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # 如果 tiktoken 初始化失败,使用备用估算方法
            return self._estimate_tokens_fallback()

        total_tokens = 0

        # 遍历所有消息,计算每部分的 token
        for msg in self.messages:
            # 1. 计算文本内容
            if isinstance(msg.content, str):
                total_tokens += len(encoding.encode(msg.content))
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict):
                        # 将字典转为字符串计算
                        total_tokens += len(encoding.encode(str(block)))

            # 2. 计算 thinking (思考过程)
            if msg.thinking:
                total_tokens += len(encoding.encode(msg.thinking))

            # 3. 计算 tool_calls (工具调用)
            if msg.tool_calls:
                total_tokens += len(encoding.encode(str(msg.tool_calls)))

            # 4. 每条消息的元数据开销 (约 4 tokens)
            total_tokens += 4

        return total_tokens

    def _estimate_tokens_fallback(self) -> int:
        """备用 token 估算方法

        当 tiktoken 不可用时使用此方法。
        估算逻辑:平均 2.5 个字符 ≈ 1 个 token

        返回:
            估算的 token 总数
        """
        total_chars = 0
        for msg in self.messages:
            if isinstance(msg.content, str):
                total_chars += len(msg.content)
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict):
                        total_chars += len(str(block))

            if msg.thinking:
                total_chars += len(msg.thinking)

            if msg.tool_calls:
                total_chars += len(str(msg.tool_calls))

        # 粗略估算:平均 2.5 个字符 = 1 个 token
        return int(total_chars / 2.5)

    # ============================================================
    # 消息历史摘要方法
    # 当 token 数量超过限制时,自动压缩消息历史,保留关键信息
    # ============================================================

    async def _summarize_messages(self):
        """消息历史自动摘要

        当 token 数量超过限制时,将用户消息之间的执行过程压缩为摘要,
        以便继续对话而不超出上下文限制。

        摘要策略 (Agent 模式):
        1. 保留所有用户消息 (因为它们代表用户意图)
        2. 将每个用户消息之后的执行过程压缩为摘要
        3. 如果最后一个用户消息还在执行中 (有 assistant/tool 消息但没有下一个 user),
           也将其执行过程压缩为摘要

        压缩后的消息结构:
        system -> user1 -> summary1 -> user2 -> summary2 -> user3 -> summary3 (执行中)

        触发条件 (满足任一即触发):
        - 本地估算的 token 数量超过限制
        - API 返回的 total_tokens 超过限制
        """
        # 如果刚完成摘要,跳过此次检查,等待下次 LLM 调用后 api_total_tokens 更新
        if self._skip_next_token_check:
            self._skip_next_token_check = False
            return

        # 估算当前 token 数量
        estimated_tokens = self._estimate_tokens()

        # 检查是否需要触发摘要 (本地估算或 API 返回任一超过限制)
        should_summarize = estimated_tokens > self.token_limit or self.api_total_tokens > self.token_limit

        # 如果都没超过限制,无需摘要
        if not should_summarize:
            return

        # 打印摘要开始信息
        print(
            f"\n{Colors.BRIGHT_YELLOW}📊 Token usage - Local estimate: {estimated_tokens}, API reported: {self.api_total_tokens}, Limit: {self.token_limit}{Colors.RESET}"
        )
        print(f"{Colors.BRIGHT_YELLOW}🔄 Triggering message history summarization...{Colors.RESET}")

        # 记录日志事件
        self.logger.log_intercept_event(
            "summary_start",
            {
                "estimated_tokens": estimated_tokens,
                "api_total_tokens": self.api_total_tokens,
                "token_limit": self.token_limit,
                "message_count_before": len(self.messages),
            },
        )

        # 找出所有用户消息的索引 (跳过 system 提示词)
        user_indices = [i for i, msg in enumerate(self.messages) if msg.role == "user" and i > 0]

        # 至少需要 1 条用户消息才能进行摘要
        if len(user_indices) < 1:
            print(f"{Colors.BRIGHT_YELLOW}⚠️  Insufficient messages, cannot summarize{Colors.RESET}")
            return

        # 构建新的消息列表
        new_messages = [self.messages[0]]  # 保留 system 提示词
        summary_count = 0

        # 遍历每个用户消息,将其后的执行过程压缩为摘要
        for i, user_idx in enumerate(user_indices):
            # 添加当前用户消息
            new_messages.append(self.messages[user_idx])

            # 确定需要摘要的消息范围
            # 如果不是最后一个用户,摘要到下一个用户之前
            # 如果是最后一个用户,摘要到消息列表末尾
            if i < len(user_indices) - 1:
                next_user_idx = user_indices[i + 1]
            else:
                next_user_idx = len(self.messages)

            # 提取本轮执行的消息
            execution_messages = self.messages[user_idx + 1 : next_user_idx]

            # 如果有执行消息,对其进行摘要
            if execution_messages:
                summary_text = await self._create_summary(execution_messages, i + 1)
                if summary_text:
                    # 将摘要作为用户消息添加 (便于 LLM 理解)
                    summary_message = Message(
                        role="user",
                        content=f"[Assistant Execution Summary]\n\n{summary_text}",
                    )
                    new_messages.append(summary_message)
                    summary_count += 1

        # 替换消息列表
        self.messages = new_messages

        # 跳过下次 token 检查,避免连续触发摘要
        # (因为 api_total_tokens 需要在下次 LLM 调用后才能更新)
        self._skip_next_token_check = True

        # 记录摘要完成日志
        new_tokens = self._estimate_tokens()
        self.logger.log_intercept_event(
            "summary_end",
            {
                "message_count_after": len(self.messages),
                "summary_count": summary_count,
                "estimated_tokens_after": new_tokens,
            },
        )

        # 打印摘要完成信息
        print(f"{Colors.BRIGHT_GREEN}✓ Summary completed, local tokens: {estimated_tokens} → {new_tokens}{Colors.RESET}")
        print(f"{Colors.DIM}  Structure: system + {len(user_indices)} user messages + {summary_count} summaries{Colors.RESET}")
        print(f"{Colors.DIM}  Note: API token count will update on next LLM call{Colors.RESET}")

    async def _create_summary(self, messages: list[Message], round_num: int) -> str:
        """为单轮执行创建摘要

        将一组消息 (assistant 和 tool 消息) 压缩为简洁的文本摘要。

        参数:
            messages: 需要摘要的消息列表
            round_num: 轮次编号 (从 1 开始)

        返回:
            摘要文本
        """
        if not messages:
            return ""

        # ============================================================
        # 构建待摘要的内容
        # 格式:
        #   Round N execution process:
        #
        #   Assistant: [响应内容]
        #     → Called tools: [tool1, tool2, ...]
        #   ← Tool returned: [结果预览]...
        # ============================================================
        summary_content = f"Round {round_num} execution process:\n\n"
        for msg in messages:
            if msg.role == "assistant":
                content_text = msg.content if isinstance(msg.content, str) else str(msg.content)
                summary_content += f"Assistant: {content_text}\n"
                if msg.tool_calls:
                    tool_names = [tc.function.name for tc in msg.tool_calls]
                    summary_content += f"  → Called tools: {', '.join(tool_names)}\n"
            elif msg.role == "tool":
                result_preview = msg.content if isinstance(msg.content, str) else str(msg.content)
                summary_content += f"  ← Tool returned: {result_preview}...\n"

        # ============================================================
        # 调用 LLM 生成简洁摘要
        # 使用专门的提示词让 LLM 聚焦于:
        # - 完成的任务
        # - 调用的工具
        # - 关键结果
        # ============================================================
        try:
            summary_prompt = f"""Please provide a concise summary of the following Agent execution process:

{summary_content}

Requirements:
1. Focus on what tasks were completed and which tools were called
2. Keep key execution results and important findings
3. Be concise and clear, within 1000 words
4. Use English
5. Do not include "user" related content, only summarize the Agent's execution process"""

            summary_msg = Message(role="user", content=summary_prompt)
            response = await self.llm.generate(
                messages=[
                    Message(
                        role="system",
                        content="You are an assistant skilled at summarizing Agent execution processes.",
                    ),
                    summary_msg,
                ]
            )

            summary_text = response.content
            print(f"{Colors.BRIGHT_GREEN}✓ Summary for round {round_num} generated successfully{Colors.RESET}")
            return summary_text

        except Exception as e:
            print(f"{Colors.BRIGHT_RED}✗ Summary generation failed for round {round_num}: {e}{Colors.RESET}")
            # 如果摘要失败,使用原始内容
            return summary_content

    # ============================================================
    # Agent 主执行循环
    # 这是 Agent 的核心方法,实现了完整的 ReAct (Reasoning + Acting) 循环
    # ============================================================

    async def run(self, cancel_event: Optional[asyncio.Event] = None) -> str:
        """执行 Agent 主循环

        这是 Agent 的核心执行方法,实现了完整的执行流程:
        1. 检查是否需要摘要 (防止上下文溢出)
        2. 调用 LLM 获取响应
        3. 如果有工具调用,执行工具并返回结果
        4. 重复步骤 1-3,直到任务完成或达到最大步数

        参数:
            cancel_event: 可选的 asyncio.Event,用于取消执行
                          当设置为"已触发"状态时,Agent 会在下一个安全检查点停止

        返回:
            最终响应内容,或错误消息 (包括取消消息)
        """
        # 设置取消事件 (也可以在调用 run() 之前通过 self.cancel_event 设置)
        if cancel_event is not None:
            self.cancel_event = cancel_event

        # 开始新的运行,初始化日志文件
        self.logger.start_new_run()
        if self.logger.enabled:
            print(f"{Colors.DIM}📝 Log file: {self.logger.get_log_file_path()}{Colors.RESET}")
            print(f"{Colors.DIM}🧩 Intercept log: {self.logger.get_intercept_log_file_path()}{Colors.RESET}")

        # 初始化步数和计时
        step = 0
        run_start_time = perf_counter()

        # ============================================================
        # 主循环:持续执行直到任务完成或达到最大步数
        # ============================================================
        while step < self.max_steps:
            # 1. 检查是否已取消
            if self._check_cancelled():
                self._cleanup_incomplete_messages()
                cancel_msg = "Task cancelled by user."
                print(f"\n{Colors.BRIGHT_YELLOW}⚠️  {cancel_msg}{Colors.RESET}")
                return cancel_msg

            step_start_time = perf_counter()

            # 2. 检查并摘要消息历史 (防止上下文溢出)
            await self._summarize_messages()

            # 3. 打印步骤头部 (带边框的进度提示)
            BOX_WIDTH = 58
            step_text = f"{Colors.BOLD}{Colors.BRIGHT_CYAN}💭 Step {step + 1}/{self.max_steps}{Colors.RESET}"
            step_display_width = calculate_display_width(step_text)
            padding = max(0, BOX_WIDTH - 1 - step_display_width)  # -1 for leading space

            print(f"\n{Colors.DIM}╭{'─' * BOX_WIDTH}╮{Colors.RESET}")
            print(f"{Colors.DIM}│{Colors.RESET} {step_text}{' ' * padding}{Colors.DIM}│{Colors.RESET}")
            print(f"{Colors.DIM}╰{'─' * BOX_WIDTH}╯{Colors.RESET}")

            # 4. 获取可用工具列表
            tool_list = list(self.tools.values())

            # 记录发送前的事件日志
            estimated_tokens = self._estimate_tokens()
            self.logger.log_intercept_event(
                "before_send",
                {
                    "step": step + 1,
                    "message_count": len(self.messages),
                    "estimated_tokens": estimated_tokens,
                    "api_total_tokens": self.api_total_tokens,
                    "tool_count": len(tool_list),
                    "tools": [tool.name for tool in tool_list],
                },
            )

            # 5. 记录 LLM 请求并调用 LLM
            self.logger.log_request(messages=self.messages, tools=tool_list)

            try:
                # 调用 LLM 获取响应
                # 传入:消息历史 + 可用工具列表
                response = await self.llm.generate(messages=self.messages, tools=tool_list)
            except Exception as e:
                # 处理 LLM 调用错误
                from .retry import RetryExhaustedError

                if isinstance(e, RetryExhaustedError):
                    # 重试次数耗尽
                    error_msg = f"LLM call failed after {e.attempts} retries\nLast error: {str(e.last_exception)}"
                    print(f"\n{Colors.BRIGHT_RED}❌ Retry failed:{Colors.RESET} {error_msg}")
                else:
                    # 其他错误
                    error_msg = f"LLM call failed: {str(e)}"
                    print(f"\n{Colors.BRIGHT_RED}❌ Error:{Colors.RESET} {error_msg}")
                return error_msg

            # 6. 更新 API 返回的 token 使用量
            if response.usage:
                self.api_total_tokens = response.usage.total_tokens

            # 记录响应事件日志
            self.logger.log_intercept_event(
                "after_response",
                {
                    "step": step + 1,
                    "finish_reason": response.finish_reason,
                    "tool_call_count": len(response.tool_calls) if response.tool_calls else 0,
                    "tool_calls": [tc.function.name for tc in response.tool_calls] if response.tool_calls else [],
                    "content_chars": len(response.content) if response.content else 0,
                    "thinking_chars": len(response.thinking) if response.thinking else 0,
                    "usage": response.usage.model_dump() if response.usage else None,
                },
            )

            # 7. 记录 LLM 响应日志
            self.logger.log_response(
                content=response.content,
                thinking=response.thinking,
                tool_calls=response.tool_calls,
                finish_reason=response.finish_reason,
            )

            # 8. 将 assistant 响应添加到消息历史
            assistant_msg = Message(
                role="assistant",
                content=response.content,
                thinking=response.thinking,
                tool_calls=response.tool_calls,
            )
            self.messages.append(assistant_msg)

            # 9. 打印 thinking (思考过程) - 如果有的话
            if response.thinking:
                print(f"\n{Colors.BOLD}{Colors.MAGENTA}🧠 Thinking:{Colors.RESET}")
                print(f"{Colors.DIM}{response.thinking}{Colors.RESET}")

            # 10. 打印 assistant 响应内容
            if response.content:
                print(f"\n{Colors.BOLD}{Colors.BRIGHT_BLUE}🤖 Assistant:{Colors.RESET}")
                print(f"{response.content}")

            # ============================================================
            # 11. 检查任务是否完成
            # 如果没有工具调用,说明任务已完成,返回响应内容
            # ============================================================
            if not response.tool_calls:
                step_elapsed = perf_counter() - step_start_time
                total_elapsed = perf_counter() - run_start_time
                print(f"\n{Colors.DIM}⏱️  Step {step + 1} completed in {step_elapsed:.2f}s (total: {total_elapsed:.2f}s){Colors.RESET}")
                return response.content

            # 12. 执行工具调用前的取消检查
            if self._check_cancelled():
                self._cleanup_incomplete_messages()
                cancel_msg = "Task cancelled by user."
                print(f"\n{Colors.BRIGHT_YELLOW}⚠️  {cancel_msg}{Colors.RESET}")
                return cancel_msg

            # ============================================================
            # 13. 执行工具调用
            # 遍历响应中的所有工具调用,依次执行并返回结果
            # ============================================================
            for tool_call in response.tool_calls:
                tool_call_id = tool_call.id          # 工具调用 ID (用于关联结果)
                function_name = tool_call.function.name  # 工具名称
                arguments = tool_call.function.arguments  # 工具参数

                # 打印工具调用头部
                print(f"\n{Colors.BRIGHT_YELLOW}🔧 Tool Call:{Colors.RESET} {Colors.BOLD}{Colors.CYAN}{function_name}{Colors.RESET}")

                # 打印参数 (格式化显示,过长内容截断)
                print(f"{Colors.DIM}   Arguments:{Colors.RESET}")
                truncated_args = {}
                for key, value in arguments.items():
                    value_str = str(value)
                    if len(value_str) > 200:
                        truncated_args[key] = value_str[:200] + "..."
                    else:
                        truncated_args[key] = value
                args_json = json.dumps(truncated_args, indent=2, ensure_ascii=False)
                for line in args_json.split("\n"):
                    print(f"   {Colors.DIM}{line}{Colors.RESET}")

                # 记录工具执行前的事件日志
                self.logger.log_intercept_event(
                    "before_tool",
                    {
                        "step": step + 1,
                        "tool_call_id": tool_call_id,
                        "tool_name": function_name,
                        "argument_keys": list(arguments.keys()),
                    },
                )

                # 查找并执行工具
                if function_name not in self.tools:
                    # 工具不存在
                    result = ToolResult(
                        success=False,
                        content="",
                        error=f"Unknown tool: {function_name}",
                    )
                else:
                    try:
                        # 获取工具实例并执行
                        tool = self.tools[function_name]
                        result = await tool.execute(**arguments)
                    except Exception as e:
                        # 工具执行过程中发生异常,捕获并转换为失败的 ToolResult
                        import traceback

                        error_detail = f"{type(e).__name__}: {str(e)}"
                        error_trace = traceback.format_exc()
                        result = ToolResult(
                            success=False,
                            content="",
                            error=f"Tool execution failed: {error_detail}\n\nTraceback:\n{error_trace}",
                        )

                # 记录工具执行后的事件日志
                self.logger.log_intercept_event(
                    "after_tool",
                    {
                        "step": step + 1,
                        "tool_call_id": tool_call_id,
                        "tool_name": function_name,
                        "success": result.success,
                        "result_chars": len(result.content) if result.content else 0,
                        "error_chars": len(result.error) if result.error else 0,
                    },
                )

                # 记录工具执行结果日志
                self.logger.log_tool_result(
                    tool_name=function_name,
                    arguments=arguments,
                    result_success=result.success,
                    result_content=result.content if result.success else None,
                    result_error=result.error if not result.success else None,
                )

                # 打印执行结果
                if result.success:
                    result_text = result.content
                    if len(result_text) > 300:
                        result_text = result_text[:300] + f"{Colors.DIM}...{Colors.RESET}"
                    print(f"{Colors.BRIGHT_GREEN}✓ Result:{Colors.RESET} {result_text}")
                else:
                    print(f"{Colors.BRIGHT_RED}✗ Error:{Colors.RESET} {Colors.RED}{result.error}{Colors.RESET}")

                # 14. 将工具结果作为消息添加到历史
                tool_msg = Message(
                    role="tool",
                    content=result.content if result.success else f"Error: {result.error}",
                    tool_call_id=tool_call_id,
                    name=function_name,
                )
                self.messages.append(tool_msg)

                # 15. 每个工具执行完成后检查是否取消
                if self._check_cancelled():
                    self._cleanup_incomplete_messages()
                    cancel_msg = "Task cancelled by user."
                    print(f"\n{Colors.BRIGHT_YELLOW}⚠️  {cancel_msg}{Colors.RESET}")
                    return cancel_msg

            # 打印本步完成信息
            step_elapsed = perf_counter() - step_start_time
            total_elapsed = perf_counter() - run_start_time
            print(f"\n{Colors.DIM}⏱️  Step {step + 1} completed in {step_elapsed:.2f}s (total: {total_elapsed:.2f}s){Colors.RESET}")

            # 步数 +1,继续循环
            step += 1

        # ============================================================
        # 达到最大步数,任务未能完成
        # ============================================================
        error_msg = f"Task couldn't be completed after {self.max_steps} steps."
        print(f"\n{Colors.BRIGHT_YELLOW}⚠️  {error_msg}{Colors.RESET}")
        return error_msg

    # ============================================================
    # 公共方法:获取历史记录
    # ============================================================

    def get_history(self) -> list[Message]:
        """获取消息历史记录

        返回:
            消息列表的副本
        """
        return self.messages.copy()
