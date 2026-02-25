"""Configuration management module

Provides unified configuration loading and management functionality
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class RetryConfig(BaseModel):
    """Retry configuration"""

    enabled: bool = True
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0


class LLMConfig(BaseModel):
    """LLM configuration"""

    api_key: str
    api_base: str = "https://api.minimax.io"
    model: str = "MiniMax-M2.5"
    provider: str = "anthropic"  # "anthropic" or "openai"
    retry: RetryConfig = Field(default_factory=RetryConfig)


class AgentConfig(BaseModel):
    """Agent configuration"""

    max_steps: int = 50
    workspace_dir: str = "./workspace"
    system_prompt_path: str = "system_prompt.md"
    enable_intercept_log: bool = True


class MCPConfig(BaseModel):
    """MCP (Model Context Protocol) timeout configuration"""

    connect_timeout: float = 10.0  # Connection timeout (seconds)
    execute_timeout: float = 60.0  # Tool execution timeout (seconds)
    sse_read_timeout: float = 120.0  # SSE read timeout (seconds)


class ToolsConfig(BaseModel):
    """Tools configuration"""

    # Basic tools (file operations, bash)
    enable_file_tools: bool = True
    enable_bash: bool = True
    enable_note: bool = True
    enable_stock_tools: bool = True
    tushare_token: str = ""

    # Skills
    enable_skills: bool = True
    skills_dir: str = "./skills"

    # MCP tools
    enable_mcp: bool = True
    mcp_config_path: str = "mcp.json"
    mcp: MCPConfig = Field(default_factory=MCPConfig)


class SmtpNoticeConfig(BaseModel):
    """SMTP notice channel configuration."""

    enabled: bool = False
    host: str = ""
    port: int = 465
    username: str = ""
    password: str = ""
    use_ssl: bool = True
    use_starttls: bool = False
    from_addr: str = ""
    to_addrs: list[str] = Field(default_factory=list)
    subject_prefix: str = "[Big-A-Helper]"
    timeout_seconds: float = 10.0


class NoticeConfig(BaseModel):
    """Notice channels configuration."""

    smtp: SmtpNoticeConfig = Field(default_factory=SmtpNoticeConfig)


class NoticeLevelConfig(BaseModel):
    """Notice severity switches."""

    p0_trade_realtime: bool = True
    p1_risk_realtime: bool = False
    p2_digest: bool = False


class NoticeRiskConfig(BaseModel):
    """Risk notice trigger thresholds."""

    no_trade_with_veto: bool = True
    consecutive_failures_threshold: int = 3


class Config(BaseModel):
    """Main configuration class"""

    llm: LLMConfig
    agent: AgentConfig
    tools: ToolsConfig
    notice: NoticeConfig = Field(default_factory=NoticeConfig)
    notice_levels: NoticeLevelConfig = Field(default_factory=NoticeLevelConfig)
    notice_risk: NoticeRiskConfig = Field(default_factory=NoticeRiskConfig)

    @classmethod
    def load(cls) -> "Config":
        """Load configuration from the default search path."""
        config_path = cls.get_default_config_path()
        if not config_path.exists():
            raise FileNotFoundError(
                "Configuration file not found. Run scripts/setup-config.sh or place config.yaml in project root or mini_agent/config/."
            )
        return cls.from_yaml(config_path)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "Config":
        """Load configuration from YAML file

        Args:
            config_path: Configuration file path

        Returns:
            Config instance

        Raises:
            FileNotFoundError: Configuration file does not exist
            ValueError: Invalid configuration format or missing required fields
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError("Configuration file is empty")

        # Parse LLM configuration
        if "api_key" not in data:
            raise ValueError("Configuration file missing required field: api_key")

        if not data["api_key"] or data["api_key"] == "YOUR_API_KEY_HERE":
            raise ValueError("Please configure a valid API Key")

        # Parse retry configuration
        retry_data = data.get("retry", {})
        retry_config = RetryConfig(
            enabled=retry_data.get("enabled", True),
            max_retries=retry_data.get("max_retries", 3),
            initial_delay=retry_data.get("initial_delay", 1.0),
            max_delay=retry_data.get("max_delay", 60.0),
            exponential_base=retry_data.get("exponential_base", 2.0),
        )

        llm_config = LLMConfig(
            api_key=data["api_key"],
            api_base=data.get("api_base", "https://api.minimax.io"),
            model=data.get("model", "MiniMax-M2.5"),
            provider=data.get("provider", "anthropic"),
            retry=retry_config,
        )

        # Parse Agent configuration
        agent_config = AgentConfig(
            max_steps=data.get("max_steps", 50),
            workspace_dir=data.get("workspace_dir", "./workspace"),
            system_prompt_path=data.get("system_prompt_path", "system_prompt.md"),
            enable_intercept_log=data.get("enable_intercept_log", True),
        )

        # Parse tools configuration
        tools_data = data.get("tools", {})

        # Parse MCP configuration
        mcp_data = tools_data.get("mcp", {})
        mcp_config = MCPConfig(
            connect_timeout=mcp_data.get("connect_timeout", 10.0),
            execute_timeout=mcp_data.get("execute_timeout", 60.0),
            sse_read_timeout=mcp_data.get("sse_read_timeout", 120.0),
        )

        tools_config = ToolsConfig(
            enable_file_tools=tools_data.get("enable_file_tools", True),
            enable_bash=tools_data.get("enable_bash", True),
            enable_note=tools_data.get("enable_note", True),
            enable_stock_tools=tools_data.get("enable_stock_tools", True),
            tushare_token=str(tools_data.get("tushare_token", "") or ""),
            enable_skills=tools_data.get("enable_skills", True),
            skills_dir=tools_data.get("skills_dir", "./skills"),
            enable_mcp=tools_data.get("enable_mcp", True),
            mcp_config_path=tools_data.get("mcp_config_path", "mcp.json"),
            mcp=mcp_config,
        )

        # Parse notice configuration
        notice_data = data.get("notice", {}) or {}
        smtp_data = notice_data.get("smtp", {}) or {}
        smtp_config = SmtpNoticeConfig(
            enabled=bool(smtp_data.get("enabled", False)),
            host=str(smtp_data.get("host", "") or ""),
            port=int(smtp_data.get("port", 465) or 465),
            username=str(smtp_data.get("username", "") or ""),
            password=str(smtp_data.get("password", "") or ""),
            use_ssl=bool(smtp_data.get("use_ssl", True)),
            use_starttls=bool(smtp_data.get("use_starttls", False)),
            from_addr=str(smtp_data.get("from_addr", "") or ""),
            to_addrs=[str(x).strip() for x in (smtp_data.get("to_addrs", []) or []) if str(x).strip()],
            subject_prefix=str(smtp_data.get("subject_prefix", "[Big-A-Helper]") or "[Big-A-Helper]"),
            timeout_seconds=float(smtp_data.get("timeout_seconds", 10.0) or 10.0),
        )
        notice_config = NoticeConfig(smtp=smtp_config)
        levels_data = notice_data.get("levels", {}) or {}
        notice_levels = NoticeLevelConfig(
            p0_trade_realtime=bool(levels_data.get("p0_trade_realtime", True)),
            p1_risk_realtime=bool(levels_data.get("p1_risk_realtime", False)),
            p2_digest=bool(levels_data.get("p2_digest", False)),
        )
        risk_data = notice_data.get("risk", {}) or {}
        notice_risk = NoticeRiskConfig(
            no_trade_with_veto=bool(risk_data.get("no_trade_with_veto", True)),
            consecutive_failures_threshold=int(risk_data.get("consecutive_failures_threshold", 3) or 3),
        )

        return cls(
            llm=llm_config,
            agent=agent_config,
            tools=tools_config,
            notice=notice_config,
            notice_levels=notice_levels,
            notice_risk=notice_risk,
        )

    @staticmethod
    def get_package_dir() -> Path:
        """Get the package installation directory

        Returns:
            Path to the mini_agent package directory
        """
        # Get the directory where this config.py file is located
        return Path(__file__).parent

    @classmethod
    def find_config_file(cls, filename: str) -> Path | None:
        """Find configuration file with priority order

        Search for config file in the following order of priority:
        1) ./{filename} in current project directory
        2) ./mini_agent/config/{filename} in current directory (development mode)
        3) ~/.mini-agent/config/{filename} in user home directory
        4) {package}/mini_agent/config/{filename} in package installation directory

        Args:
            filename: Configuration file name (e.g., "config.yaml", "mcp.json", "system_prompt.md")

        Returns:
            Path to found config file, or None if not found
        """
        # Priority 1: project root config file
        project_root_config = Path.cwd() / filename
        if project_root_config.exists():
            return project_root_config

        # Priority 2: Development mode - current directory's mini_agent/config/ subdirectory
        dev_config = Path.cwd() / "mini_agent" / "config" / filename
        if dev_config.exists():
            return dev_config

        # Priority 3: User config directory
        user_config = Path.home() / ".mini-agent" / "config" / filename
        if user_config.exists():
            return user_config

        # Priority 4: Package installation directory's config/ subdirectory
        package_config = cls.get_package_dir() / "config" / filename
        if package_config.exists():
            return package_config

        return None

    @classmethod
    def get_default_config_path(cls) -> Path:
        """Get the default config file path with priority search

        Returns:
            Path to config.yaml (prioritizes: project root > dev config/ > user config/ > package config/)
        """
        config_path = cls.find_config_file("config.yaml")
        if config_path:
            return config_path

        # Fallback to package config directory for error message purposes
        return cls.get_package_dir() / "config" / "config.yaml"
