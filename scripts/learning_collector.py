#!/usr/bin/env python3
"""
A 股知识学习搜集器
定期搜集和整理 A 股相关知识，生成学习文档
"""

import asyncio
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

# 配置
SCRIPT_DIR = Path(__file__).parent
DOC_DIR = SCRIPT_DIR.parent / "doc"
MEMORY_DB = SCRIPT_DIR.parent / ".agent_memory.db"

# 创建基础目录
DOC_DIR.mkdir(parents=True, exist_ok=True)
(DOC_DIR / "基础知识").mkdir(exist_ok=True)
(DOC_DIR / "技术分析").mkdir(exist_ok=True)
(DOC_DIR / "基本面").mkdir(exist_ok=True)
(DOC_DIR / "策略").mkdir(exist_ok=True)
(DOC_DIR / "实战案例").mkdir(exist_ok=True)

# 学习主题
LEARNING_TOPICS = [
    {
        "topic": "K线基础",
        "prompt": "详细讲解A股K线的基础知识，包括阳线、阴线、上影线、下影线等的含义和作用",
    },
    {
        "topic": "均线系统",
        "prompt": "讲解5日、10日、20日、60日、120日、250日均线的含义和使用技巧",
    },
    {
        "topic": "MACD指标",
        "prompt": "详细讲解MACD指标的原理、参数设置、金叉死叉判断方法",
    },
    {
        "topic": "KDJ指标",
        "prompt": "讲解KDJ随机指标的含义、超买超卖判断、实战应用技巧",
    },
    {
        "topic": "成交量分析",
        "prompt": "讲解成交量的重要性，放量、缩量的判断方法，与价格的关系",
    },
    {
        "topic": "趋势判断",
        "prompt": "讲解如何判断A股趋势，上升趋势、下降趋势、横盘震荡的识别方法",
    },
    {
        "topic": "支撑位与阻力位",
        "prompt": "讲解支撑位和阻力位的概念、画法、实战应用",
    },
    {
        "topic": "波浪理论",
        "prompt": "简要讲解艾略特波浪理论的基本原则和数浪方法",
    },
    {
        "topic": "基本面分析框架",
        "prompt": "讲解A股基本面分析框架，包括行业分析、公司分析、财务指标解读",
    },
    {
        "topic": "估值方法",
        "prompt": "讲解PE、PB、DCF等估值方法的优缺点和适用场景",
    },
    {
        "topic": "ROE与盈利能力",
        "prompt": "讲解ROE净资产收益率的含义、高ROE股票的选择标准",
    },
    {
        "topic": "股息率分析",
        "prompt": "讲解股息率的意义，高股息股票的筛选方法和投资逻辑",
    },
    {
        "topic": "成长股投资",
        "prompt": "讲解成长股的识别方法，营收增长、净利润增长的判断标准",
    },
    {
        "topic": "行业轮动策略",
        "prompt": "讲解A股行业轮动的规律和投资策略",
    },
    {
        "topic": "龙头股筛选",
        "prompt": "讲解如何识别和筛选行业龙头股",
    },
    {
        "topic": "仓位管理",
        "prompt": "讲解股票投资中的仓位管理方法，分批建仓、止损技巧",
    },
    {
        "topic": "风险控制",
        "prompt": "讲解A股投资中的风险识别和控制方法",
    },
    {
        "topic": "基金定投",
        "prompt": "讲解基金定投的原理、优势和注意事项",
    },
    {
        "topic": "打新攻略",
        "prompt": "讲解A股打新股的规则、技巧和注意事项",
    },
    {
        "topic": "指数基金",
        "prompt": "讲解沪深300、中证500等指数基金的投资价值",
    },
]


async def generate_lesson(topic: str, prompt: str) -> str:
    """使用LLM生成课程内容"""
    import yaml
    from mini_agent.llm.llm_wrapper import LLMClient
    
    # 从配置文件读取 API key
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    api_key = config.get("api_key", "")
    api_base = config.get("api_base", "https://api.minimaxi.com")
    model = config.get("model", "MiniMax-M2.5")
    
    llm = LLMClient(api_key=api_key, api_base=api_base, model=model)
    
    full_prompt = f"""你是一位资深的A股投资培训师。请用通俗易懂的语言，为投资新手编写一份详细的 学习教程。

要求：
1. 语言通俗易懂，适合小白学习
2. 包含具体例子和实战应用
3. 重点内容用加粗标注
4. 最后有练习题或思考题

主题：{topic}

{prompt}

请生成一份完整的学习教程。"""

    try:
        from mini_agent.schema import Message
        messages = [Message(role="user", content=full_prompt)]
        response = await llm.generate(messages)
        return response.content
    except Exception as e:
        return f"生成失败: {e}"


def get_learning_progress() -> dict:
    """获取学习进度"""
    if not MEMORY_DB.exists():
        return {"total": len(LEARNING_TOPICS), "completed": 0, "topics": []}
    
    conn = sqlite3.connect(str(MEMORY_DB))
    cursor = conn.cursor()
    
    # 创建表如果不存在
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS learning_progress (
            id INTEGER PRIMARY KEY,
            topic TEXT UNIQUE,
            created_at TEXT
        )
    """)
    
    # 查询已完成的 topic
    cursor.execute("SELECT topic FROM learning_progress")
    completed = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    
    return {
        "total": len(LEARNING_TOPICS),
        "completed": len(completed),
        "topics": completed
    }


def save_progress(topic: str):
    """保存学习进度"""
    from datetime import datetime
    now = datetime.now().isoformat()
    
    conn = sqlite3.connect(str(MEMORY_DB))
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS learning_progress (
            id INTEGER PRIMARY KEY,
            topic TEXT UNIQUE,
            created_at TEXT
        )
    """)
    
    cursor.execute("INSERT OR IGNORE INTO learning_progress (topic, created_at) VALUES (?, ?)", (topic, now))
    conn.commit()
    conn.close()


def save_lesson(topic: str, content: str):
    """保存课程到文件"""
    # 确定分类
    category = "基础知识"
    if any(x in topic for x in ["K线", "均线", "MACD", "KDJ", "成交量", "趋势", "支撑", "阻力", "波浪"]):
        category = "技术分析"
    elif any(x in topic for x in ["基本面", "估值", "ROE", "股息", "成长", "行业", "龙头"]):
        category = "基本面"
    elif any(x in topic for x in ["策略", "轮动", "仓位", "风险", "打新", "定投", "指数"]):
        category = "策略"
    
    # 创建分类目录（使用 parents=True）
    category_dir = DOC_DIR / category
    category_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存文件
    filename = topic.replace(" ", "_") + ".md"
    filepath = category_dir / filename
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {topic}\n\n")
        f.write(f"*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write("---\n\n")
        f.write(content)
    
    print(f"✅ 已保存: {category}/{filename}")
    return filepath


async def run_learning_task():
    """运行学习任务"""
    print("=" * 50)
    print("📚 A股知识学习搜集器启动")
    print("=" * 50)
    
    # 获取进度
    progress = get_learning_progress()
    print(f"\n📊 当前进度: {progress['completed']}/{progress['total']}")
    
    # 找出未完成的 topic
    pending_topics = [t for t in LEARNING_TOPICS if t["topic"] not in progress["topics"]]
    
    if not pending_topics:
        print("\n✅ 所有课程已生成完毕！")
        return
    
    print(f"\n📝 待生成: {len(pending_topics)} 个主题\n")
    
    # 每次生成一个
    topic_data = pending_topics[0]
    topic = topic_data["topic"]
    prompt = topic_data["prompt"]
    
    print(f"🔄 正在生成: {topic}")
    
    content = await generate_lesson(topic, prompt)
    
    if "生成失败" not in content:
        save_lesson(topic, content)
        # 保存进度到数据库
        save_progress(topic)
        print(f"✅ {topic} 完成!")
    else:
        print(f"❌ {topic} 失败: {content}")
    
    print(f"\n📊 更新进度: {progress['completed'] + 1}/{progress['total']}")


if __name__ == "__main__":
    asyncio.run(run_learning_task())
