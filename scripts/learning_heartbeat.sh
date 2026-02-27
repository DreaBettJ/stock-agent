#!/bin/bash
# A股知识学习 Heartbeat
# 每2小时运行一次学习搜集器

LOG_FILE="/tmp/learning_heartbeat.log"

echo "$(date '+%Y-%m-%d %H:%M:%S') 运行学习搜集器..." >> $LOG_FILE

cd ~/workspace/Mini-Agent
source .venv/bin/activate

python scripts/learning_collector.py >> $LOG_FILE 2>&1

echo "$(date '+%Y-%m-%d %H:%M:%S') 完成" >> $LOG_FILE
