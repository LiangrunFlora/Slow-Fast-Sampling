#!/bin/bash

# 定义要执行的命令
COMMANDS=(
    "pip install -r requirements.txt"
    "huggingface-cli download --resume-download GSAI-ML/LLaDA-8B-Instruct"
    "huggingface-cli download --resume-download GSAI-ML/LLaDA-8B-Base"
    "huggingface-cli download --resume-download Dream-org/Dream-v0-Instruct-7B"
    "huggingface-cli download --resume-download Dream-org/Dream-v0-Base-7B"
    "huggingface-cli download --resume-download Qwen/Qwen2.5-7B-Instruct"
    "huggingface-cli download --resume-download Qwen/Qwen2.5-7B"
    "huggingface-cli download --resume-download deepseek-ai/deepseek-llm-7b-base"
    "huggingface-cli download --resume-download meta-llama/Meta-Llama-3-8B-Instruct"
    "huggingface-cli download --resume-download meta-llama/Meta-Llama-3-8B"

)

# 日志文件
LOG_FILE="download_log.txt"

# 重试间隔（秒）
RETRY_INTERVAL=6

# 最大重试次数（0 表示无限重试）
MAX_RETRIES=0

# 循环间隔（命令序列完成后等待时间，秒）
LOOP_INTERVAL=30

# 记录日志的函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 检查命令是否成功的函数
run_command() {
    local cmd="$1"
    local attempt=1

    while true; do
        log "尝试执行命令（第 $attempt 次）：$cmd"
        # 执行命令并捕获返回码
        $cmd >> "$LOG_FILE" 2>&1
        local exit_code=$?

        if [ $exit_code -eq 0 ]; then
            log "命令执行成功：$cmd"
            return 0
        else
            log "命令失败，返回码：$exit_code"
            # 检查是否达到最大重试次数
            if [ $MAX_RETRIES -ne 0 ] && [ $attempt -ge $MAX_RETRIES ]; then
                log "达到最大重试次数 ($MAX_RETRIES)，放弃命令：$cmd"
                return 1
            fi
            log "将在 $RETRY_INTERVAL 秒后重试..."
            sleep $RETRY_INTERVAL
            ((attempt++))
        fi
    done
}

# 主循环
log "开始执行模型下载脚本"

while true; do
    for cmd in "${COMMANDS[@]}"; do
        run_command "$cmd"
        if [ $? -ne 0 ]; then
            log "命令 $cmd 失败，脚本终止"
            exit 1
        fi
    done
    log "本次命令序列执行完成，将在 $LOOP_INTERVAL 秒后重新开始..."
    sleep $LOOP_INTERVAL
done

log "脚本意外终止"
exit 0