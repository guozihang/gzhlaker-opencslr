#!/usr/bin/env bash
# 训练看门狗:训练进程中断/崩溃后自动加载最新 checkpoint 续跑,直到跑满 num_epoch。
#
# 用法:
#   bash script/train_watchdog.sh <实验名> <work_dir> [额外 main.py 参数]
#
# 示例:
#   bash script/train_watchdog.sh vac /sda/data/guozihang/work_dir/vac_ph_20260901/ \
#       --device 1 --decode-mode max
#
# 说明:
#   - 每次重启自动查找 work_dir 下 epoch 最大的 dev_*_epochN_model.pt 作为
#     --load-checkpoints 续跑(恢复 optimizer/scheduler/随机数状态,start_epoch=N+1);
#   - 训练正常跑满 num_epoch 后(日志出现 "Training costs")自动退出;
#   - 连续失败超过 30 次停止重启,日志写入 <work_dir>/train.log。

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="$SCRIPT_DIR/../core"
EXP="${1:?用法: train_watchdog.sh <exp> <work_dir> [额外 main.py 参数]}"
WORK_DIR="${2:?用法: train_watchdog.sh <exp> <work_dir> [额外 main.py 参数]}"
shift 2

mkdir -p "$WORK_DIR"
LOG="$WORK_DIR/train.log"

# 找出 epoch 最大的周期 checkpoint(文件名形如 dev_XX.XX_epochN_model.pt)
latest_ckpt() {
    local f epoch latest="" latest_epoch=-1
    for f in "$WORK_DIR"dev_*_epoch*_model.pt; do
        [ -f "$f" ] || continue
        epoch=$(basename "$f" | sed -E 's/.*_epoch([0-9]+)_model\.pt/\1/')
        if [ "$epoch" -gt "$latest_epoch" ] 2>/dev/null; then
            latest_epoch=$epoch
            latest=$f
        fi
    done
    echo "$latest"
}

FAILS=0
while true; do
    LATEST=$(latest_ckpt)
    CMD="python main.py --config configs/exp.yaml --exp $EXP --work-dir $WORK_DIR"
    if [ -n "$LATEST" ]; then
        CMD="$CMD --load-checkpoints $LATEST"
        echo "[$(date '+%F %T')] 续跑: checkpoint=$LATEST" | tee -a "$LOG"
    fi
    echo "[$(date '+%F %T')] 启动: $CMD $*" | tee -a "$LOG"

    if (cd "$CORE_DIR" && $CMD "$@") >> "$LOG" 2>&1; then
        if grep -q "Training costs" "$LOG"; then
            echo "[$(date '+%F %T')] 训练正常完成,看门狗退出。" | tee -a "$LOG"
            break
        fi
    fi

    FAILS=$((FAILS + 1))
    if [ "$FAILS" -ge 30 ]; then
        echo "[$(date '+%F %T')] 连续失败 $FAILS 次,停止重启,请检查 $LOG" | tee -a "$LOG"
        exit 1
    fi
    echo "[$(date '+%F %T')] 训练进程退出,第 $FAILS 次重启,10 秒后续跑..." | tee -a "$LOG"
    sleep 10
done
