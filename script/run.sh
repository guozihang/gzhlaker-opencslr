#!/usr/bin/env bash
# OpenSLR 实验启动脚本
#
# 用法:
#   bash script/run.sh <实验名> [额外 main.py 参数]
#
# 示例:
#   bash script/run.sh baseline
#   bash script/run.sh corrnet --device 1 --batch-size 1
#
# 可用实验: baseline / tlp / vac / corrnet (core/configs/exp.yaml)
# 未传 --work-dir 时按实验名使用下方默认目录(注意必须以 / 结尾);
# 其余参数原样透传给 main.py(如 --device、--batch-size、--load-weights)。

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="$SCRIPT_DIR/../core"

EXP="${1:-baseline}"
shift || true

# 各实验默认 work_dir(config 中 baseline 的 /share 路径不可用,统一在此覆盖)
declare -A DEFAULT_WORK_DIR=(
    [baseline]="/sda/data/guozihang/work_dir/www2026_test_1/"
    [tlp]="/sda/data/guozihang/work_dir/tlp/"
    [vac]="/sda/data/guozihang/work_dir/vac_smkd/"
    [corrnet]="/sda/data/guozihang/work_dir/baseline_res18/"
)

if ! printf '%s\n' "$@" | grep -q -- '--work-dir'; then
    set -- --work-dir "${DEFAULT_WORK_DIR[$EXP]:-./work_dir/$EXP/}" "$@"
fi

cd "$CORE_DIR"
python main.py --config configs/exp.yaml --exp "$EXP" "$@"

# 多卡 DDP 参考(单卡 24GB 跑不动 SlowFast batch 2):
#   cd core && CUDA_VISIBLE_DEVICES=1,4 python -m torch.distributed.run \
#     --nproc_per_node=2 --master_port=29517 main.py \
#     --config configs/exp.yaml --exp baseline --device 1,4 \
#     --batch-size 1 --test-batch-size 1 \
#     --work-dir /sda/data/guozihang/work_dir/www2026_test_1/
#
# 旧指令格式(路径式模型写法仍兼容,模型/数据集也可从配置读):
#   CUDA_VISIBLE_DEVICES=0 python main.py --device 0 \
#     --model models.build_function.build_slowfast \
#     --work-dir /sda/data/renjiawei/slowfast/csl/ --dataset CSL-Daily
