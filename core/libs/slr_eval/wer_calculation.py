"""WER 评估工具:基于仓库内置的 pysclite(纯 Python 移植版 sclite)。

原实现依赖编译安装的 sclite 二进制(``./software/sclite``)。现改为:
  1. 数据清洗(preprocess.sh)与 [EMPTY] 补齐(mergectmstm.py)流程保持不变;
  2. 对齐与统计在进程内直接调用 ``libs.pysclite``,无需任何编译依赖;
  3. 报告输出 sum / rsum / pra 与 sclite 对应;原 sgml / dtl 报告
     pysclite 不支持,已移除,WER 数值改由 SCORES 统计直接计算。
"""

import os
import shutil
import subprocess
import sys

from libs.pysclite import scores as _sc
from libs.pysclite import stmctm as _stmctm


def evaluate(prefix="./", mode="dev", evaluate_dir=None, evaluate_prefix=None,
             output_file=None, output_dir=None):
    """计算指定模式(dev/test)的 WER。

    流程:
      1. preprocess.sh 清洗假设 CTM(去特殊标签、合并重复词);
      2. ground-truth STM 按句 ID 排序(等价 sort -k1,1);
      3. mergectmstm.py 为缺失句补齐 [EMPTY] 行;
      4. pysclite 对齐 CTM 与 STM,输出 sum/rsum/pra 报告并计算 WER。

    Args:
        prefix: 输出路径前缀(通常是 work_dir,以 / 结尾)。
        mode: 评估模式("dev"/"test"),用于选择 ground-truth 文件。
        evaluate_dir: slr_eval 工具目录(含 preprocess.sh、*.stm 等)。
        evaluate_prefix: ground-truth 文件名前缀(如 "phoenix2014-groundtruth")。
        output_file: 模型输出的假设 CTM 文件名。
        output_dir: 可选,报告输出子目录(相对 prefix)。

    Returns:
        str: 形如 "Percent Total Error       =  23.45%   (OpenCSLR)" 的行,
        调用方(single.py)通过 split("=") 与 split("%") 提取数值。
    """
    evaluate_dir = evaluate_dir or "./libs/slr_eval"
    evaluate_prefix = evaluate_prefix or "phoenix2014-groundtruth"
    output_file = output_file or "output-hypothesis-{}.ctm".format(mode)

    hyp_file = prefix + output_file
    tmp_ctm = prefix + "tmp.ctm"
    tmp2_ctm = prefix + "tmp2.ctm"
    stm_src = os.path.join(evaluate_dir, "{}-{}.stm".format(evaluate_prefix, mode))
    tmp_stm = prefix + "tmp.stm"

    # 1. 清洗假设 CTM(与 sclite 版流程一致)
    subprocess.check_call(
        ["bash", os.path.join(evaluate_dir, "preprocess.sh"), hyp_file, tmp_ctm, tmp2_ctm])

    # 2. ground-truth STM 按句 ID 排序(字节序,与 sort -k1,1 行为一致)
    with open(stm_src, "rb") as f:
        lines = f.read().splitlines(keepends=True)
    lines.sort(key=lambda l: (l.split(None, 1)[0] if l.split(None, 1) else b"", l))
    with open(tmp_stm, "wb") as f:
        f.writelines(lines)

    # 3. 为缺失句补齐 [EMPTY] 行(mergectmstm.py 原地改写 tmp2.ctm)
    subprocess.check_call(
        [sys.executable, os.path.join(evaluate_dir, "mergectmstm.py"), tmp2_ctm, tmp_stm])

    out_ctm = prefix + "out." + output_file
    shutil.copyfile(tmp2_ctm, out_ctm)

    # 4. pysclite 对齐并统计(等价 sclite -h ... ctm -r ... stm)。
    #    left_to_right=True 与 pysclite CLI 默认一致,保证假设词序不被反转
    #    (False 时移植版构建反向网络);reduce_ref 对应 sclite 默认 -m ref,
    #    对本仓库数据(每个 STM 段 file id 唯一)无实际影响。
    scor = _stmctm.align_ctm_to_stm_dp(
        tmp_stm, out_ctm, os.path.basename(out_ctm), keep_path=True, case_sense=0,
        char_align=0, fcorr=0, opt_del=0, reduce_ref=True, reduce_hyp=False,
        left_to_right=True)
    if scor is None:
        return "Percent Total Error       =  100.00%   (ALIGN FAILED)"

    tot_ref = tot_err = 0
    for g in scor.grp:
        tot_ref += g.sub + g.corr + g.del_ + g.merges + g.splits
        tot_err += g.sub + g.ins + g.del_ + g.merges + g.splits
    wer = 100.0 * tot_err / tot_ref if tot_ref > 0 else 100.0

    # 5. 输出报告:sum / rsum / pra(原 sgml 报告 pysclite 不支持,已移除)
    if output_dir is not None:
        os.makedirs(prefix + output_dir, exist_ok=True)
        outroot = os.path.join(prefix, output_dir, os.path.basename(out_ctm))
    else:
        outroot = prefix + os.path.basename(out_ctm)
    with open(outroot + ".sys", "wb") as f:
        _sc.print_system_summary(scor, lambda s: f.write(s.encode("latin-1")))
    with open(outroot + ".raw", "wb") as f:
        _sc.print_system_summary(scor, lambda s: f.write(s.encode("latin-1")), do_raw=True)
    with open(outroot + ".pra", "wb") as f:
        _sc.dump_SCORES_alignments(scor, f.write, 1000, False)

    return "Percent Total Error       =  {:6.2f}%   (OpenCSLR)".format(wer)


if __name__ == "__main__":
    evaluate(mode="dev", output_file="output-hypothesis-dev.ctm")
    evaluate(mode="test", output_file="output-hypothesis-test.ctm")
