# -*- encoding: utf-8 -*-
"""WER 评估管理器模块。

基于仓库内置 pysclite(纯 Python 移植版 sclite)的评估流程,无任何编译依赖,
也无需 shell 工具(原 preprocess.sh 的 sed/awk/perl 清洗逻辑已移植为
preprocess_hyp)。与初始化链中的其他管理器不同,本管理器不持有跨模块状态,
在评估阶段按需调用。

流程:
  1. preprocess_hyp 清洗假设 CTM(标注归一、去特殊标签、合并重复词);
  2. ground-truth STM(evaluate_dir/groundtruth/)按句 ID 排序(等价 sort -k1,1);
  3. merge_ctm_stm 为缺失句补齐 [EMPTY] 行;
  4. pysclite 对齐 CTM 与 STM,输出 sum/rsum/pra 报告并计算 WER。
"""

import os
import re
import shutil

from libs.pysclite import scores as _sc
from libs.pysclite import stmctm as _stmctm


class EvaluationManager:
    """WER 评估管理器,提供 CTM/STM 对齐评分与报告输出。

    评估所需数据(groundtruth/*.stm)位于 libs/slr_eval/,由
    ``evaluate_dir`` 参数指定。
    """

    @classmethod
    def preprocess_hyp(cls, hyp_file, output_file):
        """清洗假设 CTM(原 preprocess.sh 的纯 Python 移植,规则逐条对应)。

        处理步骤与原 bash 脚本一致:
          1. 标注归一(sed 部分):loc-/cl-/qu-/poss-/lh- 前缀、S0NNE/HABEN2、
             特殊标签、WIE AUSSEHEN / ZEIGEN 复合词、指拼词 A B -> A+B 等;
          2. 合并连续重复词(perl 部分,4 遍);
          3. 去掉含特殊标签的行(grep -v 部分);
          4. 词被清空的句补 [EMPTY](awk 部分);
          5. 按句 ID 与起始时间排序(sort -k1,1 -k3,3 部分)。

        Args:
            hyp_file: 模型输出的假设 CTM 文件(每行 5 列)。
            output_file: 清洗后 CTM 输出路径。
        """
        with open(hyp_file, "r", encoding="utf-8") as f:
            raw_lines = f.readlines()

        cleaned = []
        for line in raw_lines:
            line = line.rstrip("\n")

            # 1. sed 部分:标注归一(规则顺序与原脚本一致)
            line = (line.replace("loc-", "").replace("cl-", "")
                        .replace("qu-", "").replace("poss-", "")
                        .replace("lh-", "")
                        .replace("S0NNE", "SONNE").replace("HABEN2", "HABEN")
                        .replace("__EMOTION__", "").replace("__PU__", "")
                        .replace("__LEFTHAND__", "")
                        .replace("WIE AUSSEHEN", "WIE-AUSSEHEN")
                        .replace("ZEIGEN ", "ZEIGEN-BILDSCHIRM "))
            line = re.sub(r"ZEIGEN$", "ZEIGEN-BILDSCHIRM", line)
            line = re.sub(r"^([A-Z]) ([A-Z][+ ])", r"\1+\2", line)
            line = re.sub(r"[ +]([A-Z]) ([A-Z]) ", r" \1+\2 ", line)
            for _ in range(3):  # 原脚本连续 3 个 sed 重复此规则
                line = re.sub(r"([ +][A-Z]) ([A-Z][ +])", r"\1+\2", line)
            line = re.sub(r"([ +]SCH) ([A-Z][ +])", r"\1+\2", line)
            line = re.sub(r"([ +]NN) ([A-Z][ +])", r"\1+\2", line)
            line = re.sub(r"([ +][A-Z]) (NN[ +])", r"\1+\2", line)
            line = re.sub(r"([ +][A-Z]) ([A-Z])$", r"\1+\2", line)
            line = re.sub(r"([A-Z][A-Z])RAUM", r"\1", line)
            line = line.replace("-PLUSPLUS", "")

            # 2. perl 部分:合并连续重复词(原脚本 4 个 perl 各一遍)
            for _ in range(4):
                line = re.sub(r"(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])",
                              r"\1", line)

            # 3. grep -v 部分:去掉含特殊标签的行
            if any(tok in line for tok in
                   ("__LEFTHAND__", "__EPENTHESIS__", "__EMOTION__")):
                continue

            # 4. sed 's,\s*$,':去行尾空白。按原脚本 GNU sed 语义实现(\s 即空白);
            #    macOS 的 BSD sed 把 \s 当字面 's'(会误删行尾小写 s 且不去空白),
            #    属平台缺陷;本实现与下游行为(按空白切分)对 WER 无影响。
            cleaned.append(line.rstrip())

        # 5. awk 部分:词被清空($5 为空)的行不输出,并为其句补 [EMPTY]
        cnt = {}
        last_id = ""
        last_row = ""
        filled = []
        for line in cleaned:
            fields = line.split()
            cur_id = fields[0] if fields else ""
            if last_id != cur_id and cnt.get(last_id, 0) < 1 and last_row:
                filled.append(last_row + " [EMPTY]")
            if len(fields) >= 5 and fields[4] != "":
                cnt[cur_id] = cnt.get(cur_id, 0) + 1
                filled.append(line)
            last_id = cur_id
            last_row = line

        # 6. sort -k1,1 -k3,3 部分:按句 ID、起始时间排序(键相同时按整行,
        #    与原 sort 的兜底比较一致)
        filled.sort(key=lambda l: (l.split(None, 2)[0] if l.split() else "",
                                   l.split()[2] if len(l.split()) > 2 else "", l))

        with open(output_file, "w", encoding="utf-8") as f:
            for line in filled:
                f.write(line + "\n")

    @classmethod
    def merge_ctm_stm(cls, ctm_file, stm_file):
        """为 CTM 中缺失的句补齐 [EMPTY] 行。

        按 STM 中的句 ID 顺序遍历:CTM 中该 ID 对应的行缺失时插入
        "{id} 1 0.000 0.030 [EMPTY]",使每条参考句都有假设行,保证对齐
        覆盖全部数据。原地改写 ctm_file。

        Args:
            ctm_file: 假设 CTM 文件路径(按句 ID 分组、组内按起始时间排序)。
            stm_file: 按句 ID 排序的参考 STM 文件路径。
        """
        with open(ctm_file, "r", encoding="utf-8") as f:
            ctm_dict = [line.strip().split() for line in f]
        with open(stm_file, "r", encoding="utf-8") as f:
            stm_dict = [line.strip().split() for line in f]

        added_lines = 0
        for idx, stm_line in enumerate(stm_dict):
            # ctm and stm match:跳过该句 ID 的全部 CTM 行
            if (len(ctm_dict) > idx + added_lines
                    and ctm_dict[idx + added_lines][0] == stm_line[0]):
                while (len(ctm_dict) > idx + added_lines + 1
                       and ctm_dict[idx + added_lines + 1][0] == stm_line[0]):
                    added_lines += 1
            else:
                ctm_dict.insert(idx + added_lines,
                                [stm_line[0], "1 0.000 0.030 [EMPTY]"])

        with open(ctm_file, "w", encoding="utf-8") as f:
            for line in ctm_dict:
                f.write(" ".join(line) + "\n")

    @classmethod
    def evaluate(cls, prefix="./", mode="dev", evaluate_dir=None, evaluate_prefix=None,
                 output_file=None, output_dir=None):
        """计算指定模式(dev/test)的 WER。

        Args:
            prefix: 输出路径前缀(通常是 work_dir,以 / 结尾)。
            mode: 评估模式("dev"/"test"),用于选择 ground-truth 文件。
            evaluate_dir: slr_eval 工具目录(含 groundtruth/ 子目录)。
            evaluate_prefix: ground-truth 文件名前缀(如 "phoenix2014-groundtruth"),
                对应 groundtruth/ 下的 {prefix}-{mode}.stm 文件。
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
        tmp2_ctm = prefix + "tmp2.ctm"
        stm_src = os.path.join(evaluate_dir, "groundtruth",
                               "{}-{}.stm".format(evaluate_prefix, mode))
        tmp_stm = prefix + "tmp.stm"

        # 1. 清洗假设 CTM(纯 Python,无需 shell 工具)
        cls.preprocess_hyp(hyp_file, tmp2_ctm)

        # 2. ground-truth STM 按句 ID 排序(字节序,与 sort -k1,1 行为一致)
        with open(stm_src, "rb") as f:
            lines = f.read().splitlines(keepends=True)
        lines.sort(key=lambda l: (l.split(None, 1)[0] if l.split(None, 1) else b"", l))
        with open(tmp_stm, "wb") as f:
            for line in lines:
                if not line.endswith(b"\n"):
                    line += b"\n"
                f.write(line)

        # 3. 为缺失句补齐 [EMPTY] 行(原地改写 tmp2.ctm)
        cls.merge_ctm_stm(tmp2_ctm, tmp_stm)

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
