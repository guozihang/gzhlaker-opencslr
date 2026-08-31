# pysclite — NIST sclite 的 Python 移植

pysclite 是 NIST SCTK 中 `sclite`（语音识别 WER 评分标准工具，SCTK 2.10 版）
的纯 Python 移植。算法、动态规划对齐、统计计算与报告排版逐行复刻自 C 源码，
**计算指标与原版 sclite 完全一致**。

## 验证

使用 SCTK 官方测试套件（`src/sclite/testdata/tsclite.sh` 中 40 个 DP 对齐
测试用例）与原版 C 二进制逐字节对拍：

- 全部 77 个指标报告文件（`.sys` / `.raw`）**字节一致**
- 对齐明细文件（`.pra` / `.prf`）字节一致
- 唯一差异：错误消息中的程序名（`pysclite` vs `sclite`），仅出现在
  stderr，不影响任何输出文件

覆盖的测试维度：

- TRN ↔ TRN（含 `-i wsj/spu_id` 等说话人 ID 规则、`-F` 碎片词、
  `-D` 可删词、`-c` 字符级对齐与 `DH/NOASCII` 选项）
- STM ↔ CTM（含 `IGNORE_TIME_SEGMENT_IN_SCORING`、置信度/NCE、
  `-m ref/hyp` 交集缩减、参考缺失/多余等边界情况）
- CTM ↔ CTM（含 `-T` 时间中介对齐）
- 编码：ASCII、UTF-8（1–4 字节字符、粤语/土耳其语/乌克兰语/越南语
  大小写转换表）、GB（普通话字符对齐）

## 用法

```bash
python3 -m pysclite -r ref.trn -h hyp.trn -i wsj -o all
python3 -m pysclite -r ref.stm stm -h hyp.ctm ctm -o all prf
python3 -m pysclite -r ref.ctm ctm -h hyp.ctm ctm -o sum -O outdir -n mytest
```

包无需安装，将 `pysclite/` 目录置于 `PYTHONPATH` 即可（无第三方依赖，
仅需 Python 3.6+）。

## 支持的选项

| 选项 | 说明 |
|------|------|
| `-r file [fmt]` | 参考文件，格式 `trn`（默认）/`stm`/`ctm`/`tmk` |
| `-h file [fmt] [title]` | 假设文件，格式 `trn`/`ctm`/`tmk`/`txt`（txt 不可用，见下） |
| `-i id` | TRN 模式的说话人 ID 规则：`sp rm wsj swb atis spu_id` |
| `-o ...` | 输出：`sum rsum pra pralign prf all none stdout` |
| `-O dir` / `-n name` | 输出目录 / 文件名（生成 `name.sys`、`name.raw`、`name.pra`、`name.prf`） |
| `-s` | 大小写敏感 |
| `-F` | 碎片词（`word-` / `-word`）判对 |
| `-D` | 可删词 `(word)` 处理 |
| `-c [NOASCII] [DH]` | 字符级对齐（可去连字符、跳过 ASCII 词） |
| `-e enc [prof]` | 编码：`ascii`/`utf-8`/`gb`/`extascii`，可选本地化大小写表（如 `babel_turkish`、`ukrainian`、`babel_vietnamese`） |
| `-T` | CTM↔CTM 时间中介对齐 |
| `-m [ref] [hyp]` | STM↔CTM 输入交集缩减 |
| `-l n` | pra/prf 报告行宽（默认 1000） |
| `-f n` | 反馈信息级别（默认 1） |

## 未移植的功能

以下功能在官方构建中默认即被禁用或属于非核心扩展，本移植不支持
（调用时会给出明确报错）：

- `-d` diff 对齐（官方构建已禁用：`*Alignments via diff have been disabled*`）
- `-S` 推断分词（`-S algo1/algo2`）
- `-w` 词权重文件（WWL）、`-L` 语言模型权重（SLM）
- `-C` 置信度 DET/直方图报告
- `-p`/`-P` 管道输入输出
- `lur`/`snt`/`spk`/`dtl`/`sgml`/`nl.sgml` 输出格式

## 实现说明

内部全部按字节（bytes）处理文本，复刻 C 的 unsigned-char 语义；
DP 代价计算复刻原版 32 位 `float` 精度行为，保证对齐选择与统计数值
逐位一致。模块结构对应 C 源码：

| 模块 | 对应 C 文件 |
|------|-------------|
| `text.py` | text.c（编码、大小写表、分词） |
| `word.py` | word.c（词结构、DP 代价） |
| `network.py` | net_adt.c、net_dp.c（词网络与 DP 对齐） |
| `path.py` | path.c（对齐路径与打印） |
| `rpg.py` | rpg.c、pad.c（报告排版引擎） |
| `scores.py` | scores.c、statdist.c（统计与汇总报告） |
| `align.py` | align.c（TRN 模式） |
| `stmctm.py` | stm.c、fillmrks.c、stm2ctm.c、ctm2ctm.c |
| `cli.py` | sclite.c（参数解析与主流程） |
