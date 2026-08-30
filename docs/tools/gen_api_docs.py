#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 core/ 源码自动生成 Sphinx API 参考页(确定性,仅标准库)。

提取内容(全部来自源码本身,无需手写):
  - 模块级 / 类 / 方法 docstring
  - 类与方法签名(从 AST 重建,不导入代码)
  - 行内注释(整行注释 + 行尾注释,经 tokenize 解析,归属到所属函数/类)
  - ``parser.add_argument(...)`` 调用(生成命令行参数表)

输出:
  docs/source/api/<模块点分路径>.rst   每个模块一页
  docs/source/api.rst                 API 索引页

用法(仓库根目录或任意目录均可):
  python docs/tools/gen_api_docs.py
  # 可选参数
  python docs/tools/gen_api_docs.py --src core --out docs/source/api \
      --index docs/source/api.rst

重新运行会覆盖旧输出,并删除 api/ 下不再存在的模块页。
"""
import argparse
import ast
import bisect
import io
import os
import sys
import tokenize
import unicodedata
import warnings

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
DEFAULT_SRC = os.path.join(REPO_ROOT, "core")
DEFAULT_OUT = os.path.join(REPO_ROOT, "docs", "source", "api")
DEFAULT_INDEX = os.path.join(REPO_ROOT, "docs", "source", "api.rst")

# 相对 core/ 的第三方(vendored)目录前缀,不生成文档
EXCLUDED_DIRS = [
    "models/modules/slowfast",
    "libs/sync_batchnorm",
    "libs/slr_eval",
]

# 顶层包/文件的分组标题(用于 api.rst 索引)
GROUP_LABELS = {
    "main": "程序入口",
    "manager": "管理器(Manager)",
    "models": "模型(Models)",
    "dataset": "数据集(Dataset)",
    "pipline": "训练流水线(Pipeline)",
    "preprocess": "数据预处理(Preprocess)",
    "libs": "工具库(Libs)",
}
GROUP_ORDER = list(GROUP_LABELS.keys())

GENERATED_NOTE = (
    ".. note::\n"
    "   本页由 ``docs/tools/gen_api_docs.py`` 从源码注释自动生成,请勿手动编辑。\n"
    "   重新生成:``cd docs && make gen-api``(随后 ``make html`` 构建)。\n"
)


# --------------------------------------------------------------------------- #
# 发现模块
# --------------------------------------------------------------------------- #

def find_modules(src_root):
    """返回 [(dotted_name, absolute_path)] ,dotted 名相对 src_root。"""
    mods = []
    for root, dirs, files in os.walk(src_root):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, src_root).replace(os.sep, "/")
            if any(rel.startswith(p + "/") for p in EXCLUDED_DIRS):
                continue
            rel_dir = os.path.relpath(root, src_root).replace(os.sep, ".")
            dotted = "{}.{}".format(rel_dir, name[:-3]) if rel_dir != "." else name[:-3]
            mods.append((dotted, path, rel.replace(os.sep, "/")))
    return mods


# --------------------------------------------------------------------------- #
# 源码解析
# --------------------------------------------------------------------------- #

def parse_file(path):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    with warnings.catch_warnings():
        # 源码里的非原始字符串转义(如 "\d")会触发 SyntaxWarning,与本工具无关
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(src)
    comments, code_lines = extract_comments(src)
    return tree, src, comments, code_lines


def extract_comments(src):
    """返回 (comments, code_lines):
    comments: [(lineno, text)] 全部注释(tokenize 解析,正确处理字符串)
    code_lines: 含真实代码的行的行号列表(升序)
    """
    comments = []
    code_lines = set()
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # 兜底:仅整行注释
        for i, line in enumerate(src.splitlines(), 1):
            if line.strip().startswith("#"):
                comments.append((i, line.strip()[1:].strip()))
        return comments, sorted(set(i for i, l in enumerate(src.splitlines(), 1)
                                    if l.strip() and not l.strip().startswith("#")))
    non_code = {tokenize.COMMENT, tokenize.NEWLINE, tokenize.NL,
                tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER,
                tokenize.ENCODING}
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            comments.append((tok.start[0], tok.string.strip()))
        elif tok.type not in non_code:
            code_lines.add(tok.start[0])
    return comments, sorted(code_lines)


def collect_top_nodes(tree):
    classes, funcs = [], []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append(node)
    return classes, funcs


def method_nodes(cls_node):
    return [m for m in cls_node.body
            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]


def assign_comments(comments, code_lines, classes, funcs):
    """把注释归属到节点。
    返回 (per_node_inline, per_node_pre, module_comments)
    per_node_*: {id(node): [(lineno, text)]}
    """
    all_nodes = []
    for c in classes:
        all_nodes.append(c)
        all_nodes.extend(method_nodes(c))
    all_nodes.extend(funcs)

    def innermost(line):
        best = None
        for n in all_nodes:
            if n.lineno <= line <= n.end_lineno:
                if best is None or (n.end_lineno - n.lineno) < \
                        (best.end_lineno - best.lineno):
                    best = n
        return best

    inline = {}
    pre = {}
    module_comments = []
    for line, text in comments:
        owner = innermost(line)
        if owner is not None:
            inline.setdefault(id(owner), []).append((line, text))
            continue
        # 前置注释:注释行之后、下一个代码行恰好是某节点起始行
        idx = bisect.bisect_right(code_lines, line)
        if idx < len(code_lines) and code_lines[idx] > line:
            next_code = code_lines[idx]
            target = next((n for n in all_nodes if n.lineno == next_code), None)
            if target is not None:
                pre.setdefault(id(target), []).append((line, text))
                continue
        module_comments.append((line, text))
    return inline, pre, module_comments


# --------------------------------------------------------------------------- #
# 签名 / 参数重建
# --------------------------------------------------------------------------- #

def signature(node):
    a = node.args
    parts = []
    pos = list(a.posonlyargs) + list(a.args)
    n_posonly = len(a.posonlyargs)
    defaults = [None] * (len(pos) - len(a.defaults)) + list(a.defaults)
    for i, (arg, d) in enumerate(zip(pos, defaults)):
        s = arg.arg
        if arg.annotation is not None:
            s += ": " + ast.unparse(arg.annotation)
        if d is not None:
            s += " = " + ast.unparse(d)
        parts.append(s)
        if i == n_posonly - 1:
            parts.append("/")
    if a.vararg is not None:
        parts.append("*" + a.vararg.arg)
    for arg, d in zip(a.kwonlyargs, a.kw_defaults):
        s = arg.arg
        if arg.annotation is not None:
            s += ": " + ast.unparse(arg.annotation)
        if d is not None:
            s += " = " + ast.unparse(d)
        parts.append(s)
    if a.kwarg is not None:
        parts.append("**" + a.kwarg.arg)
    sig = "def {}({})".format(node.name, ", ".join(parts))
    if node.returns is not None:
        sig += " -> " + ast.unparse(node.returns)
    return sig


def decorator_lines(node):
    return ["@" + ast.unparse(d) for d in node.decorator_list]


def _unq(s):
    """去掉 unparse 出来的字符串常量外层引号。"""
    if len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
        return s[1:-1]
    return s


def _short_type(node):
    if isinstance(node, ast.Name):
        return node.id
    return ast.unparse(node)


def extract_argparse(node):
    """提取函数内 parser.add_argument(...) 调用,返回行列表。"""
    rows = []
    for sub in ast.walk(node):
        if not (isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "add_argument"):
            continue
        options = []
        for arg in sub.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                options.append(arg.value)
            elif isinstance(arg, (ast.List, ast.Tuple)):
                for el in arg.elts:
                    if isinstance(el, ast.Constant) and isinstance(el.value, str):
                        options.append(el.value)
            else:
                options.append(ast.unparse(arg))
        row = {"options": options, "type": "-", "default": "-",
               "help": "", "action": None}
        for kw in sub.keywords:
            v = kw.value
            if kw.arg == "help":
                row["help"] = _unq(ast.unparse(v))
            elif kw.arg == "default":
                row["default"] = ast.unparse(v)
            elif kw.arg == "type":
                row["type"] = _short_type(v)
            elif kw.arg == "action":
                row["action"] = _unq(ast.unparse(v))
        if row["action"]:
            row["help"] = "{}  ({})".format(row["help"] or "-", row["action"])
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# RST 渲染
# --------------------------------------------------------------------------- #

def _display_width(s):
    """RST 标题下划线按显示宽度计算(全角/CJK 字符按 2 计)。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1
               for c in s)


def heading(text, level):
    chars = "=-~^"
    ch = chars[min(level, len(chars) - 1)]
    return "{}\n{}{}".format(text, ch, ch * max(_display_width(text), 3))


def code_block(lang, text):
    lines = [".. code-block:: {}".format(lang), ""]
    for line in text.rstrip("\n").splitlines():
        lines.append("    " + line if line.strip() else "")
    return "\n".join(lines)


def comments_block(comments):
    lines = [t if t.startswith("#") else "# " + t for _, t in comments]
    return code_block("text", "\n".join(lines))


def _cell_inline(text):
    """单元格内容压成单行,代码样内容用行内字面量(等宽字体)。"""
    text = " ".join(str(text).split()) or "-"
    if "`" in text:
        return text
    return "``{}``".format(text)


def list_table(header, rows, inline_cols=()):
    """RST list-table。

    不用 grid table 的原因:grid table 的右边界对齐按"pad 全角字符后
    各行长度相等"检查,含中文的表格很难手工对齐;list-table 是两级
    列表语法,不要求列宽对齐,对中文安全。
    inline_cols: 以行内字面量渲染的列下标(代码风格)。
    """
    out = [".. list-table::", "   :header-rows: 1", "   :widths: auto", ""]
    # 表头是普通文本;inline_cols 只作用于数据行(代码样内容)
    out.append("   * - " + " ".join(str(header[0]).split()))
    for i in range(1, len(header)):
        out.append("     - " + " ".join(str(header[i]).split()))
    for row in rows:
        for i, c in enumerate(row):
            marker = "   * - " if i == 0 else "     - "
            text = _cell_inline(c) if i in inline_cols else str(c)
            out.append(marker + " ".join(str(text).split()))
    return "\n".join(out)


def argparse_table(rows):
    header = ["参数", "类型", "默认值", "说明"]
    data = []
    for r in rows:
        data.append([
            ", ".join(r["options"]) or "-",
            r["type"],
            r["default"] or "-",
            r["help"] or "-",
        ])
    return list_table(header, data, inline_cols=(0, 1, 2))


def render_signature(node):
    lines = decorator_lines(node)
    lines.append(signature(node))
    return code_block("python", "\n".join(lines))


def render_method(node, inline, pre):
    out = []
    out.append(".. rubric:: {}".format(node.name))
    out.append("")
    out.append(render_signature(node))
    out.append("")
    doc = ast.get_docstring(node)
    if doc:
        out.append(code_block("text", doc.strip()))
        out.append("")
    if id(node) in pre:
        out.append(".. rubric:: 前置注释")
        out.append("")
        out.append(comments_block(pre[id(node)]))
        out.append("")
    if id(node) in inline:
        out.append(".. rubric:: 注释")
        out.append("")
        out.append(comments_block(inline[id(node)]))
        out.append("")
    ap = extract_argparse(node)
    if ap:
        out.append(".. rubric:: 命令行参数")
        out.append("")
        out.append(argparse_table(ap))
        out.append("")
    return "\n".join(out)


def render_class(cls_node, inline, pre):
    out = []
    out.append(heading(cls_node.name, 2))
    out.append("")
    doc = ast.get_docstring(cls_node)
    if doc:
        out.append(code_block("text", doc.strip()))
        out.append("")
    if id(cls_node) in pre:
        out.append(".. rubric:: 前置注释")
        out.append("")
        out.append(comments_block(pre[id(cls_node)]))
        out.append("")
    methods = method_nodes(cls_node)
    if methods:
        out.append(heading("方法", 3))
        out.append("")
        for m in methods:
            out.append(render_method(m, inline, pre))
    else:
        if id(cls_node) in inline:
            out.append(".. rubric:: 注释")
            out.append("")
            out.append(comments_block(inline[id(cls_node)]))
            out.append("")
    return "\n".join(out)


def render_module(dotted, relpath, tree, comments, code_lines):
    classes, funcs = collect_top_nodes(tree)
    inline, pre, module_comments = assign_comments(comments, code_lines,
                                                   classes, funcs)
    doc = ast.get_docstring(tree)

    out = []
    out.append(".. _api-{}:".format(dotted.replace(".", "-")))
    out.append("")
    out.append(heading(dotted, 0))
    out.append("")
    out.append(GENERATED_NOTE)
    out.append("")
    out.append("**源文件**: ``{}``".format(relpath))
    out.append("")

    if not classes and not funcs and not doc and not module_comments:
        return None  # 无可生成内容

    out.append(heading("模块文档", 1))
    out.append("")
    out.append(code_block("text", doc.strip()) if doc
               else "该模块没有模块级 docstring。")
    out.append("")

    if classes:
        out.append(heading("类", 1))
        out.append("")
        for c in classes:
            out.append(render_class(c, inline, pre))
    if funcs:
        out.append(heading("模块级函数", 1))
        out.append("")
        for f in funcs:
            out.append(render_method(f, inline, pre))
    if module_comments:
        out.append(heading("模块级注释", 1))
        out.append("")
        out.append(comments_block(module_comments))
        out.append("")
    return "\n".join(out)


def render_index(groups):
    """groups: [(group_label, [dotted, ...])]
    每个分组一个带 :caption: 的 toctree(Sphinx 单个 toctree 只支持一个 caption)。"""
    out = []
    out.append(heading("API 参考", 0))
    out.append("")
    out.append(GENERATED_NOTE)
    out.append("")
    for label, dotted_list in groups:
        if not dotted_list:
            continue
        out.append(".. toctree::")
        out.append("   :maxdepth: 1")
        out.append("   :caption: {}".format(label))
        out.append("")
        for d in dotted_list:
            out.append("   api/{}".format(d))
        out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=DEFAULT_SRC, help="源码根目录(默认 core/)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="API 页输出目录")
    ap.add_argument("--index", default=DEFAULT_INDEX, help="索引页输出路径")
    args = ap.parse_args(argv)

    src_root = os.path.abspath(args.src)
    out_dir = os.path.abspath(args.out)
    index_path = os.path.abspath(args.index)
    os.makedirs(out_dir, exist_ok=True)

    mods = find_modules(src_root)
    groups = {}
    order = []
    written = []
    for dotted, path, rel in mods:
        try:
            tree, _src, comments, code_lines = parse_file(path)
        except (SyntaxError, UnicodeDecodeError, OSError) as e:
            print("[warn] 跳过 {} : {}".format(rel, e), file=sys.stderr)
            continue
        top = dotted.split(".")[0] if "." in dotted else dotted
        src_label = os.path.basename(src_root) + "/" + rel
        try:
            rst = render_module(dotted, src_label, tree, comments,
                                code_lines)
        except Exception as e:  # noqa: BLE001 - 单文件失败不应中断整体
            print("[warn] 渲染失败 {} : {!r}".format(rel, e), file=sys.stderr)
            continue
        if rst is None:
            continue
        out_name = os.path.join(out_dir, dotted + ".rst")
        with open(out_name, "w", encoding="utf-8") as f:
            f.write(rst + "\n")
        written.append(out_name)
        if top not in groups:
            groups[top] = []
            order.append(top)
        groups[top].append(dotted)

    # 清理过期页面
    for name in os.listdir(out_dir):
        if name.endswith(".rst") and os.path.join(out_dir, name) not in written:
            stale = os.path.join(out_dir, name)
            os.remove(stale)
            print("[info] 删除过期页面 {}".format(stale))

    # 索引分组:按 GROUP_ORDER,其余按字母序
    labeled = []
    for name in GROUP_ORDER:
        if name in groups:
            labeled.append((GROUP_LABELS[name], groups[name]))
    for name in sorted(groups):
        if name not in GROUP_ORDER:
            labeled.append((name, groups[name]))

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(render_index(labeled) + "\n")

    print("[ok] 生成 {} 个模块页 -> {}".format(len(written), out_dir))
    print("[ok] 生成索引页 -> {}".format(index_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
