"""WORD structure and cost functions: port of sclite word.c."""
from . import text as _t

WORD_OPT_DEL_PRE = ord('(')
WORD_OPT_DEL_POST = ord(')')
WORD_FRAG_CHAR = ord('-')
WORD_SGML_SUB_WORD_SEP = ord(';')
WORD_SGML_ESCAPE = ord('\\')


class WORD:
    __slots__ = ('value', 'intern_value', 'tag1', 'tag2', 'value_id',
                 'T1', 'T2', 'T_dur', 'weight', 'conf',
                 'frag_corr', 'opt_del')

    def __init__(self):
        self.value = None
        self.intern_value = None
        self.tag1 = None
        self.tag2 = None
        self.value_id = -1
        self.T1 = 0.0
        self.T2 = 0.0
        self.T_dur = 0.0
        self.weight = -1.0
        self.conf = -1.0
        self.frag_corr = False
        self.opt_del = False


def null_alt_WORD(w):
    return w.value == b'@'


def opt_del_WORD(w):
    return w.opt_del


def new_WORD(t, id=-1, t1=0.0, t2=0.0, conf=0.0, tag1=None, tag2=None,
             fcorr=False, odel=False, weight=-1.0):
    tw = WORD()
    if t is not None:
        n = len(t)
        if n > 1 and t[n - 1] == ord('*'):
            tw.value = t[:n - 1]
        else:
            tw.value = t
    tw.tag1 = tag1
    tw.tag2 = tag2
    tw.value_id = id
    tw.T1 = t1
    tw.T2 = t2
    tw.T_dur = t2 - t1
    tw.weight = weight
    tw.conf = conf
    tw.frag_corr = fcorr
    tw.opt_del = odel
    return tw


def _strbdup_noescape(p, n):
    """TEXT_strBdup_noEscape: copy n bytes skipping backslash escapes."""
    out = bytearray()
    i = 0
    while i < n:
        c = p[i]
        if c != WORD_SGML_ESCAPE:
            out.append(c)
        i += 1
    return bytes(out)


def _next_colon(t, start=0):
    """nextColon: index of next unescaped ';' at/after start, or len(t)."""
    pos = start
    while True:
        idx = t.find(b';', pos)
        if idx < 0:
            return len(t)
        if idx > 0 and t[idx - 1] == WORD_SGML_ESCAPE:
            pos = idx + 1
        else:
            return idx


def new_WORD_parseText(t, id=-1, t1=0.0, t2=0.0, conf=0.0, fcorr=False,
                       odel=False, weight=-1.0):
    """Parse token; '{'-starting tokens stay whole; else split on ';'."""
    if len(t) > 0 and t[0] == ord('{'):
        return new_WORD(t, id, t1, t2, conf, None, None, fcorr, odel, weight)
    text = None
    tag1 = None
    tag2 = None
    ptr = 0
    end = _next_colon(t, ptr)
    text = _strbdup_noescape(t[ptr:end], end - ptr)
    if end < len(t):
        ptr = end + 1
        end = _next_colon(t, ptr)
        tag1 = _strbdup_noescape(t[ptr:end], end - ptr)
        if end < len(t):
            ptr = end + 1
            end = _next_colon(t, ptr)
            tag2 = _strbdup_noescape(t[ptr:end], end - ptr)
    return new_WORD(text, id, t1, t2, conf, tag1, tag2, fcorr, odel, weight)


def make_empty_WORD():
    return new_WORD(b'', -1, 0.0, 0.0, 0.0, None, None, False, False, 0.0)


def wwd_WORD(w1, w2, cmp):
    """Weight/cost function: correct 0, null ins/del 0.001, opt_del 2,
    ins/del 3, sub 4, null<->null sub 1."""
    if w1 is None and w2 is None:
        raise RuntimeError("wwd_WORD computed for two NULL_WORDS")
    if w2 is None and w1.value == b'@':
        return 0.001
    if w1 is None and w2.value == b'@':
        return 0.001
    if (w1 is None and w2.opt_del) or (w2 is None and w1.opt_del):
        return 2.0
    if w1 is None or w2 is None:
        return 3.0
    if w1.value == b'@' and w2.value == b'@':
        return 1.0
    if cmp(w1, w2) == 0:
        return 0.0
    return 4.0


def wwd_time_WORD(w1, w2, cmp):
    if w1 is None and w2 is None:
        raise RuntimeError("wwd_WORD computed for two NULL_WORDS")
    if w2 is None and w1.value == b'@':
        return 0.001
    if w1 is None and w2.value == b'@':
        return 0.001
    if w1 is None:
        return w2.T_dur
    if w2 is None:
        return w1.T_dur
    return (abs(w1.T1 - w2.T1) + abs(w1.T2 - w2.T2)
            + (0.0 if cmp(w1, w2) == 0 else 0.001))


def wwd_weight_WORD(w1, w2, cmp):
    if w1 is None and w2 is None:
        raise RuntimeError("wwd_WORD computed for two NULL_WORDS")
    if w2 is None and w1.value == b'@':
        return 0.001
    if w1 is None and w2.value == b'@':
        return 0.001
    if w1 is None:
        return w2.weight
    if w2 is None:
        return w1.weight
    if cmp(w1, w2) == 0:
        return 0.0
    return w1.weight + w2.weight


def equal_WORD(p1, p2):
    if p1 is None:
        return -1
    if p2 is None:
        return 1
    return _t.strcmp(p1.value, p2.value)


def equal_WORD2(p1, p2):
    """Comparison honoring opt_del (intern_value) and frag_corr."""
    if p1 is None:
        return -1
    if p2 is None:
        return 1
    it1 = p1.intern_value if p1.opt_del else p1.value
    it2 = p2.intern_value if p2.opt_del else p2.value
    l1 = len(it1)
    l2 = len(it2)
    if p1.frag_corr or p2.frag_corr:
        if it1[0:1] == b'-' and l1 > 1:
            if l2 < l1 - 1:
                return 1
            return _strncmp(it1[1:], it2[l2 - (l1 - 1):], l1 - 1)
        elif it1[l1 - 1:l1] == b'-' and l1 > 1:
            return _strncmp(it1, it2, l1 - 1)
        if it2[0:1] == b'-' and l2 > 1:
            if l1 < l2 - 1:
                return -1
            return _strncmp(it1[l1 - (l2 - 1):], it2[1:], l2 - 1)
        elif it2[l2 - 1:l2] == b'-' and l2 > 1:
            return _strncmp(it1, it2, l2 - 1)
    return _t.strcmp(it1, it2)


def _strncmp(a, b, n):
    a = a[:n]
    b = b[:n]
    if a == b:
        return 0
    return -1 if a < b else 1
