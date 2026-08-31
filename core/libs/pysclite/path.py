"""PATH structure and alignment printing: port of sclite path.c."""
from . import text as _t

P_CORR = 0x01
P_SUB = 0x02
P_INS = 0x04
P_DEL = 0x08
P_MRG = 0x10
P_SPL = 0x20

PA_NONE = 0x0000
PA_CHAR_ALIGN = 0x0001
PA_CASE_SENSE = 0x0002
PA_HYP_WTIMES = 0x0004
PA_REF_WTIMES = 0x0008
PA_HYP_TIMES = 0x0010
PA_REF_TIMES = 0x0020
PA_HYP_CONF = 0x0040
PA_REF_CONF = 0x0080
PA_HYP_WEIGHT = 0x0100
PA_REF_WEIGHT = 0x0200
PA_HYP_SPKR = 0x0400
PA_HYP_ISSPKRSUB = 0x0800

_ASTER = b'****************************************'  # 40 chars
_ALEN = 40


class PathSet:
    __slots__ = ('eval', 'a_ptr', 'b_ptr')

    def __init__(self, eval=0, a_ptr=None, b_ptr=None):
        self.eval = eval
        self.a_ptr = a_ptr
        self.b_ptr = b_ptr


class PATH:
    def __init__(self):
        self.pset = []
        self.num = 0
        self.id = None
        self.labels = None
        self.file = None
        self.channel = None
        self.attrib = PA_NONE
        self.sequence = -1
        self.ref_t1 = 0.0
        self.ref_t2 = 0.0
        self.hyp_t1 = 0.0
        self.hyp_t2 = 0.0


def sort_PATH_time_marks(path):
    """Bubble-sort runs of consecutive INS/DEL by T1."""
    x = 0
    n = path.num
    while x < n:
        while x < n and path.pset[x].eval not in (P_DEL, P_INS):
            x += 1
        change = True
        while x < n and change:
            change = False
            tx = x + 1
            while tx < n and path.pset[tx].eval in (P_DEL, P_INS):
                p_prev = path.pset[tx - 1]
                p_cur = path.pset[tx]
                ta = p_prev.a_ptr.T1 if p_prev.eval == P_DEL \
                    else p_prev.b_ptr.T1
                tb = p_cur.a_ptr.T1 if p_cur.eval == P_DEL \
                    else p_cur.b_ptr.T1
                if ta > tb:
                    path.pset[tx], path.pset[tx - 1] = p_prev, p_cur
                    change = True
                tx += 1
        while x < n and path.pset[x].eval in (P_DEL, P_INS):
            x += 1


def _ljust(b, w):
    """C '%-Ns ' formatting on bytes: left-justify, pad with spaces."""
    if len(b) < w:
        return b + b' ' * (w - len(b))
    return b


def _rjust(b, w):
    if len(b) < w:
        return b' ' * (w - len(b)) + b
    return b


def PATH_n_print(path, out, frm, to, maxw):
    """PATH_n_print: writes bytes to `out` (a callable taking bytes)."""
    if to > path.num:
        to = path.num
    used = frm if (0 <= frm < to) else 0
    c = s = d = n = u = 0
    for i in range(used, to):
        ev = path.pset[i].eval
        if ev & P_CORR:
            c += 1
        elif ev & P_SUB:
            s += 1
        elif ev & P_INS:
            n += 1
        elif ev & P_DEL:
            d += 1
        else:
            u += 1
    aux_max = 0
    for flag in (PA_HYP_CONF, PA_REF_CONF, PA_HYP_WTIMES, PA_REF_WTIMES,
                 PA_HYP_WEIGHT, PA_REF_WEIGHT):
        if path.attrib & flag:
            aux_max = max(aux_max, 7)

    def s_(x):
        return x if isinstance(x, bytes) else x.encode('latin-1')

    out(b"id: %s\n" % s_(path.id if path.id is not None else b''))
    if frm != 0 or to != path.num:
        out(b"Range: %d-%d of %d words\n" % (frm, to, path.num))
    if path.labels is not None:
        out(b"Labels: %s\n" % s_(path.labels))
    if path.file is not None:
        out(b"File: %s\n" % s_(path.file))
    if path.channel is not None:
        out(b"Channel: %s\n" % s_(path.channel))
    if u > 0:
        out(b"Scores: (#C #S #D #I #UNK) %d %d %d %d %d\n" % (c, s, d, n, u))
    else:
        out(b"Scores: (#C #S #D #I) %d %d %d %d\n" % (c, s, d, n))
    if path.attrib & (PA_CHAR_ALIGN | PA_CASE_SENSE | PA_REF_WEIGHT
                      | PA_HYP_WEIGHT):
        out(b"Attributes: ")
        if path.attrib & PA_CHAR_ALIGN:
            out(b"Character_align ")
        if path.attrib & PA_CASE_SENSE:
            out(b"Case_sensitve ")
        if path.attrib & PA_REF_WEIGHT:
            out(b"Ref_weight ")
        if path.attrib & PA_HYP_WEIGHT:
            out(b"Hyp_weight ")
        out(b"\n")
    if path.attrib & PA_REF_TIMES:
        out(b"Ref times: t1= %.2f t2= %.2f\n" % (path.ref_t1, path.ref_t2))
    if path.attrib & PA_HYP_TIMES:
        out(b"Hyp times: t1= %.2f t2= %.2f\n" % (path.hyp_t1, path.hyp_t2))

    while used < to:
        # calculate the max words for this pass
        plen = 9 if used > 0 else 6
        i = used
        while plen < maxw and i < to:
            if path.pset[i].eval == P_INS:
                lena = 1
            else:
                lena = len(path.pset[i].a_ptr.value)
            if path.pset[i].eval == P_DEL:
                lenb = 1
            else:
                lenb = len(path.pset[i].b_ptr.value)
            plen += 1 + max(aux_max, lena, lenb)
            i += 1
        if used > 0:
            out(b"\n")
        endpass = i - (1 if i < to else 0)

        for line in range(11):
            if line == 0:
                if used > 0:
                    out(b">> ")
                out(b"REF:  ")
            elif line in (1, 2):
                if not (path.attrib & PA_REF_WTIMES):
                    continue
                if used > 0:
                    out(b">> ")
                out(b"R_T1: " if line == 1 else b"R_T2: ")
            elif line == 3:
                if not (path.attrib & PA_REF_CONF):
                    continue
                if used > 0:
                    out(b">> ")
                out(b"RCNF: ")
            elif line == 4:
                if used > 0:
                    out(b">> ")
                out(b"HYP:  ")
            elif line in (5, 6):
                if not (path.attrib & PA_HYP_WTIMES):
                    continue
                if used > 0:
                    out(b">> ")
                out(b"H_T1: " if line == 5 else b"H_T2: ")
            elif line == 7:
                if not (path.attrib & PA_HYP_CONF):
                    continue
                if used > 0:
                    out(b">> ")
                out(b"CONF: ")
            elif line == 8:
                if not (path.attrib & PA_REF_WEIGHT):
                    continue
                if used > 0:
                    out(b">> ")
                out(b"R_WE: ")
            elif line == 9:
                if not (path.attrib & PA_HYP_WEIGHT):
                    continue
                if used > 0:
                    out(b">> ")
                out(b"H_WE: ")
            elif line == 10:
                if used > 0:
                    out(b">> ")
                out(b"Eval: ")

            # loop through the words
            i = used
            while i < endpass and i < to:
                ps = path.pset[i]
                if ps.eval == P_INS:
                    lena = 1
                    astr = _ASTER
                else:
                    astr = ps.a_ptr.value
                    lena = len(astr)
                if ps.eval == P_DEL:
                    lenb = 1
                    bstr = _ASTER
                else:
                    bstr = ps.b_ptr.value
                    lenb = len(bstr)
                wmax = max(aux_max, lena, lenb)

                if line == 0:
                    if astr is _ASTER:
                        astr = _ASTER[_ALEN - lenb:]
                    else:
                        if not (path.attrib & PA_CASE_SENSE) \
                                and ps.eval != P_CORR:
                            astr = _t.str_to_master(astr, False)
                    out(_ljust(astr, wmax) + b' ')
                elif line in (1, 2):
                    if line == 1 and ps.eval != P_INS:
                        out(_ljust(('%.2f' % ps.a_ptr.T1).encode(), wmax)
                            + b' ')
                    elif line == 2 and ps.eval != P_INS:
                        out(_ljust(('%.2f' % ps.a_ptr.T2).encode(), wmax)
                            + b' ')
                    else:
                        out(_rjust(b'', wmax) + b' ')
                elif line == 3:
                    if ps.eval != P_INS:
                        out(_ljust(('%.4f' % ps.a_ptr.conf).encode(), wmax)
                            + b' ')
                    else:
                        out(_rjust(b'', wmax) + b' ')
                elif line == 4:
                    if bstr is _ASTER:
                        bstr = _ASTER[_ALEN - lena:]
                    else:
                        if not (path.attrib & PA_CASE_SENSE) \
                                and ps.eval != P_CORR:
                            bstr = _t.str_to_master(bstr, False)
                    out(_ljust(bstr, wmax) + b' ')
                elif line in (5, 6):
                    if line == 5 and ps.eval != P_DEL:
                        out(_ljust(('%.2f' % ps.b_ptr.T1).encode(), wmax)
                            + b' ')
                    elif line == 6 and ps.eval != P_DEL:
                        out(_ljust(('%.2f' % ps.b_ptr.T2).encode(), wmax)
                            + b' ')
                    else:
                        out(_rjust(b'', wmax) + b' ')
                elif line == 7:
                    if ps.eval != P_DEL:
                        out(_ljust(('%.4f' % ps.b_ptr.conf).encode(), wmax)
                            + b' ')
                    else:
                        out(_rjust(b'', wmax) + b' ')
                elif line == 8:
                    if ps.eval != P_INS:
                        out(_ljust(('%.4f' % ps.a_ptr.weight).encode(), wmax)
                            + b' ')
                    else:
                        out(_rjust(b'', wmax) + b' ')
                elif line == 9:
                    if ps.eval != P_DEL:
                        out(_ljust(('%.4f' % ps.b_ptr.weight).encode(), wmax)
                            + b' ')
                    else:
                        out(_rjust(b'', wmax) + b' ')
                elif line == 10:
                    if astr is _ASTER:
                        ev = b'I'
                    elif bstr is _ASTER:
                        ev = b'D'
                    elif ps.eval != P_CORR:
                        ev = b'S'
                    else:
                        ev = b''
                    out(_ljust(ev, wmax) + b' ')
                i += 1
            out(b"\n")
        used = endpass


def PATH_print(path, out, maxw):
    PATH_n_print(path, out, 0, path.num, maxw)
