"""STM/CTM parsing and stm<->ctm, ctm<->ctm alignment drivers.

Port of sclite stm.c, fillmrks.c, stm2ctm.c, ctm2ctm.c.
"""
import sys

from . import text as _t
from . import path as _p
from . import scores as _sc
from . import network as _net
from .align import (_parse_comment_line, _is_comment, _is_comment_info,
                    _is_empty, network_dp_align_texts)


# ------------------------------------------------------------------ WTOKE

class WTOKE1:
    __slots__ = ('conv', 'turn', 't1', 'dur', 'sp', 'confidence', 'correct',
                 'bad_marking', 'overlapped', 'comment', 'unsure',
                 'mispronounced', 'crosstalk', 'ignore', 'alternate')

    def __init__(self):
        self.conv = b''
        self.turn = b''
        self.t1 = 0.0
        self.dur = 0.0
        self.sp = b''
        self.confidence = 0.0
        self.correct = -1
        self.bad_marking = False
        self.overlapped = False
        self.comment = False
        self.unsure = False
        self.mispronounced = False
        self.crosstalk = False
        self.ignore = False
        self.alternate = False


class WTOKE:
    """WTOKE_STR1: 1-based word array (word[1..n])."""

    def __init__(self, fname=''):
        self.word = [WTOKE1()]  # word[0] unused
        self.n = 0
        self.s = 1
        self.has_conf = False
        self.id = fname

    def append(self, w):
        self.word.append(w)
        self.n += 1


def fill_mark_struct(fp, wt, case_sense):
    """fillmrks.c fill_mark_struct (whole-file version).

    Returns perr (0 ok, 14 blank-line error is fatal in C).
    """
    for raw in fp:
        if raw[:1] == b';':
            continue
        if not case_sense:
            raw = _t.str_to_master(raw, True)
        if raw.endswith(b'\n'):
            raw = raw[:-1]
        w = WTOKE1()
        # tokenize
        st = _t.StrtokState()
        i = 0
        n = len(raw)
        while i < n and raw[i:i + 1] in (b' ', b'\t'):
            i += 1
        toks = []
        tok = st.strtok(raw[i:], b' \t\n')
        while tok is not None:
            toks.append(tok)
            tok = st.strtok(None, b' \t\n')
        xconv = toks[0] if len(toks) > 0 else b''
        side = toks[1] if len(toks) > 1 else b''
        s2 = toks[2] if len(toks) > 2 else b''
        s3 = toks[3] if len(toks) > 3 else b''
        xsp = toks[4] if len(toks) > 4 else b''
        xconf = toks[5] if len(toks) > 5 else b''
        xcorr = toks[6] if len(toks) > 6 else b''
        if xconv == b'':
            sys.stdout.write("; *ERR: Conversation is empty '%s'.\n"
                             % raw.decode('latin-1'))
            return 14
        if side == b'':
            sys.stdout.write("; *ERR: Conversation side is empty '%s'.\n"
                             % raw.decode('latin-1'))
            return 15
        if s2 == b'':
            sys.stdout.write("; *ERR: Start time is empty '%s'.\n"
                             % raw.decode('latin-1'))
            return 16
        if s3 == b'':
            sys.stdout.write("; *ERR: Duration time is empty '%s'.\n"
                             % raw.decode('latin-1'))
            return 17
        if xsp == b'':
            sys.stdout.write("; *ERR: Word string is empty '%s'.\n"
                             % raw.decode('latin-1'))
            return 18
        if s2[:2] == b'&&':
            w.t1 = _c_atof(s2[2:])
            w.bad_marking = True
        else:
            w.t1 = _c_atof(s2)
            w.bad_marking = False
        w.dur = _c_atof(s3)
        w.turn = side
        if xconf == b'' or xconf.upper() == b'NA':
            w.confidence = 0.0
        else:
            w.confidence = _c_atof(xconf)
            wt.has_conf = True
        w.correct = -1 if xcorr == b'' else _c_atof(xcorr)
        w.sp = xsp
        w.conv = xconv
        # alternate marking
        in_alternate = False
        if wt.n >= 1:
            prev = wt.word[wt.n]
            if prev.alternate and _t.strcasecmp(prev.sp, b'<ALT_END>') != 0:
                in_alternate = True
        if _t.strcmp_master(w.sp, b'<ALT', 4, True) == 0:
            in_alternate = True
        w.alternate = in_alternate
        wt.append(w)
    return 0


def _c_atof(b):
    """C atof semantics: parse leading float prefix, 0.0 on failure."""
    try:
        s = b.decode('latin-1') if isinstance(b, bytes) else b
        s = s.strip()
        if not s:
            return 0.0
        # longest valid float prefix
        import re
        m = re.match(r'^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?', s)
        if not m:
            return 0.0
        return float(m.group(0))
    except (ValueError, OverflowError):
        return 0.0


def locate_WTOKE_boundary(wt, start, by_conv, by_turn):
    """locate_WTOKE_boundary: returns end index (1-based inclusive)."""
    if start == wt.n:
        return start
    limit = 0
    w = start
    while w <= wt.n and limit == 0:
        tchg = True if not by_conv else wt.word[start].conv == wt.word[w].conv
        cchg = True if not by_turn else wt.word[start].turn == wt.word[w].turn
        if not (tchg and cchg):
            limit = w - 1
        w += 1
    if limit == 0:
        limit = wt.n
    return limit


# ------------------------------------------------------------------- STM

class STM_SEG:
    __slots__ = ('file', 'chan', 'spkr', 't1', 't2', 'text', 'labels',
                 'flag1')

    def __init__(self):
        self.file = b''
        self.chan = b''
        self.spkr = b''
        self.t1 = 0.0
        self.t2 = 0.0
        self.text = b''
        self.labels = None
        self.flag1 = 0


class STM:
    """0-based segment array (seg[0..num-1])."""

    def __init__(self):
        self.seg = []
        self.num = 0
        self.s = 0


def _read_stm_line(fp):
    for line in fp:
        if _is_comment(line) or _is_comment_info(line) or _is_empty(line):
            continue
        return line
    return None


def parse_stm_line(seg, buf, case_sense):
    if not buf:
        return
    if not case_sense:
        buf = _t.str_to_master(buf, True)
    st = _t.StrtokState()
    tok = st.strtok(buf, b' \t\n')
    seg.file = tok if tok is not None else None
    tok = st.strtok(None, b' \t\n')
    seg.chan = tok if tok is not None else None
    tok = st.strtok(None, b' \t\n')
    seg.spkr = tok if tok is not None else None
    tok = st.strtok(None, b' \t\n')
    seg.t1 = _c_atof(tok) if tok is not None else 0.0
    rp = st.strtok(None, b' \t\n')
    seg.t2 = _c_atof(rp) if rp is not None else 0.0
    seg.flag1 = 0
    seg.labels = None
    if rp is not None:
        rp2 = st.strtok(None, b' \t\n')
        if rp2 is not None and rp2[:1] == b'<':
            seg.labels = rp2
            rp2 = st.strtok(None, b' \t\n')
        if rp2 is not None:
            # text = rest of line starting at rp2; the C code restores
            # the '\0' strtok wrote after the token to a space, so the
            # text runs from the token start to the end of the buffer
            seg.text = buf[st.last_start:]
        else:
            seg.text = b''
    else:
        seg.text = b''
    i = len(seg.text) - 1
    if i > 0 and seg.text[i:i + 1] == b'\n':
        seg.text = seg.text[:i]


def fill_STM(fp, case_sense):
    stm = STM()
    while True:
        line = _read_stm_line(fp)
        if line is None:
            break
        seg = STM_SEG()
        parse_stm_line(seg, line, case_sense)
        stm.seg.append(seg)
        stm.num += 1
    return stm


def locate_STM_boundary(stm, start, by_file, by_chan):
    if start == stm.num:
        return start
    limit = -1
    w = start
    while w < stm.num and limit == -1:
        tchg = True if not by_file else stm.seg[start].file == stm.seg[w].file
        cchg = True if not by_chan else stm.seg[start].chan == stm.seg[w].chan
        if not (tchg and cchg):
            limit = w - 1
        w += 1
    if limit == -1:
        limit = stm.num
    return limit


# ------------------------------------------------------- stm <-> ctm

def segment_hyp_for_utt(hyp_segs, stm, curhyp, curhend, h_st, h_end,
                        rs, r_st, r_end):
    """Returns (curhyp, curhend) — curhyp unchanged in C (in/out)."""
    if rs == r_end or rs == stm.num - 1:
        curhend = h_end
    else:
        cont = True
        while curhend + 1 <= h_end and cont:
            w = hyp_segs.word[curhend + 1]
            if not w.alternate:
                mid = w.t1 + w.dur / 2.0
            else:
                mid = 0.0
                ch = curhend + 1
                while ch <= h_end and _t.strcasecmp(
                        b'<ALT_END>', hyp_segs.word[ch].sp) != 0:
                    if _t.strcmp_master(hyp_segs.word[ch].sp, b'<ALT', 4,
                                        True) != 0:
                        nmid = (hyp_segs.word[ch].t1
                                + hyp_segs.word[ch].dur / 2.0)
                        if nmid > mid:
                            mid = nmid
                    ch += 1
            if mid >= stm.seg[rs].t2:
                cont = False
            else:
                if not hyp_segs.word[curhend + 1].alternate:
                    curhend += 1
                else:
                    while curhend <= h_end and _t.strcasecmp(
                            b'<ALT_END>', hyp_segs.word[curhend + 1].sp) != 0:
                        curhend += 1
                    if curhend <= h_end:
                        curhend += 1
    return curhyp, curhend


def align_one_channel(scor, chan, hyp_segs, stm, h_st, h_end, r_st, r_end,
                      keep_path, case_sense, char_align, fcorr, opt_del,
                      reduce_ref, reduce_hyp, left_to_right):
    # set flags within the ref and hyp structure to enable reductions
    for rs in range(r_st, min(stm.num - 1, r_end) + 1):
        stm.seg[rs].flag1 = 1

    if reduce_ref:
        for rs in range(r_st, min(stm.num - 1, r_end) + 1):
            if h_end >= h_st:
                if (hyp_segs.word[h_st].t1 >= stm.seg[rs].t2) or \
                   (hyp_segs.word[h_end].t1 + hyp_segs.word[h_end].dur
                    <= stm.seg[rs].t1):
                    stm.seg[rs].flag1 = 0

    if reduce_hyp:
        bt = 9999999.0
        et = -999999.0
        for rs in range(r_st, min(stm.num - 1, r_end) + 1):
            if stm.seg[rs].flag1:
                if bt > stm.seg[rs].t1:
                    bt = stm.seg[rs].t1
                if et < stm.seg[rs].t2:
                    et = stm.seg[rs].t2
        for xx in range(h_st, h_end + 1):
            if _t.strcmp_master(hyp_segs.word[xx].sp, b'<ALT', 4, True) != 0:
                if (hyp_segs.word[xx].t1 > et) or \
                   (hyp_segs.word[xx].t1 + hyp_segs.word[xx].dur < bt):
                    hyp_segs.word[xx].ignore = True
        # if any part of an alternate is legal, the whole alternate is
        xx = h_st
        while xx <= h_end:
            if _t.strcasecmp(b'<ALT_BEGIN>', hyp_segs.word[xx].sp) == 0:
                usable = 0
                yy = xx
                while yy <= h_end and _t.strcasecmp(
                        b'<ALT_END>', hyp_segs.word[yy].sp) != 0:
                    if _t.strcmp_master(hyp_segs.word[yy].sp, b'<ALT', 4,
                                        True) != 0:
                        if not hyp_segs.word[yy].ignore:
                            usable += 1
                    yy += 1
                yy = xx
                while yy <= h_end and _t.strcasecmp(
                        b'<ALT_END>', hyp_segs.word[yy].sp) != 0:
                    hyp_segs.word[yy].ignore = (usable == 0)
                    yy += 1
            xx += 1

    curhyp = h_st
    curhend = h_st - 1
    for rs in range(r_st, stm.num):
        if rs > r_end:
            break
        ignore_segment = False
        if reduce_ref or reduce_hyp:
            if not stm.seg[rs].flag1:
                continue

        curhyp, curhend = segment_hyp_for_utt(hyp_segs, stm, curhyp,
                                              curhend, h_st, h_end,
                                              rs, r_st, r_end)

        txt = stm.seg[rs].text
        # TEXT_strstr is case-insensitive (uses TEXT_strCcasecmp)
        txt_ci = txt.lower()
        if txt_ci.find(b'ignore_time_segment_in_scoring') < 0 and \
                txt_ci.find(b'ignoretimesegmentinscoring') < 0:
            pass
        else:
            ignore_segment = True

        if ignore_segment:
            curhyp = curhend + 1
            continue

        spkr = scor.get_grp(stm.seg[rs].spkr)

        hnet = _net.create_from_WTOKE(hyp_segs, curhyp, curhend,
                                      'Hypothesis net', left_to_right)
        idb = b'(%s-%03d)' % (stm.seg[rs].spkr, scor.grp[spkr].nsent)
        path = network_dp_align_texts(txt, None, None, hnet,
                                      char_align, case_sense, idb,
                                      fcorr, opt_del, False)
        _sc.add_PATH_score(scor, path, spkr, keep_path)
        if stm.seg[rs].labels is not None:
            path.labels = stm.seg[rs].labels
        path.file = stm.seg[rs].file
        path.channel = stm.seg[rs].chan
        if case_sense:
            path.attrib |= _p.PA_CASE_SENSE
        if char_align:
            path.attrib |= _p.PA_CHAR_ALIGN
        path.attrib |= _p.PA_HYP_WTIMES
        path.attrib |= _p.PA_REF_TIMES
        path.ref_t1 = stm.seg[rs].t1
        path.ref_t2 = stm.seg[rs].t2
        if hyp_segs.has_conf:
            path.attrib |= _p.PA_HYP_CONF
        curhyp = curhend + 1
    stm.s = r_end + 1
    return 1


def align_ctm_to_stm_dp(ref_file, hyp_file, set_title, keep_path,
                        case_sense, char_align, fcorr, opt_del,
                        reduce_ref, reduce_hyp, left_to_right):
    """stm2ctm.c align_ctm_to_stm_dp.  Returns SCORES or None on error."""
    scor = _sc.SCORES()
    scor.title = set_title
    scor.ref_fname = ref_file
    scor.hyp_fname = hyp_file
    import time as _time
    scor.creation_date = _time.strftime('%c')
    scor.frag_corr = bool(fcorr)
    scor.opt_del = bool(opt_del)

    with open(hyp_file, 'rb') as fp:
        hyp_segs = WTOKE(hyp_file)
        perr = fill_mark_struct(fp, hyp_segs, case_sense)
    if perr != 0:
        sys.stdout.write("; *Err: Error detected in ctm file '%s'\n"
                         % hyp_file)
        raise RuntimeError('CTM parse error')
    with open(ref_file, 'rb') as fp:
        stm = fill_STM(fp, case_sense)

    # load comment labels
    _load_comment_labels(scor, ref_file)
    _load_comment_labels(scor, hyp_file)

    i = 0
    while (hyp_segs.s <= hyp_segs.n) or (stm.s < stm.num):
        hend = locate_WTOKE_boundary(hyp_segs, hyp_segs.s, True, True) \
            if hyp_segs.s <= hyp_segs.n else hyp_segs.s - 1
        rend = locate_STM_boundary(stm, stm.s, True, True) \
            if stm.s < stm.num else stm.s

        if hyp_segs.s > hyp_segs.n:
            sys.stderr.write(
                "align_ctm_to_stm: Hyp files ends before ref but continuing "
                "for ref file/channel '%s' '%s'.\n"
                % (stm.seg[stm.s].file.decode('latin-1'),
                   stm.seg[stm.s].chan.decode('latin-1')))
        elif hyp_segs.s <= hyp_segs.n and stm.s > stm.num:
            sys.stderr.write(
                "align_ctm_to_stm: Error: Hyp file has more data than ref "
                "file, beginning at hyp file/channel '%s' '%s'.\n"
                % (hyp_segs.word[hyp_segs.s].conv.decode('latin-1'),
                   hyp_segs.word[hyp_segs.s].turn.decode('latin-1')))
            return None
        elif stm.s < stm.num and \
                hyp_segs.word[hyp_segs.s].conv == stm.seg[stm.s].file and \
                hyp_segs.word[hyp_segs.s].turn == stm.seg[stm.s].chan:
            pass  # align this channel as-is
        else:
            sys.stderr.write(
                "align_ctm_to_stm: File identifiers do not match but "
                "continuing. ref file/channel '%s' '%s', next hyp '%s' "
                "'%s'.\n"
                % (stm.seg[stm.s].file.decode('latin-1'),
                   stm.seg[stm.s].chan.decode('latin-1'),
                   hyp_segs.word[hyp_segs.s].conv.decode('latin-1'),
                   hyp_segs.word[hyp_segs.s].turn.decode('latin-1')))
            hend = hyp_segs.s - 1  # align ref to nothing

        if align_one_channel(scor, i, hyp_segs, stm, hyp_segs.s, hend,
                             stm.s, rend, keep_path, case_sense, char_align,
                             fcorr, opt_del, reduce_ref, reduce_hyp,
                             left_to_right) != 1:
            return None
        hyp_segs.s = hend + 1
        stm.s = rend + 1
    return scor


def _load_comment_labels(scor, fname):
    try:
        with open(fname, 'rb') as fp:
            for line in fp:
                _parse_comment_line(scor, line)
    except OSError:
        pass


# ------------------------------------------------------- ctm <-> ctm

def overlap(s1_t1, s1_t2, s2_t1, s2_t2):
    if s1_t2 < s2_t1:
        rval = s1_t2 - s2_t1
    if s1_t1 > s2_t2:
        rval = s2_t2 - s1_t1
    if s1_t1 < s2_t1:
        if s1_t2 < s2_t2:
            rval = s1_t2 - s2_t1
        else:
            rval = s2_t2 - s2_t1
    else:
        if s1_t2 < s2_t2:
            rval = s1_t2 - s1_t1
        else:
            rval = s2_t2 - s1_t1
    return rval


def chop_WTOKE_2(ref, hyp, Rstart, Hstart, Rendconv, Hendconv, max_words):
    """Returns (ok, Rret_end, Hret_end)."""
    if Rstart > Rendconv and Hstart > Hendconv:
        return 0, 0, 0
    if (Rendconv - Rstart) < max_words and (Hendconv - Hstart) < max_words:
        return 1, Rendconv, Hendconv

    Rend = min(Rstart + max_words, Rendconv)
    Hend = min(Hstart + max_words, Hendconv)

    while Rend >= Rstart and ref.word[Rend].alternate:
        Rend -= 1
    skipped_alt = True
    while skipped_alt:
        skipped_alt = False
        if ref.word[Rend].t1 > hyp.word[Hend].t1:
            while (not skipped_alt and Rend >= Rstart
                   and ref.word[Rend].t1 > hyp.word[Hend].t1
                   and overlap(ref.word[Rend].t1,
                               ref.word[Rend].t1 + ref.word[Rend].dur,
                               hyp.word[Hend].t1,
                               hyp.word[Hend].t1 + hyp.word[Hend].dur) < 0.0):
                Rend -= 1
                while Rend >= Rstart and ref.word[Rend].alternate:
                    Rend -= 1
                    skipped_alt = True
        elif ref.word[Rend].t1 < hyp.word[Hend].t1:
            while (Hend >= Hstart
                   and ref.word[Rend].t1 < hyp.word[Hend].t1
                   and overlap(ref.word[Rend].t1,
                               ref.word[Rend].t1 + ref.word[Rend].dur,
                               hyp.word[Hend].t1,
                               hyp.word[Hend].t1 + hyp.word[Hend].dur) < 0.0):
                Hend -= 1

    # backward search
    while Rend > Rstart and Hend > Hstart:
        if Rend >= Rstart:
            Rgap_t1 = ref.word[Rend].t1 + ref.word[Rend].dur
            Rgap_t2 = 999999.99 if Rend == Rendconv else ref.word[Rend + 1].t1
        else:
            Rgap_t1 = 0.0
            Rgap_t2 = ref.word[Rstart].t1
        if Hend >= Hstart:
            Hgap_t1 = hyp.word[Hend].t1 + hyp.word[Hend].dur
            Hgap_t2 = 999999.99 if Hend == Hendconv else hyp.word[Hend + 1].t1
        else:
            Hgap_t1 = 0.0
            Hgap_t2 = hyp.word[Hstart].t1

        if overlap(Rgap_t1, Rgap_t2, Hgap_t1, Hgap_t2) < 0.0:
            skipped_alt = True
            while skipped_alt:
                skipped_alt = False
                if ref.word[Rend].t1 > hyp.word[Hend].t1:
                    if Rend > Rstart:
                        Rend -= 1
                        while Rend >= Rstart and ref.word[Rend].alternate:
                            Rend -= 1
                            skipped_alt = True
                    while (not skipped_alt and Hend >= Hstart
                           and ref.word[Rend].t1 < hyp.word[Hend].t1
                           and overlap(ref.word[Rend].t1,
                                       ref.word[Rend].t1
                                       + ref.word[Rend].dur,
                                       hyp.word[Hend].t1,
                                       hyp.word[Hend].t1
                                       + hyp.word[Hend].dur) < 0.0):
                        Hend -= 1
                elif ref.word[Rend].t1 <= hyp.word[Hend].t1:
                    if Hend > Hstart:
                        Hend -= 1
                    while (not skipped_alt and Rend >= Rstart
                           and ref.word[Rend].t1 > hyp.word[Hend].t1
                           and overlap(ref.word[Rend].t1,
                                       ref.word[Rend].t1
                                       + ref.word[Rend].dur,
                                       hyp.word[Hend].t1,
                                       hyp.word[Hend].t1
                                       + hyp.word[Hend].dur) < 0.0):
                        Rend -= 1
                        while Rend >= Rstart and ref.word[Rend].alternate:
                            Rend -= 1
                            skipped_alt = True
        else:
            return 1, Rend, Hend

    if Rend > Rstart or Hend > Hstart:
        return 1, Rend, Hend
    return 0, 0, 0


def align_ctm_to_ctm(hyp_file, ref_file, set_title, fcorr, opt_del,
                     case_sense, time_align, left_to_right):
    """ctm2ctm.c align_ctm_to_ctm.  Returns SCORES."""
    scor = _sc.SCORES()
    scor.title = set_title
    scor.ref_fname = ref_file
    scor.hyp_fname = hyp_file
    import time as _time
    scor.creation_date = _time.strftime('%c')
    scor.frag_corr = bool(fcorr)
    scor.opt_del = bool(opt_del)

    with open(hyp_file, 'rb') as fp:
        hyp = WTOKE(hyp_file)
        perr = fill_mark_struct(fp, hyp, case_sense)
        if perr != 0:
            sys.stdout.write("; *Err: Error detected in ctm file '%s'\n"
                             % hyp_file)
            raise RuntimeError('CTM parse error')
    with open(ref_file, 'rb') as fp:
        ref = WTOKE(ref_file)
        perr = fill_mark_struct(fp, ref, case_sense)
        if perr != 0:
            sys.stdout.write("; *Err: Error detected in ctm file '%s'\n"
                             % ref_file)
            raise RuntimeError('CTM parse error')

    while hyp.s <= hyp.n or ref.s <= ref.n:
        if hyp.n < hyp.s and ref.n > ref.s:
            raise RuntimeError(
                "premature EOF in hyp file '%s' at conversation '%s' of ref "
                "file '%s'" % (hyp_file, ref.word[ref.s].conv, ref_file))
        if hyp.n > hyp.s and ref.n < ref.s:
            raise RuntimeError(
                "premature EOF in ref file '%s' at conversation '%s' of hyp "
                "file '%s'" % (ref_file, hyp.word[hyp.s].conv, hyp_file))

        ref_conv_end = locate_WTOKE_boundary(ref, ref.s, True, False)
        hyp_conv_end = locate_WTOKE_boundary(hyp, hyp.s, True, False)

        if ref.word[ref.s].conv != hyp.word[hyp.s].conv:
            raise RuntimeError(
                "file strings out of synchronization around ref-file='%s' "
                "hyp-file='%s'" % (ref.word[ref.s].conv, hyp.word[hyp.s].conv))

        ref_end_chan1 = locate_WTOKE_boundary(ref, ref.s, True, True)
        hyp_end_chan1 = locate_WTOKE_boundary(hyp, hyp.s, True, True)

        number_of_channels = 1
        hyp_end_chan2 = ref_end_chan2 = -1
        if ref_end_chan1 + 1 < ref.n and \
                ref.word[ref.s].conv == ref.word[ref_end_chan1 + 1].conv:
            ref_end_chan2 = locate_WTOKE_boundary(ref, ref_end_chan1 + 1,
                                                  True, True)
            number_of_channels = 2
        if hyp_end_chan1 + 1 < hyp.n and \
                hyp.word[hyp.s].conv == hyp.word[hyp_end_chan1 + 1].conv:
            hyp_end_chan2 = locate_WTOKE_boundary(hyp, hyp_end_chan1 + 1,
                                                  True, True)
            number_of_channels = 2
        if number_of_channels == 2:
            if hyp_end_chan2 == -1:
                raise RuntimeError("Hyp is missing second channel of file "
                                   "'%s'" % ref.word[ref.s].conv)
            if ref_end_chan2 == -1:
                raise RuntimeError("ref is missing second channel of file "
                                   "'%s'" % hyp.word[hyp.s].conv)
        else:
            ref_end_chan2 = ref_end_chan1
            hyp_end_chan2 = hyp_end_chan1

        for i in range(number_of_channels):
            chunk = 0
            if i == 0:
                ref_end = ref_end_chan1
                hyp_end = hyp_end_chan1
                ref_begin = ref.s
                hyp_begin = hyp.s
            else:
                ref_end = ref_end_chan2
                hyp_end = hyp_end_chan2
                ref_begin = ref_end_chan1 + 1
                hyp_begin = hyp_end_chan1 + 1
            while ref_begin <= ref_end or hyp_begin <= hyp_end:
                base_len = 50
                while True:
                    ok, ref_chop_end, hyp_chop_end = chop_WTOKE_2(
                        ref, hyp, ref_begin, hyp_begin, ref_end, hyp_end,
                        base_len)
                    if ok:
                        break
                    base_len *= 2

                uttid = b'(%s-%s-%04d)' % (ref.word[ref_begin].conv,
                                           ref.word[ref_begin].turn, chunk)
                spkr = scor.get_grp(b'%s-%s' % (ref.word[ref_begin].conv,
                                                ref.word[ref_begin].turn))

                hnet = _net.create_from_WTOKE(hyp, hyp_begin, hyp_chop_end,
                                              'Hypothesis net',
                                              left_to_right)
                rnet = _net.create_from_WTOKE(ref, ref_begin, ref_chop_end,
                                              'Reference net', left_to_right)
                path = network_dp_align_texts(None, rnet, None, hnet, False,
                                              case_sense, uttid, fcorr,
                                              opt_del, time_align)
                path.attrib |= _p.PA_HYP_WTIMES
                path.attrib |= _p.PA_REF_WTIMES
                if ref.has_conf:
                    path.attrib |= _p.PA_REF_CONF
                if hyp.has_conf:
                    path.attrib |= _p.PA_HYP_CONF
                _p.sort_PATH_time_marks(path)
                _sc.add_PATH_score(scor, path, spkr, True)
                path.file = ref.word[ref_begin].conv
                path.channel = ref.word[ref_begin].turn

                ref_begin = ref_chop_end + 1
                hyp_begin = hyp_chop_end + 1
                chunk += 1

        hyp.s = hyp_conv_end + 1
        ref.s = ref_conv_end + 1

    return scor
