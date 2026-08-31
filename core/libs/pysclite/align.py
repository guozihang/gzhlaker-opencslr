"""Alignment drivers for TRN mode: port of sclite align.c."""
import sys

from . import text as _t
from . import word as _w
from . import network as _net
from . import path as _p
from . import scores as _sc

# id types
SP = 0
RM = 1
WSJ = 2
SWB = 3
ATIS = 4
SPUID = 5


def remove_id(utt):
    """align.c remove_id: returns (text_without_id, id_with_parens).

    The id is the LAST '(...)' group; text is truncated at the '('.
    """
    r = utt.rfind(b')')
    l = utt.rfind(b'(')
    if r < 0 and l < 0:
        return utt, b''
    if r < 0 or l < 0:
        raise ValueError("Unparsable utterance id %s" % utt)
    return utt[:l], utt[l:r + 1]


def extract_speaker(idb, idt):
    """align.c extract_speaker: idb includes the parens."""
    if idt == SP:
        return idb
    if idt in (ATIS, WSJ):
        return idb[1:4]
    if idt in (RM, SPUID, SWB):
        # from id+1 up to first '-' or '_'
        best = -1
        for ch in (b'-', b'_'):
            idx = idb.find(ch)
            if idx >= 0 and (best < 0 or idx < best):
                best = idx
        if best < 0:
            raise ValueError("can't locate speaker in id %s" % idb)
        return idb[1:best]
    raise ValueError("unknown id type")


def _is_comment(buf):
    return len(buf) >= 2 and buf[0:1] == b';' and buf[1:2] == b';'


def _is_comment_info(buf):
    return len(buf) >= 2 and buf[0:1] == b'*' and buf[1:2] == b'*'


def _is_empty(buf):
    i = 0
    n = len(buf)
    while i < n:
        sz = _t.nbytes_of_char(buf, i)
        if sz > 1:
            return False
        if buf[i] not in (0x20, 0x09, 0x0A, 0x0B, 0x0C, 0x0D):
            return False
        i += 1
    return True


def _parse_comment_line(sc, buf):
    """parse_input_comment_line: returns True if a comment line."""
    if not _is_comment(buf):
        return False
    if buf[:9] == b';; LABEL ':
        try:
            q1 = buf.index(b'"')
            q2 = buf.index(b'"', q1 + 1)
            lid = buf[q1 + 1:q2]
            q3 = buf.index(b'"', q2 + 1)
            q4 = buf.index(b'"', q3 + 1)
            title = buf[q3 + 1:q4]
            q5 = buf.index(b'"', q4 + 1)
            q6 = buf.index(b'"', q5 + 1)
            desc = buf[q5 + 1:q6]
            item = (lid, title, desc)
            if not hasattr(sc, 'plab'):
                sc.plab = []
            if item not in sc.plab:
                sc.plab.append(item)
        except ValueError:
            pass
    elif buf[:12] == b';; CATEGORY ':
        try:
            q1 = buf.index(b'"')
            q2 = buf.index(b'"', q1 + 1)
            lid = buf[q1 + 1:q2]
            q3 = buf.index(b'"', q2 + 1)
            q4 = buf.index(b'"', q3 + 1)
            title = buf[q3 + 1:q4]
            q5 = buf.index(b'"', q4 + 1)
            q6 = buf.index(b'"', q5 + 1)
            desc = buf[q5 + 1:q6]
            item = (lid, title, desc)
            if not hasattr(sc, 'cat'):
                sc.cat = []
            if item not in sc.cat:
                sc.cat.append(item)
        except ValueError:
            pass
    return True


def _read_data_lines(fp, sc):
    """Yield non-comment, non-empty lines (with trailing newline kept)."""
    for line in fp:
        if _parse_comment_line(sc, line) or _is_comment_info(line) \
                or _is_empty(line):
            continue
        yield line


def load_refs(sc, hyp_file, ref_file, case_sense):
    """align.c load_refs.  Returns (reftran, refid): sorted ids + texts."""
    # pre-read hypothesis ids
    idset = []
    with open(hyp_file, 'rb') as fp:
        for line in _read_data_lines(fp, sc):
            if not case_sense:
                line = _t.str_to_master(line, True)
            _txt, idb = remove_id(line)
            idset.append(idb)
    idset.sort()

    idpos = {v: i for i, v in enumerate(idset)}
    refset = [None] * len(idset)
    num_ref = 0
    failure = False
    with open(ref_file, 'rb') as fp:
        for line in _read_data_lines(fp, sc):
            if not case_sense:
                line = _t.str_to_master(line, True)
            txt, idb = remove_id(line)
            # bsearch equivalent
            rind = idpos.get(idb)
            if rind is not None:
                if refset[rind] is not None:
                    if txt != refset[rind][:len(txt)]:
                        sys.stderr.write(
                            "Error: double reference text for id '%s'\n"
                            % idb.decode('latin-1'))
                        failure = True
                else:
                    refset[rind] = txt
                    num_ref += 1
    if num_ref != len(idset):
        failure = True
        sys.stderr.write("Error: Not enough Reference files loaded\n"
                         "Missing:\n")
        for i in range(num_ref):
            if refset[i] is None:
                sys.stderr.write("    %s\n" % idset[i].decode('latin-1'))
    if failure:
        raise RuntimeError("load_refs failed")
    return refset, idset


def _decode_opt_del(arc, _p_):
    tw = arc.data
    if tw.value is None:
        return
    n = len(tw.value)
    if n >= 2 and tw.value[n - 1] == _w.WORD_OPT_DEL_POST \
            and tw.value[0] == _w.WORD_OPT_DEL_PRE:
        tw.opt_del = True
        tw.intern_value = tw.value[1:n - 1]


def _decode_fragment(arc, _p_):
    tw = arc.data
    if tw.value is None:
        return
    n = len(tw.value)
    if n >= 1 and (tw.value[n - 1] == _w.WORD_FRAG_CHAR
                   or tw.value[0] == _w.WORD_FRAG_CHAR):
        tw.frag_corr = True
    elif tw.opt_del and n >= 2 and tw.value[n - 2] == _w.WORD_FRAG_CHAR:
        tw.frag_corr = True


def _make_expand_chars(flag):
    def expand_words_to_chars(arc, _p_):
        tw = arc.data
        if tw.value is None or len(tw.value) <= 1:
            return
        src = tw.intern_value if tw.opt_del else tw.value
        chars = _t.separate_chars(src, flag)
        # escape ';'
        chars_esc = chars.replace(b';', b'\\;')
        if chars != tw.value:
            subnet = _net.create_from_TEXT(chars_esc, 'expand-net')
            dur = tw.T_dur / subnet.arc_count
            t1 = tw.T1
            conf = tw.conf
            od = tw.opt_del
            p = subnet.start_node
            # traverse arcs in order setting duration/start
            arcs = []

            def collect(a, _d):
                arcs.append(a)
            _net.traverse(subnet, None, None, collect, None, 0)
            t = t1
            for a in arcs:
                a.data.T_dur = dur
                a.data.T1 = t
                a.data.T2 = a.data.T_dur + a.data.T1
                t += a.data.T_dur
                a.data.conf = conf
                if od:
                    a.data.opt_del = True
                    _decode_opt_del(a, None)
            _net.merge_network(arc.from_node, arc.to_node, subnet)
            _net.delete_arc(arc)
    return expand_words_to_chars


def network_dp_align_texts(ref, rnet, hyp, hnet, char_align, case_sense,
                           idb, fcorr, opt_del, time_align=False):
    """align.c network_dp_align_texts (wwl/lm paths omitted)."""
    if ref is not None:
        refcc = ref if case_sense else _t.str_to_master(ref, True)
        refnet = _net.create_from_TEXT(refcc, 'Reference Net')
    else:
        refnet = rnet
    if hyp is not None:
        hypcc = hyp if case_sense else _t.str_to_master(hyp, True)
        hypnet = _net.create_from_TEXT(hypcc, 'Hypothesis Net')
    else:
        hypnet = hnet

    if opt_del:
        _net.traverse(hypnet, None, None, _decode_opt_del, None, 0)
        _net.traverse(refnet, None, None, _decode_opt_del, None, 0)
    if fcorr:
        _net.traverse(hypnet, None, None, _decode_fragment, None, 0)
        _net.traverse(refnet, None, None, _decode_fragment, None, 0)
    if char_align:
        op = _make_expand_chars(char_align)
        _net.traverse(hypnet, None, None, op, None, 0)
        _net.traverse(refnet, None, None, op, None, 0)

    wwd = _w.wwd_time_WORD if time_align else _w.wwd_WORD
    path = _net.dpalign(refnet, hypnet, wwd, False)
    path.id = idb
    _set_sequence(path)
    if case_sense:
        path.attrib |= _p.PA_CASE_SENSE
    if char_align:
        path.attrib |= _p.PA_CHAR_ALIGN
    return path


_seq_counter = [0]


def _set_sequence(path):
    path.sequence = _seq_counter[0]
    _seq_counter[0] += 1


def reset_sequence():
    _seq_counter[0] = 0


def align_trans_mode_dp(ref_file, hyp_file, set_title, keep_path,
                        case_sense, char_align, idt, fcorr, opt_del):
    """align.c align_trans_mode_dp.  Returns SCORES."""
    scor = _sc.SCORES()
    scor.title = set_title
    scor.ref_fname = ref_file
    scor.hyp_fname = hyp_file
    import time as _time
    scor.creation_date = _time.strftime('%c')
    scor.frag_corr = bool(fcorr)
    scor.opt_del = bool(opt_del)
    scor.weight_ali = False

    reftran, refid = load_refs(scor, hyp_file, ref_file, case_sense)
    idpos = {v: i for i, v in enumerate(refid)}

    with open(hyp_file, 'rb') as fp:
        for line in _read_data_lines(fp, scor):
            txt, hyp_id = remove_id(line)
            if not case_sense:
                hyp_id = _t.str_to_master(hyp_id, True)
            spkr_id = extract_speaker(hyp_id, idt)
            spk = scor.get_grp(spkr_id)
            rind = idpos.get(hyp_id)
            if rind is None:
                raise RuntimeError(
                    "Unable to locate Ref transcript for '%s'"
                    % hyp_id.decode('latin-1'))
            path = network_dp_align_texts(reftran[rind], None, txt, None,
                                          char_align, case_sense, hyp_id,
                                          fcorr, opt_del, False)
            _sc.add_PATH_score(scor, path, spk, keep_path)
    return scor
