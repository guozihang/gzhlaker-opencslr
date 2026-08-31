"""SCORES accumulation and reports: port of sclite scores.c."""
import math

from . import path as _p
from . import rpg as _rpg

M_LOG2E = 1.4426950408889634074


def pct(num, dem):
    return 0.0 if dem == 0 else num / dem * 100.0


class GRP:
    def __init__(self, name):
        self.name = name
        self.corr = 0
        self.ins = 0
        self.del_ = 0
        self.sub = 0
        self.merges = 0
        self.splits = 0
        self.weight_ref = 0.0
        self.weight_corr = 0.0
        self.weight_ins = 0.0
        self.weight_del = 0.0
        self.weight_sub = 0.0
        self.weight_merges = 0.0
        self.weight_splits = 0.0
        self.nsent = 0
        self.serr = 0
        self.path = []


class SCORES:
    def __init__(self):
        self.title = ''
        self.grp = []
        self.num_grp = 0
        self.ref_fname = None
        self.hyp_fname = None
        self.creation_date = None
        self.weight_ali = False
        self.weight_file = ''
        self.frag_corr = False
        self.opt_del = False
        # aset: utterance label/category definitions from input comments
        self.plab = []   # list of (id, title, desc) bytes tuples
        self.cat = []

    def get_grp(self, grpname):
        """SCORES_get_grp: find or append speaker group (first-seen order)."""
        if isinstance(grpname, bytes):
            grpname = grpname.decode('latin-1')
        for i, g in enumerate(self.grp):
            if g.name == grpname:
                return i
        self.grp.append(GRP(grpname))
        self.num_grp += 1
        return self.num_grp - 1


def add_PATH_score(sc, path, g, keep_path=True):
    grp = sc.grp[g]
    err = 0
    for i in range(path.num):
        ps = path.pset[i]
        if ps.eval == _p.P_INS:
            grp.ins += 1
            grp.weight_ins += ps.b_ptr.weight
            err += 1
        elif ps.eval == _p.P_DEL:
            grp.del_ += 1
            grp.weight_del += ps.a_ptr.weight
            grp.weight_ref += ps.a_ptr.weight
            err += 1
        elif ps.eval == _p.P_CORR:
            grp.corr += 1
            grp.weight_ref += ps.a_ptr.weight
            grp.weight_corr += ps.b_ptr.weight
        elif ps.eval == _p.P_SUB:
            grp.sub += 1
            grp.weight_sub += ps.a_ptr.weight + ps.b_ptr.weight
            grp.weight_ref += ps.a_ptr.weight
            err += 1
    grp.nsent += 1
    if err > 0:
        grp.serr += 1
    if keep_path:
        grp.path.append(path)


# ------------------------------------------------------------------ stats

def _calc_stats_int(lst):
    """calc_mean_var_std_dev_Zstat for int list."""
    num = len(lst)
    mean = sum(lst) / num
    srt = sorted(lst)
    if num % 2 == 0:
        median = (srt[num // 2] + srt[num // 2 - 1]) / 2
    else:
        median = float(srt[num // 2])
    tmp = sum((v - mean) ** 2 for v in lst)
    if tmp != 0.0:
        variance = tmp / (num - 1)
        std_dev = math.sqrt(variance)
    else:
        variance = std_dev = 0.0
    return mean, variance, std_dev, median


def _calc_stats_double(lst):
    """calc_mean_var_std_dev_Zstat_double."""
    num = len(lst)
    if num == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean = sum(lst) / num
    if num == 1:
        median = lst[0]
    else:
        srt = sorted(lst)
        if num % 2 == 0:
            median = (srt[num // 2] + srt[num // 2 - 1]) / 2
        else:
            median = srt[num // 2]
    tmp = sum((v - mean) ** 2 for v in lst)
    if tmp != 0.0:
        variance = tmp / (num - 1)
        std_dev = math.sqrt(variance)
    else:
        variance = std_dev = 0.0
    return mean, variance, std_dev, median


def compute_SCORE_nce(sc):
    """Returns (nce_system, nce_arr)."""
    tot_corr = tot_sub = tot_ins = 0
    for g in sc.grp:
        tot_corr += g.corr
        tot_sub += g.sub
        tot_ins += g.ins
    nce_arr = [0.0] * sc.num_grp
    sum_c_sys = 0.0
    sum_i_sys = 0.0
    conf_not_in_range = 0
    for spkr, g in enumerate(sc.grp):
        n_words = 0
        sum_c_spk = 0.0
        sum_i_spk = 0.0
        for path in g.path:
            for wd in range(path.num):
                ps = path.pset[wd]
                if ps.eval != _p.P_DEL and ps.b_ptr.value[:1] != b'':
                    conf = ps.b_ptr.conf
                    if conf < 0.0 or conf > 1.0:
                        conf_not_in_range += 1
                    n_words += 1
                    if conf < 0.0000001:
                        conf = 0.0000001
                    if conf > 0.9999999:
                        conf = 0.9999999
                    if ps.eval == _p.P_CORR:
                        sum_c_spk += math.log(conf) * M_LOG2E
                    else:
                        sum_i_spk += math.log(1.0 - conf) * M_LOG2E
        if n_words > 0:
            phyp_corr = g.corr / (g.sub + g.ins + g.corr)
            H_of_C = -((g.corr * math.log(phyp_corr) * M_LOG2E)
                       + ((g.sub + g.ins)
                          * math.log(1.0 - phyp_corr) * M_LOG2E))
            nce_arr[spkr] = (H_of_C + sum_c_spk + sum_i_spk) / H_of_C
            sum_c_sys += sum_c_spk
            sum_i_sys += sum_i_spk
        else:
            nce_arr[spkr] = 0.0
    phyp_corr = tot_corr / (tot_sub + tot_ins + tot_corr)
    H_of_C = -((tot_corr * math.log(phyp_corr) * M_LOG2E)
               + ((tot_sub + tot_ins) * math.log(1.0 - phyp_corr) * M_LOG2E))
    nce_system = (H_of_C + sum_c_sys + sum_i_sys) / H_of_C
    return nce_system, nce_arr, conf_not_in_range


# ------------------------------------------------------- system summary

def print_system_summary(sc, out, do_sm=False, do_raw=False,
                         do_weighted=False):
    """print_system_summary: writes the report to `out` (callable, str)."""
    F_ROUND = _rpg.F_ROUND
    center = _rpg.center
    SCREEN_WIDTH = 80

    # check for hyp conf attributes
    has_hyp_conf = False
    not_has_hyp_conf = False
    for g in sc.grp:
        for path in g.path:
            if path.attrib & _p.PA_HYP_CONF:
                has_hyp_conf = True
            else:
                not_has_hyp_conf = True
    if has_hyp_conf and not_has_hyp_conf:
        has_hyp_conf = False
        import sys
        sys.stderr.write("Warning: Ignoring confidence scores in output\n")

    num_grp = sc.num_grp
    corr_arr = [0.0] * num_grp
    sub_arr = [0.0] * num_grp
    del_arr = [0.0] * num_grp
    ins_arr = [0.0] * num_grp
    err_arr = [0.0] * num_grp
    serr_arr = [0.0] * num_grp
    spl_arr = [0.0] * num_grp
    mrg_arr = [0.0] * num_grp
    nce_arr = [0.0] * num_grp
    sent_num_arr = [0] * num_grp
    word_num_arr = [0] * num_grp

    tot_corr = tot_sub = tot_del = tot_ins = 0
    tot_ref = 0
    tot_st = tot_st_er = 0
    tot_spl = tot_mrg = tot_word = 0
    char_align = False
    nce_system = 0.0

    for spkr in range(num_grp):
        g = sc.grp[spkr]
        if not do_weighted:
            Trefs = g.sub + g.corr + g.del_ + g.merges + g.splits
        else:
            Trefs = g.weight_ref
        if do_raw or Trefs == 0:
            corr_arr[spkr] = float(g.corr)
            sub_arr[spkr] = float(g.sub)
            ins_arr[spkr] = float(g.ins)
            del_arr[spkr] = float(g.del_)
            mrg_arr[spkr] = float(g.merges)
            spl_arr[spkr] = float(g.splits)
            err_arr[spkr] = float(g.sub + g.ins + g.del_ + g.merges
                                  + g.splits)
        else:
            if not do_weighted:
                corr_arr[spkr] = pct(g.corr, Trefs)
                sub_arr[spkr] = pct(g.sub, Trefs)
                ins_arr[spkr] = pct(g.ins, Trefs)
                del_arr[spkr] = pct(g.del_, Trefs)
                mrg_arr[spkr] = pct(g.merges, Trefs)
                spl_arr[spkr] = pct(g.splits, Trefs)
                err_arr[spkr] = pct(g.sub + g.ins + g.del_ + g.merges
                                    + g.splits, Trefs)
            else:
                corr_arr[spkr] = pct(g.weight_corr, Trefs)
                sub_arr[spkr] = pct(g.weight_sub, Trefs)
                ins_arr[spkr] = pct(g.weight_ins, Trefs)
                del_arr[spkr] = pct(g.weight_del, Trefs)
                mrg_arr[spkr] = pct(g.weight_merges, Trefs)
                spl_arr[spkr] = pct(g.weight_splits, Trefs)
                err_arr[spkr] = pct(g.weight_sub + g.weight_ins
                                    + g.weight_del + g.weight_merges
                                    + g.weight_splits, Trefs)
        if do_raw:
            serr_arr[spkr] = float(g.serr)
        else:
            serr_arr[spkr] = pct(g.serr, g.nsent)
        sent_num_arr[spkr] = g.nsent
        word_num_arr[spkr] = g.corr + g.sub + g.del_

        if not do_weighted:
            tot_corr += g.corr
            tot_sub += g.sub
            tot_del += g.del_
            tot_ins += g.ins
            tot_spl += g.splits
            tot_mrg += g.merges
        else:
            tot_corr += g.weight_corr
            tot_sub += g.weight_sub
            tot_del += g.weight_del
            tot_ins += g.weight_ins
            tot_spl += g.weight_splits
            tot_mrg += g.weight_merges
        tot_ref += Trefs
        tot_st_er += g.serr
        tot_st += g.nsent
        tot_word += word_num_arr[spkr]

        for path in g.path:
            if path.attrib & _p.PA_CHAR_ALIGN:
                char_align = True
                break

    if has_hyp_conf:
        nce_system, nce_arr, _ = compute_SCORE_nce(sc)

    nce_fmt = "%7.3f "
    nce_prec = 3
    Znce_fmt = "%7.3f#"
    Zpct_fmt = "%5.0f*"
    tot_Zpct_fmt = "%5.1f+"
    spkr_fmt = " %s "
    sent_fmt = "%5d"
    if not do_raw:
        pct_fmt = tot_pct_fmt = "%5.1f "
        prec = tprec = 1
    else:
        pct_fmt = "%5.0f "
        prec = 0
        tot_pct_fmt = "%5.1f "
        tprec = 1

    out("\n\n\n%s\n\n" % center("SYSTEM SUMMARY PERCENTAGES by SPEAKER",
                                SCREEN_WIDTH))
    if do_weighted:
        out("\n\n%s\n" % center("*" * 70, SCREEN_WIDTH))
        out("%s\n" % center("*****     Word Percentages Computed using "
                            "Weighted Word Scoring  *****", SCREEN_WIDTH))
        out("%s\n\n" % center("*" * 70, SCREEN_WIDTH))
        out("%s\n\n" % center("** Weights defined by file: '%s'"
                              % sc.weight_file, SCREEN_WIDTH))
    if (not do_sm) and (tot_spl + tot_mrg) > 0:
        out("\nWarning: Split and/or Merges found, but not reported\n")

    rpg = _rpg.Rpg()
    rpg.set_page_center(SCREEN_WIDTH)
    hdr_wrd = "# Chr" if char_align else "# Wrd"
    rpg.add_row_values("c", [sc.title])
    rpg.add_row_separation('-', _rpg.BEFORE_ROW)
    if not has_hyp_conf:
        if do_sm:
            hdr_fmt = "l|cc|cccccccc"
            hdr_vals = [" SPKR", " # Snt", hdr_wrd, "Corr", " Sub", " Del",
                        " Ins", " Mrg", " Spl", " Err", "S.Err"]
        else:
            hdr_fmt = "l|cc|cccccc"
            hdr_vals = [" SPKR", " # Snt", hdr_wrd, "Corr", " Sub", " Del",
                        " Ins", " Err", "S.Err"]
    else:
        if do_sm:
            hdr_fmt = "l|cc|cccccccc|c"
            hdr_vals = [" SPKR", " # Snt", hdr_wrd, "Corr", " Sub", " Del",
                        " Ins", " Mrg", " Spl", " Err", "S.Err", "NCE"]
        else:
            hdr_fmt = "l|cc|cccccc|c"
            hdr_vals = [" SPKR", " # Snt", hdr_wrd, "Corr", " Sub", " Del",
                        " Ins", " Err", "S.Err", "NCE"]
    rpg.add_row_values(hdr_fmt, hdr_vals)

    Zero_spkr = 0
    for spkr in range(num_grp):
        rpg.add_row_separation('-', _rpg.BEFORE_ROW)
        vals = [spkr_fmt % sc.grp[spkr].name,
                sent_fmt % sent_num_arr[spkr],
                sent_fmt % word_num_arr[spkr]]
        if (word_num_arr[spkr] > 0 and not do_raw) or do_raw:
            vals.append(pct_fmt % F_ROUND(corr_arr[spkr], prec))
            vals.append(pct_fmt % F_ROUND(sub_arr[spkr], prec))
            vals.append(pct_fmt % F_ROUND(del_arr[spkr], prec))
            vals.append(pct_fmt % F_ROUND(ins_arr[spkr], prec))
            if do_sm:
                vals.append(pct_fmt % F_ROUND(mrg_arr[spkr], prec))
                vals.append(pct_fmt % F_ROUND(spl_arr[spkr], prec))
            vals.append(pct_fmt % F_ROUND(err_arr[spkr], prec))
        else:
            vals.append(Zpct_fmt % F_ROUND(corr_arr[spkr], 0))
            vals.append(Zpct_fmt % F_ROUND(sub_arr[spkr], 0))
            vals.append(Zpct_fmt % F_ROUND(del_arr[spkr], 0))
            vals.append(Zpct_fmt % F_ROUND(ins_arr[spkr], 0))
            if do_sm:
                vals.append(Zpct_fmt % F_ROUND(mrg_arr[spkr], 0))
                vals.append(Zpct_fmt % F_ROUND(spl_arr[spkr], 0))
            vals.append(Zpct_fmt % F_ROUND(err_arr[spkr], 0))
            Zero_spkr += 1
        vals.append(pct_fmt % F_ROUND(serr_arr[spkr], prec))
        if has_hyp_conf:
            if word_num_arr[spkr] > 0:
                vals.append(nce_fmt % F_ROUND(nce_arr[spkr], nce_prec))
            else:
                vals.append("# ")
        rpg.add_row_values(hdr_fmt, vals)

    rpg.add_row_separation('=', _rpg.BEFORE_ROW)
    # Sum/Avg row (justification for these rows differs)
    if not do_raw:
        vals = [" Sum/Avg", sent_fmt % tot_st, sent_fmt % tot_word,
                tot_pct_fmt % F_ROUND(pct(tot_corr, tot_ref), tprec),
                tot_pct_fmt % F_ROUND(pct(tot_sub, tot_ref), tprec),
                tot_pct_fmt % F_ROUND(pct(tot_del, tot_ref), tprec),
                tot_pct_fmt % F_ROUND(pct(tot_ins, tot_ref), tprec)]
        if do_sm:
            vals.append(tot_pct_fmt % F_ROUND(pct(tot_mrg, tot_ref), tprec))
            vals.append(tot_pct_fmt % F_ROUND(pct(tot_spl, tot_ref), tprec))
        vals.append(tot_pct_fmt % F_ROUND(
            pct(tot_sub + tot_ins + tot_del + tot_spl + tot_mrg, tot_ref),
            tprec))
        vals.append(tot_pct_fmt % F_ROUND(pct(tot_st_er, tot_st), tprec))
        if has_hyp_conf:
            vals.append(nce_fmt % F_ROUND(nce_system, nce_prec))
        rpg.add_row_values(hdr_fmt, vals)
    else:
        vals = [" Sum", sent_fmt % tot_st, sent_fmt % tot_word,
                pct_fmt % F_ROUND(float(tot_corr), prec),
                pct_fmt % F_ROUND(float(tot_sub), prec),
                pct_fmt % F_ROUND(float(tot_del), prec),
                pct_fmt % F_ROUND(float(tot_ins), prec)]
        if do_sm:
            vals.append(pct_fmt % F_ROUND(float(tot_mrg), prec))
            vals.append(pct_fmt % F_ROUND(float(tot_spl), prec))
        vals.append(pct_fmt % F_ROUND(
            float(tot_sub + tot_ins + tot_del + tot_spl + tot_mrg), prec))
        vals.append(pct_fmt % F_ROUND(float(tot_st_er), prec))
        if has_hyp_conf:
            vals.append(nce_fmt % F_ROUND(nce_system, nce_prec))
        rpg.add_row_values(hdr_fmt, vals)

    # remove zero-word speakers for Mean/SD/Median
    if Zero_spkr > 0:
        spkr = 0
        # C: for each spkr, if zero, shift up (loop may re-examine same idx)
        spkr_list = list(range(num_grp))
        # replicate C loop semantics exactly:
        sp = 0
        while sp < num_grp:
            if word_num_arr[sp] == 0:
                for s2 in range(sp, num_grp - 1):
                    corr_arr[s2] = corr_arr[s2 + 1]
                    sub_arr[s2] = sub_arr[s2 + 1]
                    ins_arr[s2] = ins_arr[s2 + 1]
                    del_arr[s2] = del_arr[s2 + 1]
                    err_arr[s2] = err_arr[s2 + 1]
                corr_arr[num_grp - 1] = -1000.0
                sub_arr[num_grp - 1] = -1000.0
                ins_arr[num_grp - 1] = -1000.0
                del_arr[num_grp - 1] = -1000.0
                err_arr[num_grp - 1] = -1000.0
            sp += 1

    mean_sent, _, sd_sent, med_sent = _calc_stats_int(sent_num_arr)
    mean_word, _, sd_word, med_word = _calc_stats_int(word_num_arr)
    n_eff = num_grp - Zero_spkr
    mean_corr, _, sd_corr, med_corr = _calc_stats_double(corr_arr[:n_eff])
    mean_sub, _, sd_sub, med_sub = _calc_stats_double(sub_arr[:n_eff])
    mean_ins, _, sd_ins, med_ins = _calc_stats_double(ins_arr[:n_eff])
    mean_del, _, sd_del, med_del = _calc_stats_double(del_arr[:n_eff])
    mean_spl, _, sd_spl, med_spl = _calc_stats_double(spl_arr[:n_eff])
    mean_mrg, _, sd_mrg, med_mrg = _calc_stats_double(mrg_arr[:n_eff])
    mean_err, _, sd_err, med_err = _calc_stats_double(err_arr[:n_eff])
    mean_serr, _, sd_serr, med_serr = _calc_stats_double(serr_arr)
    mean_nce, _, sd_nce, med_nce = _calc_stats_double(nce_arr)

    rpg.add_row_separation('=', _rpg.BEFORE_ROW)
    stat_fmt = hdr_fmt
    if not has_hyp_conf:
        stat_fmt = "c|cc|cccccccc" if do_sm else "c|cc|cccccc"
    else:
        stat_fmt = "c|cc|cccccccc|c" if do_sm else "c|cc|cccccc|c"
    tpct_fmt = tot_Zpct_fmt if Zero_spkr > 0 else tot_pct_fmt
    nce_row_fmt = nce_fmt if Zero_spkr == 0 else Znce_fmt

    vals = [" Mean ", tot_pct_fmt % F_ROUND(mean_sent, tprec),
            tot_pct_fmt % F_ROUND(mean_word, tprec),
            tpct_fmt % F_ROUND(mean_corr, tprec),
            tpct_fmt % F_ROUND(mean_sub, tprec),
            tpct_fmt % F_ROUND(mean_del, tprec),
            tpct_fmt % F_ROUND(mean_ins, tprec)]
    if do_sm:
        vals.append(tpct_fmt % F_ROUND(mean_spl, tprec))
        vals.append(tpct_fmt % F_ROUND(mean_mrg, tprec))
    vals.append(tpct_fmt % F_ROUND(mean_err, tprec))
    vals.append(tot_pct_fmt % F_ROUND(mean_serr, tprec))
    if has_hyp_conf:
        vals.append(nce_row_fmt % F_ROUND(mean_nce, nce_prec))
    rpg.add_row_values(stat_fmt, vals)

    vals = [" S.D. ", tot_pct_fmt % F_ROUND(sd_sent, tprec),
            tot_pct_fmt % F_ROUND(sd_word, tprec),
            tpct_fmt % F_ROUND(sd_corr, tprec),
            tpct_fmt % F_ROUND(sd_sub, tprec),
            tpct_fmt % F_ROUND(sd_del, tprec),
            tpct_fmt % F_ROUND(sd_ins, tprec)]
    if do_sm:
        vals.append(tpct_fmt % F_ROUND(sd_spl, tprec))
        vals.append(tpct_fmt % F_ROUND(sd_mrg, tprec))
    vals.append(tpct_fmt % F_ROUND(sd_err, tprec))
    vals.append(tot_pct_fmt % F_ROUND(sd_serr, tprec))
    if has_hyp_conf:
        vals.append(nce_row_fmt % F_ROUND(sd_nce, nce_prec))
    rpg.add_row_values(stat_fmt, vals)

    vals = ["Median", tot_pct_fmt % F_ROUND(med_sent, tprec),
            tot_pct_fmt % F_ROUND(med_word, tprec),
            tpct_fmt % F_ROUND(med_corr, tprec),
            tpct_fmt % F_ROUND(med_sub, tprec),
            tpct_fmt % F_ROUND(med_del, tprec),
            tpct_fmt % F_ROUND(med_ins, tprec)]
    if do_sm:
        vals.append(tpct_fmt % F_ROUND(med_spl, tprec))
        vals.append(tpct_fmt % F_ROUND(med_mrg, tprec))
    vals.append(tpct_fmt % F_ROUND(med_err, tprec))
    vals.append(tot_pct_fmt % F_ROUND(med_serr, tprec))
    if has_hyp_conf:
        vals.append(nce_row_fmt % F_ROUND(med_nce, nce_prec))
    rpg.add_row_values(stat_fmt, vals)

    rpg.dump_report(0, out)

    if Zero_spkr and not do_raw:
        out("\n"
            "* No Reference words for this/these speaker(s).  Word counts "
            "supplied\n"
            "  rather than percents.\n"
            "# No Reference words for this/these speaker(s).  NCE not "
            "computable.\n"
            "+ Speaker(s) with no reference data is ignored\n")


# ------------------------------------------------------ alignment dumps

def dump_SCORES_alignments(sc, out, lw, full):
    """dump_SCORES_alignments: out is a callable taking bytes."""
    def s_(x):
        return x if isinstance(x, bytes) else x.encode('latin-1')

    if full:
        out(b"NIST_TEXT_ALIGNMENT\n")
        out(b"VERSION 0.1\n")
    out(b"\n\n\t\tDUMP OF SYSTEM ALIGNMENT STRUCTURE\n\n")
    out(b"System name:   %s\n" % s_(sc.title))
    if full:
        if sc.ref_fname is not None:
            out(b"Ref file:      %s\n" % s_(sc.ref_fname))
        if sc.hyp_fname is not None:
            out(b"Hyp file:      %s\n" % s_(sc.hyp_fname))
        if sc.creation_date is not None:
            out(b"Creation date: \"%s\"\n" % s_(sc.creation_date))
        if sc.weight_ali:
            out(b"Word Weight Aligned by file: \"%s\"\n" % s_(sc.weight_file))
        if sc.frag_corr:
            out(b"Fragment Correct Flag Set\n")
        if sc.opt_del:
            out(b"Optionally Deletable Flag Set\n")
    out(b"\n")
    if full:
        out(b"Speaker Count: %d\n" % sc.num_grp)
    out(b"Speakers: \n")
    for i in range(sc.num_grp):
        out(b"   %2d:  %s\n" % (i, s_(sc.grp[i].name)))
    out(b"\n")

    if full:
        if sc.plab or sc.cat:
            out(b"Utterance Label definitions:\n")
            for (lid, title, desc) in sc.cat:
                out(b"    Category: id: \"%s\" title: \"%s\" "
                    b"description: \"%s\"\n" % (lid, title, desc))
            for (lid, title, desc) in sc.plab:
                out(b"    Label: id: \"%s\" title: \"%s\" "
                    b"description: \"%s\"\n" % (lid, title, desc))
        out(b"\n")
        # first available path
        pathx = None
        for g in sc.grp:
            if len(g.path) > 0:
                pathx = g.path[0]
                break
        if pathx is None:
            out(b"No utterances found\n")
            return
        if pathx.sequence >= 0:
            curpath = [0] * sc.num_grp
            while True:
                minseq = 999999
                nextgrp = -1
                for i in range(sc.num_grp):
                    if curpath[i] < len(sc.grp[i].path):
                        if sc.grp[i].path[curpath[i]].sequence < minseq:
                            minseq = sc.grp[i].path[curpath[i]].sequence
                            nextgrp = i
                if nextgrp < 0:
                    break
                i = nextgrp
                out(b"Speaker sentences%4d:  %s   utt# %d of %d\n"
                    % (i, s_(sc.grp[i].name), curpath[i], sc.grp[i].nsent))
                _p.PATH_print(sc.grp[i].path[curpath[i]], out, lw)
                out(b"\n")
                curpath[nextgrp] += 1
        else:
            for i in range(sc.num_grp):
                out(b"Speaker sentences%4d:  %s   #utts: %d\n"
                    % (i, s_(sc.grp[i].name), sc.grp[i].nsent))
                for path in sc.grp[i].path:
                    _p.PATH_print(path, out, lw)
                    out(b"\n")
    else:
        for i in range(sc.num_grp):
            out(b"Speaker sentences%4d:  %s   #utts: %d\n"
                % (i, s_(sc.grp[i].name), sc.grp[i].nsent))
            for path in sc.grp[i].path:
                attrib = path.attrib
                path.attrib = _p.PA_NONE
                if attrib & _p.PA_CASE_SENSE:
                    path.attrib |= _p.PA_CASE_SENSE
                _p.PATH_print(path, out, lw)
                path.attrib = attrib
                out(b"\n")
    out(b"\n")
