"""Command-line interface replicating sclite (sclite.c main + proc_args).

Supported: -r -h -i -l -f -o -O -n -s -m -F -D -c -e -T
Output types: sum rsum pralign/pra prf all none stdout
Not supported (exits with an error, like sclite without those features):
-d (diff), -S (inferred segmentation), -w (word weights), -L (SLM),
-C (confidence reports), -p/-P (pipes), lur/snt/spk/dtl/sgml outputs.
"""
import os
import sys

from . import text as _t
from . import scores as _sc
from . import align as _align
from . import stmctm as _stmctm

OUT_SUM = 0x0001
OUT_RSUM = 0x0002
OUT_PRALIGN = 0x0004
OUT_LUR = 0x0008
OUT_SGML = 0x0010
OUT_STDOUT = 0x0020
OUT_SENT = 0x0040
OUT_SPKR = 0x0080
OUT_DTL = 0x0100
OUT_PRALIGN_FULL = 0x0200
OUT_WWS = 0x0400
OUT_NL_SGML = 0x0800

REDUCE_NOTHING = 0x0000
REDUCE_REF_SEGMENTS = 0x0001
REDUCE_HYP_WORDS = 0x0002

PROG = 'pysclite'


class Args:
    pass


def _do_exit(msg, ret=1):
    sys.stderr.write('%s: %s\n' % (PROG, msg))
    sys.exit(ret)


def proc_args(argv):
    a = Args()
    a.linewidth = 1000
    a.feedback = 1
    a.hname = []
    a.hfmt = []
    a.title = []
    a.rname = None
    a.rfmt = None
    a.id = None
    a.outputs = 0
    a.case_sense = 0
    a.out_dir = None
    a.out_name = None
    a.char_align = 0
    a.frag_correct = 0
    a.opt_del = 0
    a.stm2ctm_reduce = 0
    a.time_align = 0
    a.left_to_right = 1
    fbset = False
    outset = False

    argc = len(argv)
    opt = 1
    while opt < argc and argv[opt].startswith('-') and argv[opt] != '-':
        o = argv[opt]
        if o == '-r':
            if argc < opt + 2:
                _do_exit('Not enough Ref arguments')
            if argv[opt + 1].startswith('-'):
                _do_exit("Req'd Ref File name")
            opt += 1
            a.rname = argv[opt]
            if argc >= opt + 2 and not argv[opt + 1].startswith('-'):
                opt += 1
                a.rfmt = argv[opt]
        elif o == '-h':
            if argc < opt + 2:
                _do_exit('Not enough Hyp arguments')
            if argv[opt + 1].startswith('-'):
                _do_exit("Req'd Hyp File name")
            opt += 1
            a.hname.append(argv[opt])
            if argc >= opt + 2 and not argv[opt + 1].startswith('-'):
                opt += 1
                a.hfmt.append(argv[opt])
            else:
                a.hfmt.append('trn')
            if argc >= opt + 2 and not argv[opt + 1].startswith('-'):
                opt += 1
                a.title.append(argv[opt])
            else:
                a.title.append(a.hname[-1])
        elif o == '-i':
            if argc <= opt + 1:
                _do_exit('Not enough ID arguments')
            opt += 1
            a.id = argv[opt]
        elif o == '-l':
            if argc <= opt + 1:
                _do_exit('Not enough Line Width arguments')
            opt += 1
            a.linewidth = int(argv[opt])
        elif o == '-f':
            if argc <= opt + 1:
                _do_exit('Not enough Feedback arguments')
            opt += 1
            a.feedback = int(argv[opt])
            fbset = True
        elif o == '-o':
            opt += 1
            if argc < opt + 1:
                _do_exit('Not enough Report arguments')
            while opt < argc and not argv[opt].startswith('-'):
                v = argv[opt]
                if v == 'sum':
                    a.outputs ^= OUT_SUM
                elif v == 'wws':
                    a.outputs ^= OUT_WWS
                elif v == 'rsum':
                    a.outputs ^= OUT_RSUM
                elif v in ('pralign', 'pra'):
                    a.outputs ^= OUT_PRALIGN
                elif v == 'prf':
                    a.outputs ^= OUT_PRALIGN_FULL
                elif v == 'lur':
                    a.outputs ^= OUT_LUR
                elif v == 'stdout':
                    a.outputs ^= OUT_STDOUT
                elif v == 'sgml':
                    a.outputs ^= OUT_SGML
                elif v == 'nl.sgml':
                    a.outputs ^= OUT_NL_SGML
                elif v == 'snt':
                    a.outputs ^= OUT_SENT
                elif v == 'spk':
                    a.outputs ^= OUT_SPKR
                elif v == 'dtl':
                    a.outputs ^= OUT_DTL
                elif v == 'all':
                    a.outputs ^= OUT_PRALIGN + OUT_SUM + OUT_RSUM
                elif v == 'none':
                    a.outputs = 0
                else:
                    sys.stderr.write("Unknown report '%s'\n" % v)
                opt += 1
            if opt < argc and argv[opt].startswith('-'):
                opt -= 1
            outset = True
        elif o == '-O':
            if argc <= opt + 1:
                _do_exit('Output directory not specified')
            opt += 1
            a.out_dir = argv[opt]
        elif o == '-n':
            if argc <= opt + 1:
                _do_exit('Output name not specified')
            opt += 1
            a.out_name = argv[opt]
        elif o == '-d':
            _do_exit('Alignments via diff have been disabled')
        elif o == '-s':
            a.case_sense = 1
        elif o == '-m':
            a.stm2ctm_reduce = REDUCE_NOTHING
            while opt + 1 < argc and not argv[opt + 1].startswith('-'):
                opt += 1
                if argv[opt] == 'ref':
                    a.stm2ctm_reduce |= REDUCE_REF_SEGMENTS
                elif argv[opt] == 'hyp':
                    a.stm2ctm_reduce |= REDUCE_HYP_WORDS
                else:
                    _do_exit("Unrecognized -m option '%s'" % argv[opt])
            if a.stm2ctm_reduce == REDUCE_NOTHING:
                a.stm2ctm_reduce = REDUCE_REF_SEGMENTS
        elif o == '-F':
            a.frag_correct = 1
        elif o == '-D':
            a.opt_del = 1
        elif o == '-c':
            a.char_align = 1  # CALI_ON
            while opt + 1 < argc and not argv[opt + 1].startswith('-'):
                opt += 1
                if argv[opt] == 'NOASCII':
                    a.char_align |= 2  # CALI_NOASCII
                elif argv[opt] == 'DH':
                    a.char_align |= 4  # CALI_DELHYPHEN
                else:
                    _do_exit("Unrecognized character alignment option '%s'"
                             % argv[opt])
        elif o == '-e':
            if opt + 1 >= argc or argv[opt + 1].startswith('-'):
                _do_exit('Argument required for character encoding')
            opt += 1
            if not _t.set_encoding(argv[opt]):
                _do_exit("Unrecognized character encoding option '%s'"
                         % argv[opt])
            if opt + 1 < argc and not argv[opt + 1].startswith('-'):
                if not _t.set_lang_prof(argv[opt + 1]):
                    _do_exit('Optional case conversion localization failed '
                             '/%s/' % argv[opt + 1])
                opt += 1
        elif o == '-T':
            a.time_align = 1
        elif o == '-S':
            _do_exit('Inferred word segmentation (-S) not supported by '
                     'pysclite')
        elif o == '-w':
            _do_exit('Word weight lists (-w) not supported by pysclite')
        elif o == '-L':
            _do_exit('Language model weights (-L) not supported by pysclite')
        elif o == '-C':
            _do_exit('Confidence report outputs (-C) not supported by '
                     'pysclite')
        elif o == '-p' or o == '-P':
            _do_exit('Piped input/output (-p/-P) not supported by pysclite')
        else:
            _do_exit('Illegal argument: %s' % o)
        opt += 1

    if a.outputs == 0 and not outset:
        a.outputs = OUT_SUM + OUT_STDOUT
    if (a.outputs & OUT_STDOUT) and outset and a.feedback > 0 and not fbset:
        a.feedback = 0

    if a.rname is None or len(a.hname) == 0:
        _do_exit('Input not specified, use transcription input')

    if a.rfmt is None:
        a.rfmt = 'trn'
    if a.rfmt not in ('trn', 'ctm', 'tmk', 'stm'):
        _do_exit("Reference file format '%s' not acceptable" % a.rfmt)
    for f in a.hfmt:
        if f not in ('trn', 'ctm', 'tmk', 'txt'):
            _do_exit("Hypothesis file format '%s' not acceptable" % f)
    if a.id is None:
        if a.rfmt == 'trn':
            _do_exit('Required utterance id (option -i) for transcript '
                     'mode')
        for f in a.hfmt:
            if f == 'trn':
                _do_exit('Required utterance id (option -i) for transcript '
                         'mode')
    if a.time_align:
        if not (a.rfmt in ('ctm', 'tmk') and
                all(f in ('ctm', 'tmk') for f in a.hfmt)):
            _do_exit('Time-mediated alignments require CTM files for both '
                     'REF and HYP')
    return a


_ID_TYPES = {
    'sp': _align.SP,
    'rm': _align.RM,
    'wsj': _align.WSJ,
    'swb': _align.SWB,
    'atis': _align.ATIS,
    'spu_id': _align.SPUID,
}


def _write_file_or_stdout(outputs, outroot, ext, writer):
    """Route a report to stdout or <outroot>.<ext>."""
    if outputs & OUT_STDOUT:
        writer(lambda b: sys.stdout.buffer.write(b))
    else:
        fname = '%s.%s' % (outroot, ext)
        try:
            os.remove(fname)
        except OSError:
            pass
        with open(fname, 'wb') as f:
            writer(lambda b: f.write(b))


def main(argv=None):
    if argv is None:
        argv = sys.argv
    if len(argv) <= 1:
        _do_exit('Arguments required')
    a = proc_args(argv)

    idt = _ID_TYPES.get(a.id.lower()) if a.id else None
    if a.id is not None and idt is None:
        _do_exit("ID type '%s' not defined" % a.id)

    for nh in range(len(a.hname)):
        hypname = a.hname[nh]
        if a.feedback >= 1:
            print("Begin alignment of Ref File: '%s' and Hyp File: '%s'"
                  % (a.rname, hypname))
        # output root name
        hroot = hypname.rsplit('/', 1)
        if len(hroot) == 2:
            hdir, hroot = hroot[0], hroot[1]
        else:
            hroot = hypname
            hdir = '.'
        if a.out_dir is not None or (
                (a.out_dir if a.out_dir else hdir) != '.'):
            outroot = '%s/%s' % ((a.out_dir if a.out_dir else hdir),
                                 (a.out_name if a.out_name else hroot))
        else:
            outroot = a.out_name if a.out_name else hroot

        title = a.title[nh]
        if a.rfmt == 'trn' and a.hfmt[nh] == 'trn':
            scor = _align.align_trans_mode_dp(
                a.rname, hypname, title, True, a.case_sense,
                a.char_align, idt, a.frag_correct, a.opt_del)
        elif a.rfmt == 'stm' and a.hfmt[nh] in ('ctm', 'tmk'):
            scor = _stmctm.align_ctm_to_stm_dp(
                a.rname, hypname, title, True, a.case_sense, a.char_align,
                a.frag_correct, a.opt_del,
                (a.stm2ctm_reduce & REDUCE_REF_SEGMENTS) != 0,
                (a.stm2ctm_reduce & REDUCE_HYP_WORDS) != 0,
                a.left_to_right)
            if scor is None:
                sys.stderr.write('%s: Alignment failed.  Exiting.\n' % PROG)
                sys.exit(1)
        elif a.rfmt in ('ctm', 'tmk') and a.hfmt[nh] in ('ctm', 'tmk'):
            scor = _stmctm.align_ctm_to_ctm(
                hypname, a.rname, title, a.frag_correct, a.opt_del,
                a.case_sense, a.time_align, a.left_to_right)
        elif a.rfmt == 'stm' and a.hfmt[nh] == 'txt':
            _do_exit('Alignments via diff have been disabled')
        else:
            sys.stderr.write("Error: Unable to score '%s' against '%s'\n"
                             % (a.hfmt[nh], a.rfmt))
            sys.exit(1)

        if a.outputs & OUT_SUM:
            _write_file_or_stdout(
                a.outputs, outroot, 'sys',
                lambda w: _sc.print_system_summary(
                    scor, lambda s: w(s.encode('latin-1'))))
        if a.outputs & OUT_WWS:
            if scor.weight_ali:
                _write_file_or_stdout(
                    a.outputs, outroot, 'wws',
                    lambda w: _sc.print_system_summary(
                        scor, lambda s: w(s.encode('latin-1')),
                        do_weighted=True))
            else:
                print('    Skipping WWS Report, no word weights supplied.')
        if a.outputs & OUT_RSUM:
            _write_file_or_stdout(
                a.outputs, outroot, 'raw',
                lambda w: _sc.print_system_summary(
                    scor, lambda s: w(s.encode('latin-1')), do_raw=True))
        if a.outputs & (OUT_SENT | OUT_SPKR | OUT_DTL | OUT_LUR
                        | OUT_SGML | OUT_NL_SGML):
            sys.stderr.write(
                'Warning: snt/spk/dtl/lur/sgml/nl.sgml outputs are not '
                'supported by pysclite\n')
        if a.outputs & OUT_PRALIGN:
            _write_file_or_stdout(
                a.outputs, outroot, 'pra',
                lambda w: _sc.dump_SCORES_alignments(scor, w, a.linewidth,
                                                     False))
        if a.outputs & OUT_PRALIGN_FULL:
            _write_file_or_stdout(
                a.outputs, outroot, 'prf',
                lambda w: _sc.dump_SCORES_alignments(scor, w, a.linewidth,
                                                     True))

    if a.feedback >= 1:
        print('\nSuccessful Completion')
    return 0


if __name__ == '__main__':
    sys.exit(main())
