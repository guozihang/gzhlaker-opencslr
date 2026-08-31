"""Report Generation Package: port of sclite rpg.c (Desc engine)."""
import struct

BEFORE_ROW = 0
AFTER_ROW = 1

UNLOCKED = 0
LOCKED = 1

PAGE_WIDTH = 80
CENTER_PAGE = False

SCREEN_WIDTH = 80

_SPACES = ' ' * 300


def F_ROUND(n, p):
    """sctk.h F_ROUND macro (note the C 32-bit float casts)."""
    scale = 10.0 ** p
    v = int(n * scale + (0.5 if n > 0 else -0.5))
    f = struct.unpack('f', struct.pack('f', float(v)))[0]
    s32 = struct.unpack('f', struct.pack('f', scale))[0]
    return struct.unpack('f', struct.pack('f', f / s32))[0]


def center(s, length):
    """pad.c center()."""
    il = len(s)
    ft = (length - il) // 2
    bk = length - (il + ft)
    if ft < 0:
        ft = 0
    if bk < 0:
        bk = 0
    return ' ' * ft + s + ' ' * bk


class ColDef:
    def __init__(self):
        self.format_str = ''
        self.col_just = []
        self.col_lock = []
        self.col_inter_space = []
        self.num_col = 0
        self.num_col_chars = 0
        self.num_locked = 0
        self.min_just_width = 0


class Rpg:
    def __init__(self):
        self.width = 80
        self.tot_num_col = 0
        self.num_col_defs = 0
        self.tot_num_row = 0
        self.col_defs = []
        self.before_row_separations = []
        self.after_row_separations = []
        self.row_just = []
        self.cell_values = {}
        self.max_col_sizes = {}

    # ------------------------------------------------------------- setup
    def erase(self):
        global PAGE_WIDTH, CENTER_PAGE
        self.width = 80
        self.tot_num_col = 0
        self.num_col_defs = 0
        self.tot_num_row = 0
        self.col_defs = []
        self.before_row_separations = []
        self.after_row_separations = []
        self.row_just = []
        self.cell_values = {}
        self.max_col_sizes = {}
        # note: C keeps PAGE_WIDTH/CENTER_PAGE across Desc_erase

    def set_page_center(self, width):
        global PAGE_WIDTH, CENTER_PAGE
        PAGE_WIDTH = width
        CENTER_PAGE = True

    def set_justification(self, just_str):
        for j, cd in enumerate(self.col_defs):
            if cd.format_str == just_str:
                return j
        cd = ColDef()
        cd.format_str = just_str
        col = 0
        p = 0
        n = len(just_str)
        while p < n and just_str[p] != ':':
            ch = just_str[p]
            if ch in 'cClLrRna':
                cd.col_just.append(ch)
                cd.col_lock.append(UNLOCKED)
                cd.col_inter_space.append(' ')
                cd.num_col += 1
                p += 1
            if p < n and just_str[p] == '|':
                cd.col_inter_space[col] = '|'
                p += 1
            elif p < n and just_str[p] == '=':
                cd.col_inter_space[col] = '='
                cd.num_col_chars += 1
                p += 1
            elif p < n and just_str[p] not in 'cClLrRna':
                # else branch: col_inter_space[col] = ' ' (already default)
                pass
            col += 1
        self.col_defs.append(cd)
        j = len(self.col_defs) - 1
        self.num_col_defs += 1
        if col > self.tot_num_col:
            self.tot_num_col = col
        cd.num_locked = 0
        if p < n and just_str[p] == ':':
            p += 1
        col = 0
        while p < n:
            if just_str[p] == 'l':
                cd.col_lock[col] = LOCKED
                cd.num_locked += 1
            col += 1
            p += 1
        return j

    def add_row_separation(self, chr_, row_attach):
        if row_attach == BEFORE_ROW:
            while len(self.before_row_separations) <= self.tot_num_row:
                self.before_row_separations.append('')
            self.before_row_separations[self.tot_num_row] += chr_
        else:
            while len(self.after_row_separations) <= self.tot_num_row:
                self.after_row_separations.append('')
            self.after_row_separations[self.tot_num_row] += chr_

    def add_row_values(self, format_, values):
        just = self.set_justification(format_)
        self.row_just.append(just)
        r = self.tot_num_row
        vi = 0
        cd = self.col_defs[just]
        for col in range(cd.num_col):
            if cd.col_just[col] != 'a':
                self.cell_values[(col, r)] = values[vi]
                vi += 1
            else:
                self.cell_values.setdefault((col, r), '')
        self.tot_num_row += 1
        while len(self.before_row_separations) <= self.tot_num_row:
            self.before_row_separations.append('')
        while len(self.after_row_separations) <= self.tot_num_row:
            self.after_row_separations.append('')

    # iterated interface
    def flush_iterated_row(self, format_, values):
        self.add_row_values(format_, values)

    # ------------------------------------------------------------- dump
    @staticmethod
    def _cis(cd, i):
        """col_inter_space with C's zero-padded array semantics."""
        if 0 <= i < len(cd.col_inter_space):
            return cd.col_inter_space[i]
        return '\0'

    def _num_span_col(self, just, col):
        cd = self.col_defs[just]
        if col == cd.num_col - 1:
            return 0
        c = col + 1
        span = 0
        while c < cd.num_col and cd.col_just[c] == 'a':
            c += 1
            span += 1
        return span

    def _is_last_just_col(self, just, col):
        cd = self.col_defs[just]
        if col == cd.num_col - 1:
            return True
        c = col + 1
        while c < cd.num_col and cd.col_just[c] == 'a':
            c += 1
        return c == cd.num_col

    @staticmethod
    def _measure_words(value, upper_just):
        """Max word length in a cell value (pass-1 measure)."""
        p = 0
        n = len(value)
        best = 0
        while p < n:
            x = 0
            if upper_just:
                while p < n and value[p] != ' ' and value[p:p + 2] != '//':
                    x += 1
                    p += 1
            else:
                while p < n and value[p:p + 2] != '//':
                    x += 1
                    p += 1
            if value[p:p + 2] == '//':
                p += 2
            if p < n and value[p] == ' ':
                p += 1
            if x > best:
                best = x
        return best

    def dump_report(self, space_pad, out):
        """Desc_dump_report.  `out` is a callable taking str."""
        self.max_col_sizes = {}
        # first pass: rows without spanning columns
        for r in range(self.tot_num_row):
            cd = self.col_defs[self.row_just[r]]
            for c in range(cd.num_col):
                if ((c < cd.num_col - 1 and cd.col_just[c + 1] != 'a')
                        or c == cd.num_col - 1):
                    val = self.cell_values.get((c, r), '')
                    x = self._measure_words(val, cd.col_just[c].isupper())
                    key = (cd.num_col, c)
                    if x > self.max_col_sizes.get(key, 0):
                        self.max_col_sizes[key] = x
        # second pass: spanning columns
        for r in range(self.tot_num_row):
            cd = self.col_defs[self.row_just[r]]
            c = 0
            while c < cd.num_col:
                if c < cd.num_col - 1 and cd.col_just[c + 1] == 'a':
                    val = self.cell_values.get((c, r), '')
                    siz = self._measure_words(val, cd.col_just[c].isupper())
                    startc = c
                    span_siz = 0
                    while c < cd.num_col - 1 and cd.col_just[c + 1] == 'a':
                        span_siz += (self.max_col_sizes.get((cd.num_col, c), 0)
                                     + (1 if c < cd.num_col - 1 else 0)
                                     + space_pad * 2)
                        c += 1
                    span_siz += self.max_col_sizes.get((cd.num_col, c), 0)
                    if siz > span_siz:
                        num_unlocked = 0
                        for c2 in range(startc, c + 1):
                            if cd.col_lock[c2] == UNLOCKED:
                                num_unlocked += 1
                        if num_unlocked != 0:
                            for c2 in range(startc, c + 1):
                                if cd.col_lock[c2] == UNLOCKED:
                                    if c2 != c:
                                        add = int(F_ROUND(
                                            (siz - span_siz) / num_unlocked,
                                            0))
                                    else:
                                        add = siz - span_siz
                                    key = (cd.num_col, c2)
                                    self.max_col_sizes[key] = \
                                        self.max_col_sizes.get(key, 0) + add
                                    span_siz += add
                c += 1
        # compute widths
        self.width = 0
        for cd in self.col_defs:
            w = 2
            for c2 in range(cd.num_col):
                w += self.max_col_sizes.get((cd.num_col, c2), 0) \
                    + space_pad * 2
            w += cd.num_col_chars + cd.num_col - 1
            cd.min_just_width = w
            if self.width < w:
                self.width = w

        pad0 = ''
        if CENTER_PAGE and self.width < PAGE_WIDTH:
            pad0 = ' ' * ((PAGE_WIDTH - self.width) // 2)
        out(pad0 + ',' + '-' * (self.width - 2) + '.\n')

        for r in range(self.tot_num_row):
            current_just = self.row_just[r]
            cd = self.col_defs[current_just]
            current_num_col = cd.num_col
            current_num_unlocked = current_num_col - cd.num_locked
            current_underage = self.width - cd.min_just_width

            desc_column_ptr = [''] * current_num_col
            desc_column_size = [0] * current_num_col
            desc_column_text_size = [0] * current_num_col

            current_add = 0
            for c in range(current_num_col):
                if cd.col_just[c] != 'a':
                    desc_column_ptr[c] = self.cell_values.get((c, r), '')
                    if cd.col_lock[c] == LOCKED:
                        desc_column_text_size[c] = self.max_col_sizes.get(
                            (current_num_col, c), 0)
                        desc_column_size[c] = (space_pad * 2) + \
                            desc_column_text_size[c]
                    else:
                        size_adjustment = 0
                        if c != current_num_col - 1:
                            size_adjustment = int(F_ROUND(
                                current_underage / current_num_unlocked
                                * (c + self._num_span_col(current_just, c)
                                   + 1), 0)) - current_add
                            current_add += size_adjustment
                        else:
                            size_adjustment += current_underage - current_add
                        desc_column_text_size[c] = self.max_col_sizes.get(
                            (current_num_col, c), 0)
                        desc_column_size[c] = (space_pad * 2) + \
                            desc_column_text_size[c] + size_adjustment
                        c2 = c + 1
                        while c2 < current_num_col \
                                and cd.col_just[c2] == 'a':
                            if c != current_num_col - 1:
                                size_adjustment = int(F_ROUND(
                                    current_underage / current_num_col
                                    * (c + self._num_span_col(current_just, c)
                                       + 1), 0)) - current_add
                                current_add += size_adjustment
                            else:
                                size_adjustment += (current_underage
                                                    - current_add)
                            desc_column_size[c] += \
                                self.max_col_sizes.get((current_num_col, c2),
                                                       0) \
                                + space_pad * 2 + 1 + size_adjustment
                            desc_column_text_size[c] += \
                                self.max_col_sizes.get((current_num_col, c2),
                                                       0) \
                                + space_pad * 2 + 1
                            c2 += 1
                else:
                    desc_column_ptr[c] = ''
                    desc_column_text_size[c] = 0
                    desc_column_size[c] = 0

            # before-row separations
            for sep in self.before_row_separations[r]:
                line = ['|']
                for c in range(current_num_col):
                    if cd.col_just[c] != 'a':
                        line.append(sep * desc_column_size[c])
                        if not self._is_last_just_col(current_just, c):
                            sp = self._num_span_col(current_just, c)
                            cchr = self._cis(cd, sp + c)
                            lcchr = '\0'
                            if r > 0:
                                lcchr = self._cis(
                                    self.col_defs[self.row_just[r - 1]],
                                    sp + c)
                            if cchr != ' ':
                                if sep == ' ':
                                    if cchr == '=':
                                        line.append('||')
                                    else:
                                        line.append(cchr)
                                elif (cchr in '|=' and sep == '-'
                                      and r > 0 and lcchr in '|='):
                                    line.append('+')
                                    if cchr == '=':
                                        line.append('+')
                                else:
                                    line.append(sep)
                                    if cchr == '=':
                                        line.append(sep)
                            else:
                                if cchr == '=':
                                    line.append(sep)
                                line.append(sep)
                line.append('|\n')
                out(pad0 + ''.join(line))

            # row content (with word wrapping)
            row_not_done = True
            while row_not_done:
                row_not_done = False
                parts = ['|']
                for c in range(current_num_col):
                    column_width = desc_column_size[c]
                    text_width = desc_column_text_size[c]
                    if cd.col_just[c] != 'a':
                        j = cd.col_just[c]
                        pad1 = (column_width - text_width) // 2
                        pad2 = (column_width - text_width) - pad1
                        val, rest = _get_next_string_value(
                            desc_column_ptr[c], text_width)
                        desc_column_ptr[c] = rest
                        if j in 'cC':
                            cell = ' ' * pad1 + center(val, text_width) \
                                + ' ' * pad2
                        elif j in 'rR':
                            cell = ' ' * pad1 + val.rjust(text_width) \
                                + ' ' * pad2
                        elif j in 'lL':
                            cell = ' ' * pad1 + val.ljust(text_width) \
                                + ' ' * pad2
                        else:
                            cell = 'undefined inter column space\n'
                        parts.append(cell)
                        if rest != '':
                            row_not_done = True
                        if not self._is_last_just_col(current_just, c):
                            sp = self._num_span_col(current_just, c)
                            cs = self._cis(cd, sp + c)
                            if cs != ' ':
                                if cs == '=':
                                    parts.append('||')
                                else:
                                    parts.append(cs)
                            else:
                                parts.append(' ')
                parts.append('|\n')
                out(pad0 + ''.join(parts))

            # after-row separations
            for sep in self.after_row_separations[r]:
                line = ['|']
                for c in range(current_num_col):
                    if cd.col_just[c] != 'a':
                        line.append(sep * desc_column_size[c])
                        if not self._is_last_just_col(current_just, c):
                            sp = self._num_span_col(current_just, c)
                            cchr = self._cis(cd, sp + c)
                            lcchr = '\0'
                            if r > 0:
                                lcchr = self._cis(
                                    self.col_defs[self.row_just[r - 1]],
                                    sp + c)
                            if cchr != ' ':
                                if sep == ' ':
                                    line.append(cchr)
                                elif (self._cis(cd, c) == '|'
                                      and sep == '-' and r > 0
                                      and lcchr == '|'):
                                    line.append('+')
                                else:
                                    line.append(sep)
                            else:
                                line.append(sep)
                line.append('|\n')
                out(pad0 + ''.join(line))

        out(pad0 + '`' + '-' * (self.width - 2) + "'\n")


def _get_next_string_value(s, width):
    """get_next_string_value: returns (chunk, rest)."""
    if s is None:
        return '', ''
    n = len(s)
    p = 0
    x = 0
    while x < width and p < n and s[p:p + 2] != '//':
        p += 1
        x += 1
    if s[p:p + 2] == '//':
        p += 2
    elif p < n:
        # backup to a space
        while p > 0 and s[p] != ' ':
            p -= 1
            x -= 1
        if p < n and s[p] == ' ':
            p += 1
    return s[:x] if x >= 0 else '', s[p:]
