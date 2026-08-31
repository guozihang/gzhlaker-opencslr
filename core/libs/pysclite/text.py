"""TEXT layer: byte-string text handling replicating SCTK sclite text.c.

All text is represented as Python ``bytes`` (C's ``unsigned char *``).
Encoding and language-profile state mirror sclite's static globals.
"""
import json
import os

# Encodings (enum TEXT_ENCODINGS)
ASCII = 0
EXTASCII = 1
GB = 2
UTF8 = 3

# Language profiles (enum TEXT_LANG_PROFILE)
LPROF_GENERIC = 0
LPROF_BABEL_TURKISH = 1
LPROF_BABEL_VIETNAMESE = 2
LPROF_BABEL_LITHUANIAN = 3
LPROF_BABEL_KURMANJI = 4
LPROF_BABEL_CEBUANO = 5
LPROF_BABEL_KAZAKH = 6
LPROF_BABEL_MONGOLIAN = 7
LPROF_BABEL_GUARANI = 8
LPROF_UKRAINIAN = 9

# char-alignment flags (from sctk.h)
CALI_ON = 1
CALI_NOASCII = 2
CALI_DELHYPHEN = 4

STATIC_ENCODING = ASCII
STATIC_LPROF = LPROF_GENERIC

_ALT_BEGIN = ord('{')
_ALT_END = ord('}')

_KCC_TABLES = None


def _load_kcc():
    global _KCC_TABLES
    if _KCC_TABLES is None:
        path = os.path.join(os.path.dirname(__file__), 'kcc_tables.json')
        with open(path, 'r') as f:
            _KCC_TABLES = json.load(f)
    return _KCC_TABLES


def set_encoding(encoding):
    """TEXT_set_encoding: returns True on success."""
    global STATIC_ENCODING
    e = encoding.upper() if isinstance(encoding, str) else encoding.decode().upper()
    if e in ('EUC', 'GB'):
        STATIC_ENCODING = GB
    elif e == 'EXT_ASCII':
        STATIC_ENCODING = EXTASCII
    elif e == 'ASCII':
        STATIC_ENCODING = ASCII
    elif e == 'UTF-8':
        STATIC_ENCODING = UTF8
    else:
        return False
    return True


def set_lang_prof(lprof):
    """TEXT_set_lang_prof: returns True on success."""
    global STATIC_LPROF
    lp = lprof.lower() if isinstance(lprof, str) else lprof.decode().lower()
    m = {
        'generic': LPROF_GENERIC,
        'babel_turkish': LPROF_BABEL_TURKISH,
        'babel_vietnamese': LPROF_BABEL_VIETNAMESE,
        'babel_lithuanian': LPROF_BABEL_LITHUANIAN,
        'babel_kurmanji': LPROF_BABEL_KURMANJI,
        'babel_cebuano': LPROF_BABEL_CEBUANO,
        'babel_kazakh': LPROF_BABEL_KAZAKH,
        'babel_mongolian': LPROF_BABEL_MONGOLIAN,
        'babel_guarani': LPROF_BABEL_GUARANI,
        'ukrainian': LPROF_UKRAINIAN,
    }
    if lp not in m:
        return False
    STATIC_LPROF = m[lp]
    return True


def _tolower(b):
    """C tolower for a single byte (ASCII rules)."""
    if 0x41 <= b <= 0x5A:
        return b + 32
    return b


def _toupper(b):
    if 0x61 <= b <= 0x7A:
        return b - 32
    return b


def _isspace(b):
    """C isspace in the C locale."""
    return b in (0x20, 0x09, 0x0A, 0x0B, 0x0C, 0x0D)


def nbytes_of_char(buf, i=0):
    """TEXT_nbytes_of_char: bytes consumed by char at buf[i]."""
    p = buf[i]
    if STATIC_ENCODING in (ASCII, EXTASCII):
        return 1
    if (p & 0x80) == 0:
        return 1
    if STATIC_ENCODING == GB:
        return 2
    # UTF8
    if (p & 0xE0) == 0xC0:
        return 2
    if (p & 0xF0) == 0xE0:
        return 3
    if (p & 0xF8) == 0xF0:
        return 4
    if (p & 0xFC) == 0xF8:
        return 5
    if (p & 0xFE) == 0xFC:
        return 6
    raise ValueError("UTF-8 parsing of character size failed: %x" % p)


def _is_2byte(buf, i):
    return nbytes_of_char(buf, i) == 2


def _is_extasc(b):
    return (b & 0x80) != 0


def VTisspace(buf, i):
    """VTisspace macro on buf[i]."""
    if _is_2byte(buf, i):
        return False
    if _is_extasc(buf[i]):
        return False
    return _isspace(buf[i])


def skip_wspace(buf, i=0):
    while i < len(buf) and VTisspace(buf, i):
        i += 1
    return i


def chrlen(buf):
    """TEXT_chrlen: number of characters (not bytes)."""
    i = 0
    n = 0
    while i < len(buf):
        i += nbytes_of_char(buf, i)
        n += 1
    return n


def get_utf_codepoint(buf, i=0):
    """TEXT_getUTFCodePoint."""
    n = nbytes_of_char(buf, i)
    if n == 1:
        return buf[i]
    if n == 2:
        return ((buf[i] & 0x1F) << 6) + (buf[i + 1] & 0x3F)
    if n == 3:
        return (((buf[i] & 0x0F) << 12) + ((buf[i + 1] & 0x3F) << 6)
                + (buf[i + 2] & 0x3F))
    if n == 4:
        return (((buf[i] & 0x07) << 18) + ((buf[i + 1] & 0x3F) << 12)
                + ((buf[i + 2] & 0x3F) << 6) + (buf[i + 3] & 0x3F))
    if n == 5:
        return (((buf[i] & 0x03) << 24) + ((buf[i + 1] & 0x3F) << 18)
                + ((buf[i + 2] & 0x3F) << 12) + ((buf[i + 3] & 0x3F) << 6)
                + (buf[i + 4] & 0x3F))
    raise ValueError("6-byte UTF8 codepoint is not supported")


def utf_codepoint_to_bytes(c):
    """TEXT_UTFCodePointToTEXT."""
    if c < 0x80:
        return bytes([c])
    if c < 0x800:
        return bytes([192 + c // 64, 128 + c % 64])
    if c < 0x10000:
        return bytes([224 + c // 4096, 128 + (c // 64) % 64, 128 + c % 64])
    if c < 0x200000:
        return bytes([240 + c // 262144, 128 + (c // 4096) % 64,
                      128 + (c // 64) % 64, 128 + c % 64])
    raise ValueError("codepoint too large")


def _get_known_utf_case_cp(in_cp, to_low):
    """getKnownUFTCaseCP: returns mapped codepoint or -1."""
    tables = _load_kcc()
    key = None
    if STATIC_LPROF == LPROF_BABEL_TURKISH:
        key = 'babel_turkish'
    elif STATIC_LPROF == LPROF_BABEL_VIETNAMESE:
        key = 'babel_vietnamese'
    elif STATIC_LPROF == LPROF_BABEL_LITHUANIAN:
        key = 'babel_lithuanian'
    elif STATIC_LPROF == LPROF_BABEL_KURMANJI:
        key = 'babel_kurmanji'
    elif STATIC_LPROF == LPROF_BABEL_CEBUANO:
        key = 'babel_cebuano'
    elif STATIC_LPROF == LPROF_BABEL_KAZAKH:
        key = 'babel_kazakh'
    elif STATIC_LPROF == LPROF_BABEL_MONGOLIAN:
        key = 'babel_mongolian'
    elif STATIC_LPROF == LPROF_BABEL_GUARANI:
        key = 'babel_guarani'
    elif STATIC_LPROF == LPROF_UKRAINIAN:
        key = 'ukrainian'
    if key is None:
        return -1
    # table entries: [upper, lower]; toLow searches column 0 -> column 1
    src_col = 0 if to_low else 1
    dst_col = 1 if to_low else 0
    for pair in tables[key]:
        if pair[src_col] == in_cp:
            return pair[dst_col]
    return -1


def str_to_master(buf, to_low=True):
    """TEXT_str_to_master: case conversion per encoding/profile."""
    if buf is None:
        return None
    out = bytearray()
    i = 0
    n = len(buf)
    while i < n:
        c1 = nbytes_of_char(buf, i)
        ch = buf[i:i + c1]
        if STATIC_ENCODING == GB:
            if c1 == 1:
                ch = bytes([_tolower(ch[0]) if to_low else _toupper(ch[0])])
        elif STATIC_ENCODING == UTF8:
            in_cp = get_utf_codepoint(buf, i)
            out_cp = _get_known_utf_case_cp(in_cp, to_low)
            if out_cp > 0:
                ch = utf_codepoint_to_bytes(out_cp)
            else:
                if c1 == 1:
                    ch = bytes([_tolower(ch[0]) if to_low else _toupper(ch[0])])
        elif STATIC_ENCODING == ASCII:
            ch = bytes([_tolower(ch[0]) if to_low else _toupper(ch[0])])
        elif STATIC_ENCODING == EXTASCII:
            b = ch[0]
            if _is_extasc(b):
                if to_low:
                    if 192 <= b <= 223:
                        b += 32
                else:
                    if b >= 224:
                        b -= 32
            else:
                b = _tolower(b) if to_low else _toupper(b)
            ch = bytes([b])
        out += ch
        i += c1
    return bytes(out)


def separate_chars(from_b, flag):
    """TEXT_separate_chars: split into space-separated characters.

    flag: CALI_NOASCII | CALI_DELHYPHEN.  Returns bytes.
    """
    not_ascii = (flag & CALI_NOASCII) != 0
    del_hyphen = (flag & CALI_DELHYPHEN) != 0
    to = bytearray()
    if len(from_b) == 0:
        return bytes(to)
    last_is_ascii = False
    last_is_space = True
    i = 0
    n = len(from_b)
    while i < n:
        cs = nbytes_of_char(from_b, i)
        # 2-character G~ in Guarani: 'G'/'g' + 0xCC 0x83
        is_guarani_g = False
        if STATIC_LPROF == LPROF_BABEL_GUARANI:
            if (from_b[i] in (ord('G'), ord('g')) and i + 2 < n
                    and from_b[i + 1] == 0xCC and from_b[i + 2] == 0x83):
                is_guarani_g = True
        is_ascii = (STATIC_ENCODING in (ASCII, EXTASCII)
                    or (from_b[i] & 0x80) == 0)
        is_hyphen = (cs == 1 and from_b[i] == ord('-'))
        is_space = (cs == 1 and from_b[i] == 0x20)
        if is_guarani_g:
            if not last_is_space:
                to += b' '
            to += from_b[i:i + 3]
            last_is_ascii = False
            last_is_space = False
            cs = 3
        elif len(to) == 0:
            if del_hyphen and is_hyphen:
                pass
            else:
                to += from_b[i:i + cs]
                last_is_ascii = is_ascii
                last_is_space = is_space
        else:
            if del_hyphen and is_hyphen:
                pass
            else:
                if last_is_space:
                    pass
                else:
                    if is_space:
                        pass
                    elif not_ascii and is_ascii and last_is_ascii:
                        pass
                    else:
                        to += b' '
                to += from_b[i:i + cs]
                last_is_ascii = is_ascii
                last_is_space = is_space
        i += cs
    return bytes(to)


def find_next_token(buf, i):
    """find_next_TEXT_token.  Returns (token_bytes_or_None, new_i)."""
    n = len(buf)
    # skip leading white space
    while i < n and VTisspace(buf, i):
        i += nbytes_of_char(buf, i)
    if i >= n:
        return None, i
    token = bytearray()
    if buf[i] == _ALT_BEGIN:
        alt_cnt = 0
        while True:
            nchar = nbytes_of_char(buf, i)
            if buf[i] == _ALT_BEGIN:
                alt_cnt += 1
            if buf[i] == _ALT_END:
                alt_cnt -= 1
            token += buf[i:i + nchar]
            i += nchar
            if i >= n or alt_cnt <= 0:
                break
        if alt_cnt > 0:
            return None, i
    else:
        while i < n and not VTisspace(buf, i):
            nchar = nbytes_of_char(buf, i)
            token += buf[i:i + nchar]
            i += nchar
    return bytes(token), i


def find_next_alternation(buf, i):
    """find_next_TEXT_alternation.  Returns (token_bytes_or_None, new_i)."""
    n = len(buf)
    alt_cnt = 0
    # skip leading white space and '/'
    while i < n and (VTisspace(buf, i) or buf[i] == ord('/')):
        i += nbytes_of_char(buf, i)
    if i >= n:
        return None, i
    token = bytearray()
    while True:
        nchar = nbytes_of_char(buf, i)
        if nchar > 1:
            token += buf[i:i + nchar]
            i += nchar
        else:
            if buf[i] == _ALT_BEGIN:
                alt_cnt += 1
            if buf[i] == _ALT_END:
                alt_cnt -= 1
            token += buf[i:i + 1]
            i += 1
        if i >= n:
            break
        c = buf[i]
        if (c == ord('/') and alt_cnt <= 0) or (c == _ALT_END and alt_cnt <= 0):
            break
    return bytes(token), i


class StrtokState:
    """Stateful TEXT_strtok equivalent."""

    def __init__(self):
        self.base = None   # bytes buffer being tokenized (mutable copy)
        self.pos = 0
        self.last_start = None  # buffer offset of the last returned token

    def strtok(self, p, t):
        """p: bytes or None (continue); t: separator bytes.

        Returns token bytes or None.  Mirrors TEXT_strtok semantics:
        separators matched only on 1-byte chars.
        """
        if p is None:
            if self.base is None:
                return None
        else:
            self.base = p
            self.pos = 0
        base = self.base
        n = len(base)
        if self.pos >= n:
            self.last_start = None
            return None
        ext = self.pos
        # skip separators (leading)
        while ext < n and base[ext] in t:
            sz = nbytes_of_char(base, ext)
            ext += sz
            self.pos += sz
        if ext >= n:
            self.pos = ext
            self.last_start = None
            return None
        start = ext
        self.last_start = start
        # skip the token
        while ext < n:
            sz = nbytes_of_char(base, ext)
            if sz == 1:
                if base[ext] in t:
                    self.pos = ext + 1
                    return base[start:ext]
                ext += 1
            else:
                ext += sz
        self.pos = ext
        return base[start:ext]


def strcmp_master(p1, p2, n=-1, do_case=False):
    """TEXT_strcmp_master: returns -1/0/1 like C."""
    if p1 is None and p2 is None:
        return 0
    if p2 is None:
        return 1
    if p1 is None:
        return -1
    if do_case:
        p1 = str_to_master(p1, True)
        p2 = str_to_master(p2, True)
    i1 = i2 = 0
    l1, l2 = len(p1), len(p2)
    iteration = 0
    while True:
        e1 = (i1 >= l1)
        e2 = (i2 >= l2)
        if e1 and e2:
            return 0
        if e2:
            return 1
        if e1:
            return -1
        c1 = nbytes_of_char(p1, i1)
        c2 = nbytes_of_char(p2, i2)
        if c1 != c2:
            return -1 if c1 < c2 else 1
        seg1 = p1[i1:i1 + c1]
        seg2 = p2[i2:i2 + c2]
        if seg1 != seg2:
            return -1 if seg1 < seg2 else 1
        i1 += c1
        i2 += c2
        iteration += 1
        if n != -1 and iteration >= n:
            return 0


def strcmp(p1, p2):
    return strcmp_master(p1, p2, -1, False)


def strcasecmp(p1, p2):
    return strcmp_master(p1, p2, -1, True)
