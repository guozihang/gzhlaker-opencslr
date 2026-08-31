"""Network model + DP alignment: port of sclite net_adt.c / net_dp.c.

Arc lists are linked-list atoms (ARC_LIST_ATOM-style) so that ordering
and mid-iteration mutation match the C code exactly.
"""
import struct

from . import text as _t
from . import word as _w


def f32(x):
    """Round a Python float to C 32-bit float precision."""
    return struct.unpack('f', struct.pack('f', x))[0]


# Network_traverse modes
NT_For = 0x01
NT_Back = 0x02
NT_Breadth = 0x04
NT_Depth = 0x08
NT_Inorder = 0x10
NT_CA_For = 0x20
NT_CA_Back = 0x40


class ArcListAtom:
    __slots__ = ('arc', 'next')

    def __init__(self, arc):
        self.arc = arc
        self.next = None


def add_to_arc_list(head, arc):
    """add_to_arc_list: append atom to END of list; returns head."""
    atom = ArcListAtom(arc)
    if head is None:
        return atom
    p = head
    while p.next is not None:
        p = p.next
    p.next = atom
    return head


def del_from_arc_list(head, arc):
    """Remove the atom holding arc; returns new head."""
    prev = None
    p = head
    while p is not None:
        if p.arc is arc:
            if prev is None:
                return p.next
            prev.next = p.next
            return head
        prev = p
        p = p.next
    return head


def arc_list_to_pylist(head):
    out = []
    p = head
    while p is not None:
        out.append(p.arc)
        p = p.next
    return out


class NODE:
    __slots__ = ('name', 'in_arcs', 'out_arcs', 'net', 'flag1')

    def __init__(self, name, net):
        self.name = name
        self.in_arcs = None
        self.out_arcs = None
        self.net = net
        self.flag1 = 0


class ARC:
    __slots__ = ('data', 'from_node', 'to_node', 'net')

    def __init__(self, data, from_node, to_node, net):
        self.data = data
        self.from_node = from_node
        self.to_node = to_node
        self.net = net


class NETWORK:
    def __init__(self, name=''):
        self.name = name
        self.start_node = NODE('START', self)
        self.stop_node = None
        self.node_count = 1
        self.arc_count = 0


def insert_arc_after_node(net, last_node, data):
    """NETWORK_insert_arc_after_node."""
    arcx = ARC(data, last_node, None, net)
    new_node = NODE('STOP' if net.stop_node is None else None, net)
    new_node.in_arcs = add_to_arc_list(new_node.in_arcs, arcx)
    if net.stop_node is None:
        net.stop_node = new_node
    net.node_count += 1
    net.arc_count += 1
    arcx.to_node = new_node
    # move last_node's out arcs to new_node
    oa = last_node.out_arcs
    while oa is not None:
        new_node.out_arcs = add_to_arc_list(new_node.out_arcs, oa.arc)
        oa.arc.from_node = new_node
        oa = oa.next
    last_node.out_arcs = None
    last_node.out_arcs = add_to_arc_list(last_node.out_arcs, arcx)
    return 0


def insert_arc_before_node(net, this_node, data):
    """NETWORK_insert_arc_before_node."""
    arcx = ARC(data, None, this_node, net)
    new_node = NODE(None, net)
    new_node.out_arcs = add_to_arc_list(new_node.out_arcs, arcx)
    net.node_count += 1
    net.arc_count += 1
    arcx.from_node = new_node
    oa = this_node.in_arcs
    while oa is not None:
        new_node.in_arcs = add_to_arc_list(new_node.in_arcs, oa.arc)
        oa.arc.to_node = new_node
        oa = oa.next
    this_node.in_arcs = None
    this_node.in_arcs = add_to_arc_list(this_node.in_arcs, arcx)
    return 0


def add_arc_to_head(net, data):
    return insert_arc_after_node(net, net.start_node, data)


def add_arc_to_tail(net, data):
    if net.stop_node is None:
        return insert_arc_after_node(net, net.start_node, data)
    return insert_arc_before_node(net, net.stop_node, data)


def merge_network(from_node, to_node, from_net):
    """Network_merge_network: splice from_net between the two nodes."""
    net = from_node.net
    net.node_count += max(from_net.node_count - 2, 0)
    net.arc_count += from_net.arc_count
    # link in the start
    oa = from_net.start_node.out_arcs
    while oa is not None:
        from_node.out_arcs = add_to_arc_list(from_node.out_arcs, oa.arc)
        oa.arc.from_node = from_node
        oa.arc.net = net
        oa = oa.next
    # link in the stop
    oa = from_net.stop_node.in_arcs
    while oa is not None:
        to_node.in_arcs = add_to_arc_list(to_node.in_arcs, oa.arc)
        oa.arc.to_node = to_node
        oa = oa.next
    return 0


def delete_arc(arc):
    """Network_delete_arc (node cleanup replicated)."""
    if arc is None:
        return 1
    net = arc.net
    from_node = arc.from_node
    to_node = arc.to_node
    from_node.out_arcs = del_from_arc_list(from_node.out_arcs, arc)
    to_node.in_arcs = del_from_arc_list(to_node.in_arcs, arc)
    net.arc_count -= 1

    if to_node.in_arcs is None:
        if to_node.name == 'STOP':
            if from_node.name != 'START':
                # move from_node's in_arcs to to_node, kill from_node
                p = from_node.in_arcs
                while p is not None:
                    to_node.in_arcs = add_to_arc_list(to_node.in_arcs, p.arc)
                    p.arc.to_node = to_node
                    p = p.next
                from_node = None
            else:
                net.stop_node = None
                to_node = None
        else:
            # move to_node's out_arcs onto from_node, kill to_node
            p = to_node.out_arcs
            while p is not None:
                from_node.out_arcs = add_to_arc_list(from_node.out_arcs,
                                                     p.arc)
                p.arc.from_node = from_node
                p = p.next
            to_node = None
    if from_node is not None and from_node.out_arcs is None:
        if from_node.name == 'START':
            if to_node is not None:
                net.stop_node = None
                to_node = None
        else:
            p = from_node.in_arcs
            while p is not None:
                to_node.in_arcs = add_to_arc_list(to_node.in_arcs, p.arc)
                p.arc.to_node = to_node
                p = p.next
            from_node = None
    return 0


def add_net_to_tail(net, mnet):
    """Network_add_net_to_tail."""
    add_arc_to_tail(net, None)
    nullarc = net.stop_node.in_arcs.arc
    merge_network(nullarc.from_node, net.stop_node, mnet)
    delete_arc(nullarc)
    return 0


def add_net_to_head(net, mnet):
    """Network_add_net_to_head."""
    add_arc_to_head(net, None)
    nullarc = net.start_node.out_arcs.arc
    merge_network(net.start_node, nullarc.to_node, mnet)
    delete_arc(nullarc)
    return 0


def traverse(net, node_op=None, node_data=None, arc_op=None, arc_data=None,
             mode=0):
    """Network_traverse (NT_For/NT_Back, NT_Depth/NT_Breadth/NT_Inorder,
    NT_CA_For/NT_CA_Back) with C-identical ordering."""
    if net is None:
        return
    if (mode & NT_CA_For) == 0 and (mode & NT_CA_Back) == 0:
        mode |= NT_CA_For
    if (mode & NT_For) == 0 and (mode & NT_Back) == 0:
        mode |= NT_For
    if (mode & NT_Breadth) == 0 and (mode & NT_Depth) == 0 \
            and (mode & NT_Inorder) == 0:
        mode |= NT_Depth

    counts = None
    if (mode & NT_Inorder) != 0:
        # set flag1 to in/out arc counts via a plain traversal
        counts = {}
        if (mode & NT_For) != 0:
            def cnt_in(node, _d):
                n = 0
                p = node.in_arcs
                while p is not None:
                    n += 1
                    p = p.next
                counts[node] = n
            traverse(net, cnt_in, None, None, None, 0)
        else:
            def cnt_out(node, _d):
                n = 0
                p = node.out_arcs
                while p is not None:
                    n += 1
                    p = p.next
                counts[node] = n
            traverse(net, cnt_out, None, None, None, NT_Back)

    fringe = []
    visited = set()
    if (mode & NT_For) != 0:
        fringe.append(net.start_node)
        visited.add(net.start_node)
    else:
        fringe.append(net.stop_node)
        visited.add(net.stop_node)

    while fringe:
        if (mode & NT_Breadth) != 0 or (mode & NT_Inorder) != 0:
            node = fringe.pop(0)
        else:
            node = fringe.pop()
        if node_op is not None:
            node_op(node, node_data)
        # push pointed-to nodes
        p = node.out_arcs if (mode & NT_For) != 0 else node.in_arcs
        while p is not None:
            pushnode = p.arc.to_node if (mode & NT_For) != 0 \
                else p.arc.from_node
            if (mode & NT_Inorder) != 0:
                counts[pushnode] -= 1
                if counts[pushnode] > 0:
                    p = p.next
                    continue
            if pushnode in visited:
                p = p.next
                continue
            visited.add(pushnode)
            if (mode & NT_Breadth) != 0 or (mode & NT_Inorder) != 0:
                fringe.append(pushnode)
            else:
                fringe.append(pushnode)  # LL_put_front on a stack == append
            p = p.next
        # travel the arcs
        if (mode & NT_CA_For) != 0 and arc_op is not None:
            p = node.out_arcs
            while p is not None:
                pn = p.next
                arc_op(p.arc, arc_data)
                p = pn
        if (mode & NT_CA_Back) != 0 and arc_op is not None:
            p = node.in_arcs
            while p is not None:
                pn = p.next
                arc_op(p.arc, arc_data)
                p = pn


def _expand_alternates(arc, _ptr):
    tw = arc.data
    if tw.value is None:
        return
    if tw.value.find(b'{') < 0:
        return
    # copy text removing beginning '{' and ending '}'
    p = tw.value[1:]
    r = p.rfind(b'}')
    if r >= 0:
        p = p[:r]
    subnet2 = None
    i = 0
    n = len(p)
    while True:
        # skip nothing here; find_next_alternation handles end check
        if i >= n:
            break
        token, i = _t.find_next_alternation(p, i)
        if token is None:
            break
        subnet = create_from_TEXT(token, 'alt')
        if subnet2 is None:
            subnet2 = subnet
        else:
            merge_network(subnet2.start_node, subnet2.stop_node, subnet)
    merge_network(arc.from_node, arc.to_node, subnet2)
    delete_arc(arc)


def create_from_TEXT(text, name=''):
    """Network_create_from_TEXT."""
    net = NETWORK(name)
    i = 0
    n = len(text)
    while i < n:
        token, i = _t.find_next_token(text, i)
        if token is None:
            break
        tword = _w.new_WORD_parseText(token, -1, 0.0, 0.0, 0.0, False,
                                      False, -1.0)
        add_arc_to_tail(net, tword)
    traverse(net, None, None, _expand_alternates, None, 0)
    return net


def create_from_WTOKE(wt, start, end, name='', left_to_right=True):
    """Network_create_from_WTOKE.  wt: WTOKE object with word[] (1-based),
    n, has_conf."""
    net = NETWORK(name)
    i = start
    while i <= wt.n and i <= end:
        w = wt.word[i]
        if (not w.ignore) and w.alternate:
            alt_net_out = None
            # skip the ALT_BEGIN
            while i <= wt.n and i <= end and \
                    _t.strcasecmp(b'<ALT_BEGIN>', wt.word[i].sp) == 0:
                i += 1
            while True:
                alt_net = NETWORK(name)
                acnt = 0
                while i <= wt.n and i <= end and \
                        _t.strcmp_master(b'<ALT', wt.word[i].sp, 4, True) != 0:
                    ww = wt.word[i]
                    tword = _w.new_WORD_parseText(
                        ww.sp, -1, ww.t1, ww.t1 + ww.dur,
                        ww.confidence if wt.has_conf else -1.0,
                        False, False, -1.0)
                    if left_to_right:
                        add_arc_to_tail(alt_net, tword)
                    else:
                        add_arc_to_head(alt_net, tword)
                    i += 1
                    acnt += 1
                if acnt == 0:
                    conf = wt.word[i].confidence if wt.has_conf else -1.0
                    tword = _w.new_WORD(b'@', -1, 0.0, 0.0, conf,
                                        None, None, False, False, -1.0)
                    add_arc_to_tail(alt_net, tword)
                    i += 1
                if alt_net_out is None:
                    alt_net_out = alt_net
                else:
                    merge_network(alt_net_out.start_node,
                                  alt_net_out.stop_node, alt_net)
                # advance i past the separator
                wsp_i = wt.word[i].sp if i <= wt.n else b''
                wsp_i1 = wt.word[i + 1].sp if i + 1 <= wt.n else b''
                if _t.strcasecmp(b'<ALT>', wsp_i) == 0 and \
                        _t.strcasecmp(b'<ALT_END>', wsp_i1) == 0:
                    pass
                elif _t.strcasecmp(b'<ALT_END>', wsp_i) != 0:
                    i += 1
                if not (i <= wt.n and i <= end and
                        _t.strcasecmp(b'<ALT_END>', wt.word[i].sp) != 0):
                    break
            if alt_net_out is None:
                raise RuntimeError('failed to produce alternation network')
            if left_to_right:
                add_net_to_tail(net, alt_net_out)
            else:
                add_net_to_head(net, alt_net_out)
            i += 1
        elif not w.ignore:
            tword = _w.new_WORD_parseText(
                w.sp, -1, w.t1, w.t1 + w.dur,
                w.confidence if wt.has_conf else -1.0, False, False, -1.0)
            if left_to_right:
                add_arc_to_tail(net, tword)
            else:
                add_arc_to_head(net, tword)
            i += 1
        else:
            i += 1
    return net


# ------------------------- DP alignment (net_dp.c) ---------------------

class _Cell:
    __slots__ = ('min_d', 'back_a', 'back_b')

    def __init__(self):
        self.min_d = f32(0.0)
        self.back_a = 0
        self.back_b = 0


class NetAlign:
    pass


def _find_arcset_id(arcset, arc, from_idx):
    """find_arcset_id: search from index `from_idx` DOWN to 0."""
    if from_idx == len(arcset):
        from_idx -= 1
    a = from_idx
    while a >= 0:
        if arcset[a] is arc:
            return a
        a -= 1
    return -1


def _add_null_network_heads(net):
    add_arc_to_head(net, None)


def _traverse_arcs_inorder(net):
    """Arcset construction: Network_traverse(net, add_to_arcset,
    NT_CA_For + NT_Inorder)."""
    out = []

    def arc_op(arc, _d):
        out.append(arc)

    traverse(net, None, None, arc_op, None, NT_CA_For + NT_Inorder)
    return out


def _calc_minimum_delete(na, a, b):
    """Returns (min_arc_d, min_ind_a) as float32."""
    min_arc_d = f32(999999999.9)
    min_ind_a = -1
    arc_a = na.arcset_a[a]
    p = arc_a.from_node.in_arcs
    while p is not None:
        arc_a_id = _find_arcset_id(na.arcset_a, p.arc, a)
        if arc_a_id < 0:
            raise RuntimeError('find_arcset_id failed (delete)')
        if na.cell[arc_a_id][b].min_d < min_arc_d:
            min_arc_d = na.cell[arc_a_id][b].min_d
            min_ind_a = arc_a_id
        p = p.next
    return min_arc_d, min_ind_a


def _calc_minimum_insert(na, a, b):
    min_arc_d = f32(999999999.9)
    min_ind_b = -1
    arc_b = na.arcset_b[b]
    p = arc_b.from_node.in_arcs
    while p is not None:
        arc_b_id = _find_arcset_id(na.arcset_b, p.arc, b)
        if arc_b_id < 0:
            raise RuntimeError('find_arcset_id failed (insert)')
        if na.cell[a][arc_b_id].min_d < min_arc_d:
            min_arc_d = na.cell[a][arc_b_id].min_d
            min_ind_b = arc_b_id
        p = p.next
    return min_arc_d, min_ind_b


def _calc_minimum_substi(na, a, b):
    min_arc_d = f32(999999999.9)
    min_ind_a = -1
    min_ind_b = -1
    arc_a = na.arcset_a[a]
    arc_b = na.arcset_b[b]
    pa = arc_a.from_node.in_arcs
    while pa is not None:
        arc_a_id = _find_arcset_id(na.arcset_a, pa.arc, a)
        if arc_a_id < 0:
            raise RuntimeError('find_arcset_id failed (sub a)')
        pb = arc_b.from_node.in_arcs
        while pb is not None:
            arc_b_id = _find_arcset_id(na.arcset_b, pb.arc, b)
            if arc_b_id < 0:
                raise RuntimeError('find_arcset_id failed (sub b)')
            if na.cell[arc_a_id][arc_b_id].min_d < min_arc_d:
                min_arc_d = na.cell[arc_a_id][arc_b_id].min_d
                min_ind_b = arc_b_id
                min_ind_a = arc_a_id
            pb = pb.next
        pa = pa.next
    return min_arc_d, min_ind_a, min_ind_b


def dpalign(net_a, net_b, wwd, include_nulls=False):
    """Network_dpalign.  Returns PATH (list of (eval, a_word, b_word)).

    eval codes: P_CORR/P_SUB/P_INS/P_DEL from path module constants.
    """
    from . import path as _p

    _add_null_network_heads(net_a)
    _add_null_network_heads(net_b)

    na = NetAlign()
    na.arcset_a = _traverse_arcs_inorder(net_a)
    na.arcset_b = _traverse_arcs_inorder(net_b)
    na_a = len(na.arcset_a)
    na_b = len(na.arcset_b)
    na.cell = [[_Cell() for _ in range(na_b)] for _ in range(na_a)]
    na.cell[0][0].back_a = -1
    na.cell[0][0].back_b = -1
    na.cell[0][0].min_d = f32(0.0)

    # margin cells
    for b in range(1, na_b):
        min_d, min_ind_b = _calc_minimum_insert(na, 0, b)
        na.cell[0][b].min_d = f32(min_d + wwd(
            None, na.arcset_b[b].data, _w.equal_WORD2))
        na.cell[0][b].back_a = 0
        na.cell[0][b].back_b = min_ind_b
    for a in range(1, na_a):
        min_d, min_ind_a = _calc_minimum_delete(na, a, 0)
        na.cell[a][0].min_d = f32(min_d + wwd(
            na.arcset_a[a].data, None, _w.equal_WORD2))
        na.cell[a][0].back_a = min_ind_a
        na.cell[a][0].back_b = 0

    # internal cells
    for a in range(1, na_a):
        for b in range(1, na_b):
            ins_min_d, ins_min_ind_b = _calc_minimum_insert(na, a, b)
            del_min_d, del_min_ind_a = _calc_minimum_delete(na, a, b)
            sub_min_d, sub_min_ind_a, sub_min_ind_b = \
                _calc_minimum_substi(na, a, b)
            sub_min_d = f32(sub_min_d + wwd(na.arcset_a[a].data,
                                            na.arcset_b[b].data,
                                            _w.equal_WORD2))
            del_min_d = f32(del_min_d + wwd(na.arcset_a[a].data, None,
                                            _w.equal_WORD2))
            ins_min_d = f32(ins_min_d + wwd(None, na.arcset_b[b].data,
                                            _w.equal_WORD2))
            if sub_min_d <= del_min_d and sub_min_d <= ins_min_d:
                na.cell[a][b].min_d = sub_min_d
                na.cell[a][b].back_a = sub_min_ind_a
                na.cell[a][b].back_b = sub_min_ind_b
            elif del_min_d < ins_min_d:
                na.cell[a][b].min_d = del_min_d
                na.cell[a][b].back_a = del_min_ind_a
                na.cell[a][b].back_b = b
            else:
                na.cell[a][b].min_d = ins_min_d
                na.cell[a][b].back_a = a
                na.cell[a][b].back_b = ins_min_ind_b

    # extract: find minimal termination cell
    min_arc_d = 999999.9
    a = b = -1
    pa = net_a.stop_node.in_arcs
    while pa is not None:
        arc_a_id = _find_arcset_id(na.arcset_a, pa.arc, na_a)
        pb = net_b.stop_node.in_arcs
        while pb is not None:
            arc_b_id = _find_arcset_id(na.arcset_b, pb.arc, na_b)
            if min_arc_d > na.cell[arc_a_id][arc_b_id].min_d:
                min_arc_d = na.cell[arc_a_id][arc_b_id].min_d
                a = arc_a_id
                b = arc_b_id
            pb = pb.next
        pa = pa.next

    # backtrace collecting kept items (forward order)
    items = []
    while a > 0 or b > 0:
        ca = na.cell[a][b]
        if a == ca.back_a:
            back = 'INS'
        elif b == ca.back_b:
            back = 'DEL'
        else:
            back = 'SUB'
        la, lb = a, b
        if back == 'DEL':
            aw = na.arcset_a[la].data
            if include_nulls or not _w.null_alt_WORD(aw):
                if not _w.opt_del_WORD(aw):
                    items.append((_p.P_DEL, aw, None))
                else:
                    items.append((_p.P_CORR, aw, _w.make_empty_WORD()))
        elif back == 'SUB':
            aw = na.arcset_a[la].data
            bw = na.arcset_b[lb].data
            if include_nulls or not (_w.null_alt_WORD(aw)
                                     and _w.null_alt_WORD(bw)):
                if _w.equal_WORD2(aw, bw) == 0:
                    items.append((_p.P_CORR, aw, bw))
                else:
                    items.append((_p.P_SUB, aw, bw))
        else:  # INS
            bw = na.arcset_b[lb].data
            if include_nulls or not _w.null_alt_WORD(bw):
                if not _w.opt_del_WORD(bw):
                    items.append((_p.P_INS, None, bw))
                else:
                    items.append((_p.P_CORR, _w.make_empty_WORD(), bw))
        a = ca.back_a
        b = ca.back_b

    items.reverse()
    path = _p.PATH()
    for ev, aw, bw in items:
        path.pset.append(_p.PathSet(ev, aw, bw))
    path.num = len(path.pset)
    return path
