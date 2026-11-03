"""
Memory-light reader/evaluator for the submitted ONNX model (a single TreeEnsembleClassifier).

Why this exists: onnxruntime expands this 47.7 MB random forest to about 415 MB of RAM, which does not
fit next to Gradio and RDKit in Render's 512 MB free tier. This module reads the SAME .onnx file
(its tree arrays are decoded straight from the protobuf bytes) and walks the trees with NumPy,
using about 40 MB. test_app_parity.py checks that it returns the same probabilities as
onnxruntime on the same file (differences ~1e-7, float rounding only).
"""

import struct

import numpy as np


def _varint(buf, pos):
    result = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            return result, pos
        shift += 7


def _fields(buf, start, end):
    """Yield (field_number, wire_type, value) for one protobuf message; value is an int for
    varint/fixed fields and a (start, end) byte range for length-delimited fields."""
    pos = start
    while pos < end:
        key, pos = _varint(buf, pos)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _varint(buf, pos)
        elif wire == 2:
            length, pos = _varint(buf, pos)
            value = (pos, pos + length)
            pos += length
        elif wire == 5:
            value, pos = struct.unpack_from("<I", buf, pos)[0], pos + 4
        elif wire == 1:
            value, pos = struct.unpack_from("<Q", buf, pos)[0], pos + 8
        else:
            raise ValueError("unsupported protobuf wire type %d" % wire)
        yield field, wire, value


def _decode_varints(raw):
    """Decode a run of `0x40 <varint>` elements (a non-packed repeated int64 field) with NumPy.
    Every element holds exactly two bytes below 0x80: the tag byte, then the varint's last byte."""
    low = np.flatnonzero(raw < 0x80)
    tags, ends = low[0::2], low[1::2]
    if len(tags) != len(ends) or not (raw[tags] == 0x40).all() or (ends[:-1] + 1 != tags[1:]).any():
        raise ValueError("unexpected int encoding in the ONNX file")
    starts = tags + 1
    length = ends - starts + 1
    out = np.zeros(len(starts), np.int64)
    for k in range(int(length.max())):
        valid = k < length
        out[valid] |= (raw[starts[valid] + k].astype(np.int64) & 0x7F) << (7 * k)
    return out


def _read_attributes(buf, node_range):
    """name -> NumPy array (or list of bytes / bytes) for every attribute of one ONNX node."""
    attrs = {}
    arr = np.frombuffer(buf, np.uint8)
    for field, _, (start, end) in _fields(buf, *node_range):
        if field != 5:                       # NodeProto.attribute
            continue
        if buf[start] != 0x0A:               # AttributeProto.name comes first
            raise ValueError("unexpected attribute layout in the ONNX file")
        name_len, pos = _varint(buf, start + 1)
        name = bytes(buf[pos:pos + name_len]).decode()
        pos += name_len
        if buf[end - 3:end - 1] == b"\xa0\x01":       # trailing `type` field (20)
            end -= 3
        tag = buf[pos] if pos < end else None
        if tag == 0x3D:                      # repeated float, one tagged 4-byte value each
            block = arr[pos:end].reshape(-1, 5)
            if not (block[:, 0] == 0x3D).all():
                raise ValueError("unexpected float encoding in the ONNX file")
            attrs[name] = np.ascontiguousarray(block[:, 1:]).view("<f4").ravel()
        elif tag == 0x40:                    # repeated int64
            attrs[name] = _decode_varints(arr[pos:end])
        elif tag == 0x4A:                    # repeated string
            items = []
            while pos < end:
                length, p = _varint(buf, pos + 1)
                items.append(bytes(buf[p:p + length]))
                pos = p + length
            attrs[name] = items
        elif tag == 0x22:                    # single string (post_transform)
            length, p = _varint(buf, pos + 1)
            attrs[name + "#s"] = bytes(buf[p:p + length])
    return attrs


class TreeForest:
    """Evaluates a TreeEnsembleClassifier; predict_proba(X) matches onnxruntime's 'probabilities'."""

    def __init__(self, onnx_path):
        buf = open(onnx_path, "rb").read()
        attrs = None
        for field, _, graph in _fields(buf, 0, len(buf)):
            if field != 7:                   # ModelProto.graph
                continue
            for f, _, node in _fields(buf, *graph):
                if f == 1:                   # GraphProto.node
                    op = [bytes(buf[v[0]:v[1]]) for g, w, v in _fields(buf, *node) if g == 4]
                    if op and op[0] == b"TreeEnsembleClassifier":
                        attrs = _read_attributes(buf, node)
        if attrs is None:
            raise ValueError("no TreeEnsembleClassifier found in %s" % onnx_path)
        if attrs.get("post_transform#s", b"NONE") != b"NONE":
            raise ValueError("unsupported post_transform %r" % attrs["post_transform#s"])
        if set(attrs["nodes_modes"]) - {b"LEAF", b"BRANCH_LEQ"}:
            raise ValueError("unsupported node modes in the ONNX file")

        tree_ids = attrs["nodes_treeids"]
        node_ids = attrs["nodes_nodeids"]
        # Global index of each node. Nodes are stored tree by tree with ids 0..n-1 inside a tree.
        starts = np.flatnonzero(np.r_[True, tree_ids[1:] != tree_ids[:-1]])
        self.roots = starts.astype(np.int32)
        offset = np.repeat(starts, np.diff(np.r_[starts, len(tree_ids)]))
        if not np.array_equal(node_ids + offset, np.arange(len(node_ids))):
            raise ValueError("unexpected node ordering in the ONNX file")
        n = len(node_ids)
        self.n_classes = len(attrs["classlabels_int64s"])

        is_leaf = np.fromiter((m == b"LEAF" for m in attrs["nodes_modes"]), bool, n)
        idx = np.arange(n, dtype=np.int32)
        self.feature = np.where(is_leaf, 0, attrs["nodes_featureids"]).astype(np.int32)
        self.threshold = attrs["nodes_values"].astype(np.float32)
        # Leaves point at themselves so the walk can simply run for a fixed number of steps.
        self.go_true = np.where(is_leaf, idx, attrs["nodes_truenodeids"] + offset).astype(np.int32)
        self.go_false = np.where(is_leaf, idx, attrs["nodes_falsenodeids"] + offset).astype(np.int32)

        leaf_rows = (attrs["class_nodeids"] + self._offset_of(attrs["class_treeids"], starts, tree_ids))
        self.leaf_weight = np.zeros((n, self.n_classes), np.float32)
        np.add.at(self.leaf_weight, (leaf_rows, attrs["class_ids"]), attrs["class_weights"])
        self.is_leaf = is_leaf
        self.single_score = self.n_classes == 2 and set(np.unique(attrs["class_ids"]).tolist()) == {0}
        del buf, attrs

    @staticmethod
    def _offset_of(class_treeids, starts, tree_ids):
        first = {int(t): int(s) for t, s in zip(tree_ids[starts], starts)}
        return np.fromiter((first[int(t)] for t in class_treeids), np.int64, len(class_treeids))

    def predict_proba(self, X):
        X = np.asarray(X, np.float32)
        rows = np.arange(len(X))[:, None]
        cur = np.broadcast_to(self.roots, (len(X), len(self.roots))).copy()
        while not self.is_leaf[cur].all():
            go_left = X[rows, self.feature[cur]] <= self.threshold[cur]
            cur = np.where(go_left, self.go_true[cur], self.go_false[cur])
        total = self.leaf_weight[cur].sum(axis=1)
        if self.single_score:                # binary file with one weight per leaf = P(class 1)
            return np.stack([1.0 - total[:, 0], total[:, 0]], axis=1)
        return total
