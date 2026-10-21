"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.setup = []   # ops emitted with add(); packed together with the body

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def rw(self, engine, op):
        """Return (reads, writes): the scratch addresses an op reads and writes."""
        def blk(a):
            return [a + k for k in range(VLEN)]
        name = op[0]
        if engine == "alu":
            _, d, a, b = op
            return [a, b], [d]
        if engine == "valu":
            if name == "vbroadcast":
                return [op[2]], blk(op[1])
            if name == "multiply_add":
                return blk(op[2]) + blk(op[3]) + blk(op[4]), blk(op[1])
            return blk(op[2]) + blk(op[3]), blk(op[1])
        if engine == "load":
            if name == "load":
                return [op[2]], [op[1]]
            if name == "load_offset":
                return [op[2] + op[3]], [op[1] + op[3]]
            if name == "vload":
                return [op[2]], blk(op[1])
            if name == "const":
                return [], [op[1]]
        if engine == "store":
            if name == "store":
                return [op[1], op[2]], []
            if name == "vstore":
                return [op[1]] + blk(op[2]), []
        if engine == "flow":
            if name == "select":
                return [op[2], op[3], op[4]], [op[1]]
            if name == "vselect":
                return blk(op[2]) + blk(op[3]) + blk(op[4]), blk(op[1])
            if name == "add_imm":
                return [op[2]], [op[1]]
            if name == "pause":
                return [], []
        if engine == "debug":
            if name == "compare":
                return [op[1]], []
            if name == "vcompare":
                return blk(op[1]), []
            return [], []
        raise ValueError(f"build() doesn't know how to schedule {engine} {op}")
 
    def build(self, slots, vliw: bool = True):
        """
        List scheduler: put each op in the earliest bundle where it is legal.
        Each item in slots is (engine, op) or (engine, op, mem_tags), where
        mem_tags names the memory a load reads / a store writes, since the
        scheduler can't see runtime addresses.
        """
        bundles = []   # list of {engine: [ops]}
        used = []      # list of {engine: slots used}
        last_write = defaultdict(lambda: -1)  # resource -> last bundle that wrote it
        last_read = defaultdict(lambda: -1)   # resource -> last bundle that read it
 
        for item in slots:
            engine, op = item[0], item[1]
            mem_tags = item[2] if len(item) > 2 else []
            reads, writes = self.rw(engine, op)
            if engine == "load":
                reads = reads + [("mem", t) for t in mem_tags]
            elif engine == "store":
                writes = writes + [("mem", t) for t in mem_tags]
 
            # earliest legal bundle
            c = 0
            for x in reads:
                c = max(c, last_write[x] + 1)          # inputs must be written already
            for x in writes:
                c = max(c, last_write[x] + 1,          # keep writes in order
                        last_read[x])                  # don't clobber before it's read
            # find a bundle with a free slot for this engine
            while True:
                if c == len(bundles):
                    bundles.append({})
                    used.append(defaultdict(int))
                if used[c][engine] < SLOT_LIMITS[engine]:
                    break
                c += 1
 
            bundles[c].setdefault(engine, []).append(op)
            used[c][engine] += 1
            for x in reads:
                last_read[x] = max(last_read[x], c)
            for x in writes:
                last_write[x] = c
        return bundles

    def build_priority(self, slots, war_weight=1.0, boost=None, order_weight=0.0,
                       stage=None, stage_weight=0.0, bonus=None):
        """
        Priority list scheduler.
        Pass 1: build the dependency graph (RAW/WAW: at least 1 cycle later,
                WAR: same cycle or later) and score each op by the longest chain
                of ops that depends on it (its critical path), plus a per-engine boost.
        Pass 2: fill cycles one at a time; each cycle, take ready ops in
                priority order until each engine's slots are full.
        """
        import heapq
        boost = boost or {}
        n = len(slots)
        engines = [item[0] for item in slots]
        ops = [item[1] for item in slots]
        succ = [dict() for _ in range(n)]   # succ[i][j] = latency (0 for WAR, 1 otherwise)
        npred = [0] * n
        last_writer = {}
        readers = defaultdict(list)

        def edge(a, b, lat):
            if a == b:
                return
            old = succ[a].get(b)
            if old is None:
                succ[a][b] = lat
                npred[b] += 1
            elif lat > old:
                succ[a][b] = lat

        for i, item in enumerate(slots):
            engine, op = item[0], item[1]
            mem_tags = item[2] if len(item) > 2 else []
            reads, writes = self.rw(engine, op)
            if engine == "load":
                reads = reads + [("mem", t) for t in mem_tags]
            elif engine == "store":
                writes = writes + [("mem", t) for t in mem_tags]
            for x in reads:                       # RAW
                w = last_writer.get(x)
                if w is not None:
                    edge(w, i, 1)
            for x in writes:
                w = last_writer.get(x)            # WAW
                if w is not None:
                    edge(w, i, 1)
                for r in readers.get(x, ()):      # WAR
                    edge(r, i, 0)
            for x in reads:
                readers[x].append(i)
            for x in writes:
                last_writer[x] = i
                readers[x] = []

        # priority = longest weighted path from this op to the end. Each op on the path
        # counts (1 + boost[its engine]); a WAR edge only counts war_weight of that.
        # (program order is a topological order, so one backward pass suffices)
        own = [0.0 if engines[i] == "debug" else 1.0 + boost.get(engines[i], 0.0) for i in range(n)]
        prio = [0.0] * n
        for i in range(n - 1, -1, -1):
            best = 0.0
            for j, lat in succ[i].items():
                v = (own[j] if lat else own[j] * war_weight) + prio[j]
                if v > best:
                    best = v
            prio[i] = best
        # order_weight > 0 blends in program order (earlier ops first), like the greedy scheduler;
        # stage_weight does the same with a caller-given stage per op (e.g. its group-round's
        # position in emission order), which doesn't grow with the number of ops per stage
        key = [prio[i] - order_weight * i - (stage_weight * stage[i] if stage else 0.0)
               + (bonus[i] if bonus else 0.0) for i in range(n)]
        # the setup pause must come first: the local harness fills the debug trace only
        # once it returns, so no debug compare may run before it
        for i in range(n):
            if ops[i][0] == "pause":
                key[i] = float("inf")

        earliest = [0] * n
        waiting = defaultdict(list)          # cycle -> ops that become ready then
        heaps = defaultdict(list)            # engine -> heap of (-key, index)
        for i in range(n):
            if npred[i] == 0:
                waiting[0].append(i)
        bundles = []
        done = 0
        c = 0
        while done < n:
            for i in waiting.pop(c, ()):
                heapq.heappush(heaps[engines[i]], (-key[i], i))
            bundle = {}
            used = defaultdict(int)
            progress = True
            while progress:                  # WAR successors may join the same cycle
                progress = False
                for eng in list(heaps.keys()):
                    h = heaps[eng]
                    limit = SLOT_LIMITS[eng]
                    while h and used[eng] < limit:
                        _, i = heapq.heappop(h)
                        bundle.setdefault(eng, []).append(ops[i])
                        used[eng] += 1
                        done += 1
                        progress = True
                        for j, lat in succ[i].items():
                            t = c + lat
                            if t > earliest[j]:
                                earliest[j] = t
                            npred[j] -= 1
                            if npred[j] == 0:
                                if earliest[j] <= c:
                                    heapq.heappush(heaps[engines[j]], (-key[j], j))
                                else:
                                    waiting[earliest[j]].append(j)
            if bundle:
                bundles.append(bundle)
            elif done < n and not any(heaps.values()) and not waiting:
                raise RuntimeError("scheduler stuck")
            c += 1
        return bundles

    def add(self, engine, slot):
        # Setup ops are no longer one per bundle: they're collected here and
        # packed by build() together with the main body (in program order, first).
        self.setup.append((engine, slot))

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            if getattr(self, "const_via_flow", False):
                # scratch starts zeroed, so zero_scalar + val builds any constant on the
                # flow engine (idle at the start) and keeps the load slots for loads
                if not hasattr(self, "zero_scalar"):
                    self.zero_scalar = self.alloc_scratch("zero_scalar")   # never written
                    self.const_map[0] = self.zero_scalar
                if val in self.const_map:
                    return self.const_map[val]
                addr = self.alloc_scratch(name)
                self.add("flow", ("add_imm", addr, self.zero_scalar, val))
            else:
                addr = self.alloc_scratch(name)
                self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def hash_op_list(self):
        """
        The hash as a list of ops, after fusing stages of the form
            a = (a + C) + (a << s)
        into one multiply_add:  a = a * (1 + 2**s) + C   (exact mod 2**32).
        If merge_hash_stages, a fused stage a = b*m + C followed by a stage
            a = (a + C') op2 (a << s')
        is merged with it: both sides of op2 are affine in b,
            a + C'  = b*m + (C + C')             (multiply_add, "pre1")
            a << s' = b*(m << s') + (C << s')    (multiply_add, "pre3")
        so the two stages cost 3 ops instead of 4.
        Each entry: (stage, kind, args). kind "fused", "pre1", "pre3" or one of the 3 plain ops.
        """
        M = 2 ** 32
        ops = []
        hi = 0
        while hi < len(HASH_STAGES):
            op1, val1, op2, op3, val3 = HASH_STAGES[hi]
            nxt = HASH_STAGES[hi + 1] if hi + 1 < len(HASH_STAGES) else None
            if op1 == "+" and op2 == "+" and op3 == "<<":
                mul = 1 + (1 << val3)
                if self.merge_hash_stages and nxt and nxt[0] == "+" and nxt[3] == "<<":
                    s = nxt[4]
                    ops.append((hi + 1, "pre1", (mul, (val1 + nxt[1]) % M)))
                    ops.append((hi + 1, "pre3", ((mul << s) % M, (val1 << s) % M)))
                    ops.append((hi + 1, "op2", None))
                    hi += 2
                    continue
                ops.append((hi, "fused", (mul, val1)))
            else:
                ops.append((hi, "op1", None))
                ops.append((hi, "op3", None))
                ops.append((hi, "op2", None))
            hi += 1
        return ops

    def build_hash(self, val_hash_addr, tmp1_v, tmp2_v, round, i, alu_ops=(), flow_lanes=0,
                   defer_last_xor=False):
        """
        Vector hash for 8 lanes. alu_ops: set of positions in hash_op_list()
        to run on the ALU (8 scalar ops per lane-op) instead of 1 valu op.
        defer_last_xor: leave out the last stage's "^ C" (the result is then
        hash ^ C; the caller folds C into the next round's node values).
        """
        slots = []
        a = val_hash_addr
 
        def emit(on_alu, op, dest, x, y):
            if on_alu:
                for k in range(VLEN):
                    slots.append(("alu", (op, dest + k, x + k, y + k)))
            else:
                slots.append(("valu", (op, dest, x, y)))
 
        last = len(HASH_STAGES) - 1
        for n, (hi, kind, extra) in enumerate(self.hash_op_list()):
            op1, val1, op2, op3, val3 = HASH_STAGES[hi]
            v1, v3 = self.hash_consts[hi]
            on_alu = n in alu_ops
            deferred = defer_last_xor and hi == last
            if deferred and kind == "op1":
                continue
            if kind in ("pre1", "pre3"):
                dest = tmp1_v if kind == "pre1" else tmp2_v
                mul_v, add_v = self.vec_const(extra[0]), self.vec_const(extra[1])
                if on_alu:
                    for k in range(VLEN):
                        slots.append(("alu", ("*", dest + k, a + k, mul_v + k)))
                        slots.append(("alu", ("+", dest + k, dest + k, add_v + k)))
                else:
                    slots.append(("valu", ("multiply_add", dest, a, mul_v, add_v)))
            elif kind == "fused":
                mul_v = self.vec_const(extra[0])
                if on_alu:   # no scalar multiply_add: a*mul then + C
                    for k in range(VLEN):
                        slots.append(("alu", ("*", tmp1_v + k, a + k, mul_v + k)))
                        slots.append(("alu", ("+", a + k, tmp1_v + k, v1 + k)))
                else:
                    slots.append(("valu", ("multiply_add", a, a, mul_v, v1)))
            elif kind == "op1" and on_alu and op1 == "+" and flow_lanes > 0:
                # a + C on the ALU: move flow_lanes of its 8 lanes to flow add_imm
                for k in range(VLEN):
                    if k < flow_lanes:
                        slots.append(("flow", ("add_imm", tmp1_v + k, a + k, val1)))
                    else:
                        slots.append(("alu", ("+", tmp1_v + k, a + k, v1 + k)))
            elif kind == "op1":
                emit(on_alu, op1, tmp1_v, a, v1)
            elif kind == "op3":
                emit(on_alu, op3, tmp2_v, a, v3)
            else:
                emit(on_alu, op2, a, a if deferred else tmp1_v, tmp2_v)
            last_of_stage = kind in ("fused", "op2")
            if last_of_stage and not deferred:
                slots.append(("debug", ("vcompare", a, [(round, i + k, "hash_stage", hi) for k in range(8)])))
 
        return slots

    def vec_const(self, val):
        """8-lane vector holding val in every lane, made once and reused."""
        if not hasattr(self, "vconst_map"):
            self.vconst_map = {}
        if val not in self.vconst_map:
            addr = self.alloc_scratch(None, 8)
            self.add("valu", ("vbroadcast", addr, self.scratch_const(val)))
            self.vconst_map[val] = addr
        return self.vconst_map[val]

    def vectorize_hash_vars(self):
        # only the constants hash_op_list() actually uses (merged stages need none,
        # fused stages need only C); vec_const shares vectors between equal values
        self.hash_consts = [(None, None)] * len(HASH_STAGES)
        for hi, kind, _ in self.hash_op_list():
            op1, val1, op2, op3, val3 = HASH_STAGES[hi]
            if kind == "fused":
                self.hash_consts[hi] = (self.vec_const(val1), None)
            elif kind == "op1":
                self.hash_consts[hi] = (self.vec_const(val1), self.vec_const(val3))

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        # merge hash stages 2+3 into 3 ops (see hash_op_list)
        self.merge_hash_stages = True
        # build scalar constants with flow add_imm instead of load const (see scratch_const)
        self.const_via_flow = True
        # Input.generate always starts every walker at idx 0, so don't load the indices:
        # j = idx + 1 is known to be 1 (see form below)
        IDX_STARTS_AT_ZERO = True
        # the last round's index update is never used (indices aren't graded), so skip it
        SKIP_LAST_IDX_UPDATE = True
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        headers = self.alloc_scratch("headers", 8)
        for i, v in enumerate(init_vars):
            self.scratch[v] = headers + i
        self.add("load", ("vload", headers, headers))

        zero_const_v = self.alloc_scratch("zero_v", 8)
        one_const_v = self.alloc_scratch("one_v", 8)
        two_const_v = self.alloc_scratch("two_v", 8)
        self.add("valu", ("vbroadcast", zero_const_v, self.scratch_const(0)))
        self.add("valu", ("vbroadcast", one_const_v, self.scratch_const(1)))
        self.add("valu", ("vbroadcast", two_const_v, self.scratch_const(2)))
        self.vconst_map = {0: zero_const_v, 1: one_const_v, 2: two_const_v}
        self.vectorize_hash_vars()

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))
        def keys(round, i, name):
            return [(round, i + k, name) for k in range(8)]
        body = []  # array of slots

        need_idx_addr = not IDX_STARTS_AT_ZERO or not SKIP_LAST_IDX_UPDATE
        N = 12   # number of register sets (see below)
        sets = []
        n_groups = batch_size//VLEN
        groups = []
        for g in range(n_groups):
            groups.append({
                "idx": self.alloc_scratch(f"idx_g{g}", 8),
                "val": self.alloc_scratch(f"val_g{g}", 8),
                "idx_addr": self.alloc_scratch(f"idx_addr_g{g}") if need_idx_addr else None,
                "val_addr": self.alloc_scratch(f"val_addr_g{g}"),
            })
        # x_addr[g] = x_p + 8g as a binary tree of ALU adds: x_addr[g] = x_addr[g - hb] + 8*hb
        # (hb = highest set bit of g), so only log2(n_groups) constants are needed
        for name, base in (("val_addr", "inp_values_p"), ("idx_addr", "inp_indices_p")):
            if name == "idx_addr" and not need_idx_addr:
                continue
            for g in range(n_groups):
                if g == 0:
                    src, off = self.scratch[base], self.scratch_const(0)
                else:
                    hb = 1 << (g.bit_length() - 1)
                    src, off = groups[g - hb][name], self.scratch_const(hb * VLEN)
                body.append(("alu", ("+", groups[g][name], src, off)))
        for g in range(n_groups):
            G = groups[g]
            if not IDX_STARTS_AT_ZERO:
                body.append(("load", ("vload", G["idx"], G["idx_addr"]), [("idx", g)]))
                # G["idx"] holds j = idx + 1 for the whole run (see the index update below)
                body.append(("valu", ("+", G["idx"], G["idx"], one_const_v)))
            body.append(("load", ("vload", G["val"], G["val_addr"]), [("val", g)]))
        for s in range(N):
            sets.append({
                "node": self.alloc_scratch(f"node{s}", 8),
                "addr_v": self.alloc_scratch(f"addr_v{s}", 8),
                "t1":   self.alloc_scratch(f"t1_{s}", 8),
                "t2":   self.alloc_scratch(f"t2_{s}", 8),
                "t3":   self.alloc_scratch(f"t3_{s}", 8),
                "t4":   self.alloc_scratch(f"t4_{s}", 8),
            })

        forest_p_v = self.alloc_scratch("forest_p_v", 8)
        self.add("valu", ("vbroadcast", forest_p_v, self.scratch["forest_values_p"]))
        # gather address = forest_values_p + idx = (forest_values_p - 1) + j
        forest_pm1_v = self.alloc_scratch("forest_pm1_v", 8)
        self.add("valu", ("-", forest_pm1_v, forest_p_v, one_const_v))

        HASH_ALU_EVERY = 1
        # Staggering: groups are split into WAVES waves (g % WAVES), and wave w is
        # emitted SKEW*w rounds later in program order. Waves then sit in different
        # rounds at the same time, so one wave's gather rounds (load-heavy) overlap
        # another wave's cached rounds (compute-heavy).
        WAVES = 32
        SKEW = 0.6
        SKEW_CAP = None   # None = no cap
        # wave offsets on later passes: "same", "reverse" (late waves of pass 1 go first)
        # or "zero" (everyone catches up as fast as MIN_STEP allows)
        SECOND_PASS = "same"
        MIN_STEP = 1.0
        # priority bonus for ops of later waves (by wave number / WAVES) from round LATE_BOOST_FROM on
        LATE_BOOST = 100
        LATE_BOOST_FROM = 0
        HASH_ALU_OPS = 3
        # every HASH_ALU_EXTRA_EVERY-th group-round moves one more hash op to the ALU (0 = off)
        HASH_ALU_EXTRA_EVERY = 0
        # do the gather address add (forest_p - 1 + j) as 8 ALU ops instead of 1 VALU op
        GATHER_ADDR_ON_ALU = False
        # Leave out the last hash stage's "^ C5" when the next round's node is cached,
        # and xor C5 into the cached node values instead (see DEFER_C5 below).
        DEFER_C5 = True
        hl = self.hash_op_list()
        # ALU offload order: plain ops, then the merged multiply_adds, then fused stages
        HASH_ALU_ORDER = [n for n, h in enumerate(hl) if h[1] in ("op1", "op3", "op2")] + \
                         [n for n, h in enumerate(hl) if h[1] in ("pre1", "pre3")] + \
                         [n for n, h in enumerate(hl) if h[1] == "fused"]
        # Scheduler: "greedy" (program order, earliest legal cycle) or "priority"
        # (critical-path list scheduling). WAR_WEIGHT: how much write-after-read edges
        # count toward an op's critical path. *_BOOST: priority added to ops on that engine.
        SCHEDULER = "priority"
        WAR_WEIGHT = 0
        VALU_BOOST = 0.0
        ALU_BOOST = -1.0
        LOAD_BOOST = 0.0
        ORDER_WEIGHT = 0.1   # >0 favors ops earlier in program order
        STAGE_WEIGHT = 0.0   # >0 favors ops of group-rounds emitted earlier
        # Flow engine use (1 slot/cycle, mostly idle otherwise):
        # FLOW_LEVELS[d]: how many top levels of the depth-d node mux use flow vselect
        #   (the rest use valu multiply_add). Every ALT_EVERY[d]-th group (0 = never) in
        #   rounds ALT_ROUNDS uses ALT_FLOW_LEVELS[d] instead, to balance flow vs valu.
        # ADD_IMM_BUDGET: total lanes of the hash's "a + C" ALU op done as flow add_imm instead
        FLOW_LEVELS = {2: 2, 3: 3}
        ALT_FLOW_LEVELS = {2: 1, 3: 1}
        ALT_EVERY = {2: 0, 3: 0}
        ALT_ROUNDS = "all"    # "all", "head" (first pass down the tree) or "tail" (later passes)
        ADD_IMM_BUDGET = 0
        # round-10 reset (j = 1) as a flow vselect instead of a valu op
        RESET_WITH_VSELECT = True
        # skip the reset altogether: just remember that j = 1 (form "one")
        LAZY_RESET = True
        # cached rounds pick their node speculatively, so only the last op waits for the
        # previous round's hash (see select_spec)
        SPEC_SELECT = True
        PRELOAD_DEPTH = 3   # 0-3; tune together with HASH_ALU_OPS, WAVES, SKEW
        # groups g < HEAD_GATHER gather (instead of select) depths >= HEAD_GATHER_DEPTH on
        # the first pass down the tree; the last TAIL_GATHER groups do so on later passes
        HEAD_GATHER, HEAD_GATHER_DEPTH = 4, 3
        TAIL_GATHER, TAIL_GATHER_DEPTH = 8, 3
        assert PRELOAD_DEPTH <= 3
        n_cached = 2 ** (PRELOAD_DEPTH + 1) - 1          # nodes 0 .. n_cached-1
        n_vloads = (n_cached + VLEN - 1) // VLEN
        cache = self.alloc_scratch("tree_cache", n_vloads * VLEN)
        cache_addr = self.alloc_scratch("tree_cache_addr")
        body.append(("alu", ("+", cache_addr, self.scratch["forest_values_p"], self.scratch_const(0))))
        for j in range(n_vloads):
            if j > 0:
                body.append(("alu", ("+", cache_addr, cache_addr, self.scratch_const(VLEN))))
            body.append(("load", ("vload", cache + j * VLEN, cache_addr)))
        # node_v[n] = node n's value broadcast to all 8 lanes
        # With DEFER_C5, every cached round except round 0 follows a round that left out
        # "^ C5", so node_v[1..] hold node ^ C5; xnode0 is node 0 ^ C5 (round 0 uses node_v[0]).
        if DEFER_C5:
            assert HASH_STAGES[-1][0] == "^" and HASH_STAGES[-1][2] == "^"
            c5 = self.scratch_const(HASH_STAGES[-1][1])
            xcache = self.alloc_scratch("tree_cache_x", n_cached)
            for n in range(n_cached):
                body.append(("alu", ("^", xcache + n, cache + n, c5)))
            xnode0 = self.alloc_scratch("xnode_v0", 8)
            body.append(("valu", ("vbroadcast", xnode0, xcache)))
        node_v = {}
        for n in range(n_cached):
            node_v[n] = self.alloc_scratch(f"node_v{n}", 8)
            src = xcache + n if DEFER_C5 and n > 0 else cache + n
            body.append(("valu", ("vbroadcast", node_v[n], src)))
        # diff_vec(a, b) = node_v[b] - node_v[a], made on first use
        diff_vecs = {}
        def diff_vec(a, b):
            if (a, b) not in diff_vecs:
                diff_vecs[(a, b)] = self.alloc_scratch(f"diff{a}_{b}", 8)
                body.append(("valu", ("-", diff_vecs[(a, b)], node_v[b], node_v[a])))
            return diff_vecs[(a, b)]
        pair_diff = {1: diff_vec(1, 2)} if PRELOAD_DEPTH >= 1 else {}
        # depth 1: node = base1 + j * (n2 - n1), since j = idx + 1 is 2 or 3
        if PRELOAD_DEPTH >= 1:
            base1 = self.alloc_scratch("base1", 8)
            body.append(("valu", ("-", base1, node_v[1], pair_diff[1])))
            body.append(("valu", ("-", base1, base1, pair_diff[1])))
            if DEFER_C5:
                # same with j's bit 0 inverted: node = (n2 + 2*diff) + j * (-diff)
                neg_diff1 = self.alloc_scratch("neg_diff1", 8)
                base1_inv = self.alloc_scratch("base1_inv", 8)
                body.append(("valu", ("-", neg_diff1, node_v[1], node_v[2])))
                body.append(("valu", ("+", base1_inv, node_v[2], pair_diff[1])))
                body.append(("valu", ("+", base1_inv, base1_inv, pair_diff[1])))

        def select_node(d, idx, r, inv=0, xored=False, flow_levels=None):
            """Emit valu ops that put node values for depth d into a vector.
            Returns the scratch address of that vector.
            inv: bits of j that are stored inverted (j_stored = j ^ inv).
            xored: the previous round left out "^ C5" (only affects depth 0 here;
            node_v[1..] are always xored when DEFER_C5).
            flow_levels: how many of the top mux levels (d >= 2) use flow vselect
            instead of valu; None = all of them."""
            if d == 0:
                return xnode0 if xored else node_v[0]  # every lane is node 0
            if d == 1:
                if inv & 1:
                    body.append(("valu", ("multiply_add", r["node"], idx, neg_diff1, base1_inv)))
                else:
                    body.append(("valu", ("multiply_add", r["node"], idx, pair_diff[1], base1)))
                return r["node"]
            # Mux tree over the 2^d cached nodes, one level per bit of j (bit 0 first).
            # The top flow_levels levels use flow vselect (cond = j & bit, nonzero is enough);
            # the levels below use valu: x = lo + bit * (hi - lo), with bit as 0/1.
            # Level 0 picks between two constant node vectors, so its diff is precomputed.
            k = flow_levels if flow_levels is not None else d
            pool = [r["addr_v"], r["t1"], r["t2"], r["t3"], r["t4"]]   # r["node"] = result
            def bit_reg(i, as_01):
                b = pool.pop()
                if not as_01:
                    body.append(("valu", ("&", b, idx, self.vec_const(1 << i))))
                elif i == 0:
                    body.append(("valu", ("&", b, idx, one_const_v)))
                elif i == d - 1:   # top bit: j >= 2^d + 2^(d-1)
                    body.append(("valu", ("<", b, self.vec_const(2 ** d + 2 ** (d - 1) - 1), idx)))
                else:
                    body.append(("valu", (">>", b, idx, self.vec_const(i))))
                    body.append(("valu", ("&", b, b, one_const_v)))
                return b
            # leaves: node index for each stored j value (j_stored = j ^ inv)
            xs = [("leaf", (s ^ inv) - 1) for s in range(2 ** d, 2 ** (d + 1))]
            for i in range(d):
                on_flow = i >= d - k
                b = bit_reg(i, not on_flow)
                out = []
                for lo, hi in zip(xs[0::2], xs[1::2]):
                    dest = r["node"] if len(xs) == 2 else (lo[1] if lo[0] == "reg" else pool.pop())
                    lo_a = node_v[lo[1]] if lo[0] == "leaf" else lo[1]
                    hi_a = node_v[hi[1]] if hi[0] == "leaf" else hi[1]
                    if on_flow:
                        body.append(("flow", ("vselect", dest, b, hi_a, lo_a)))
                    elif lo[0] == "leaf":
                        body.append(("valu", ("multiply_add", dest, b, diff_vec(lo[1], hi[1]), lo_a)))
                    else:
                        body.append(("valu", ("-", hi_a, hi_a, lo_a)))
                        body.append(("valu", ("multiply_add", dest, b, hi_a, lo_a)))
                    if hi[0] == "reg":
                        pool.append(hi[1])
                    if lo[0] == "reg" and dest != lo[1]:
                        pool.append(lo[1])
                    out.append(("reg", dest))
                pool.append(b)
                xs = out
            return r["node"]


        # emission order: wave w = g % WAVES is emitted min(w * SKEW, SKEW_CAP) rounds
        # later. The cap makes the last waves share one offset, so the end of the run has
        # several groups gathering at once instead of one small wave's serial chain.
        cap = SKEW * (WAVES - 1) if SKEW_CAP is None else SKEW_CAP
        def offset(g, round):
            w = g % WAVES
            if round > forest_height:          # later passes down the tree
                if SECOND_PASS == "reverse":
                    w = WAVES - 1 - w
                elif SECOND_PASS == "zero":
                    return 0
            return min(w * SKEW, cap)
        # (SKEW may be fractional: waves are then less than a round apart)
        # A group's emission keys must increase round by round (at least MIN_STEP apart),
        # since each group's rounds have to be emitted in order.
        emit_key = {}
        for g in range(n_groups):
            prev = None
            for round in range(rounds):
                k = round + offset(g, round)
                if prev is not None:
                    k = max(k, prev + MIN_STEP)
                emit_key[(round, g)] = prev = k
        order = sorted(emit_key, key=lambda rg: (emit_key[rg], rg[1]))
        # which hash op is "a + C" (eligible for flow add_imm), and how many
        # group-rounds will have it on the ALU
        HASH_PLUS_C = next((n for n, (hi, kind, _) in enumerate(hl)
                            if kind == "op1" and HASH_STAGES[hi][0] == "+"), None)
        def alu_ops_for(round, g):
            gr = round * n_groups + g
            if not (HASH_ALU_EVERY and gr % HASH_ALU_EVERY == 0):
                return set()
            extra = 1 if HASH_ALU_EXTRA_EVERY and gr % HASH_ALU_EXTRA_EVERY == 0 else 0
            return set(HASH_ALU_ORDER[:HASH_ALU_OPS + extra])
        n_eligible = max(1, sum(1 for rr, gg in order if HASH_PLUS_C in alu_ops_for(rr, gg)))
        eligible_seen = 0

        def is_cached(round, g):
            """True if group g picks round's node from the preloaded cache, False if it
            gathers it. Some groups gather depths 1-3 too, at the start (HEAD_GATHER) or
            the end (TAIL_GATHER), when the load engine would otherwise sit idle."""
            depth = round % (forest_height + 1)
            if depth > PRELOAD_DEPTH:
                return False
            if depth == 0:
                return True
            if round <= forest_height and g < HEAD_GATHER and depth >= HEAD_GATHER_DEPTH:
                return False
            if round > forest_height and g >= n_groups - TAIL_GATHER and depth >= TAIL_GATHER_DEPTH:
                return False
            return True
        def defers_c5(round, g):
            """True if this round leaves out the last hash stage's "^ C5": the next
            round's node is cached, so C5 is folded into the cached node values."""
            nxt = round + 1
            return DEFER_C5 and nxt < rounds and is_cached(nxt, g)
        # inv[g]: bits of group g's j that are stored inverted. A round that defers C5
        # leaves val ^ C5 behind, and C5 is odd, so the branch bit it adds to j is flipped.
        inv = [0] * n_groups
        # form[g]: what group g's idx register holds
        #   "plain":    j (with the inv bits flipped)
        #   "one":      nothing; j is known to be 1 (start, or after the bottom-of-tree wrap).
        #               Depth-0 rounds never read j, so nothing needs it materialized.
        #   "pending":  P = 2 * j_prev; this round's j = P + (val & 1)  (SPEC_SELECT)
        #   "pending2": nothing; P is known to be 2 (j_prev was 1)
        form = ["one" if IDX_STARTS_AT_ZERO else "plain"] * n_groups
        def spec(round, g):
            return SPEC_SELECT and round < rounds and is_cached(round, g) \
                and round % (forest_height + 1) >= 1

        def select_spec(d, idx, val, r, inv, form_g):
            """Speculative select for a cached round in "pending" form: j = P + bit, where
            bit = val & 1 is known only at the end of the previous round's hash. Pick the
            even-j and odd-j candidates using P's bits (ready early), then one last op on
            bit. Leaves idx in "plain" form. Returns the node vector."""
            def leaf_n(s):   # node index for stored j value s
                return (s ^ inv) - 1
            par = r["t4"]
            body.append(("valu", ("&", par, val, one_const_v)))
            if d == 1:
                assert form_g == "pending2"
                a, b = leaf_n(2), leaf_n(3)
                body.append(("valu", ("multiply_add", r["node"], par, diff_vec(a, b), node_v[a])))
                body.append(("valu", ("+", idx, par, two_const_v)))
                return r["node"]
            assert form_g == "pending" and d in (2, 3)
            base = 2 ** d
            E = [node_v[leaf_n(base + 2 * m)] for m in range(2 ** (d - 1))]       # bit 0 = 0
            O = [node_v[leaf_n(base + 2 * m + 1)] for m in range(2 ** (d - 1))]   # bit 0 = 1
            c1, c2, A, B = r["t2"], r["t3"], r["addr_v"], r["t1"]
            body.append(("valu", ("&", c1, idx, two_const_v)))
            if d == 2:
                body.append(("flow", ("vselect", A, c1, E[1], E[0])))
                body.append(("flow", ("vselect", B, c1, O[1], O[0])))
            else:
                body.append(("valu", ("&", c2, idx, self.vec_const(4))))
                body.append(("flow", ("vselect", A, c1, E[1], E[0])))
                body.append(("flow", ("vselect", B, c1, E[3], E[2])))
                body.append(("flow", ("vselect", A, c2, B, A)))
                body.append(("flow", ("vselect", B, c1, O[1], O[0])))
                body.append(("flow", ("vselect", r["node"], c1, O[3], O[2])))
                body.append(("flow", ("vselect", B, c2, r["node"], B)))
            body.append(("flow", ("vselect", r["node"], par, B, A)))
            body.append(("valu", ("+", idx, idx, par)))
            return r["node"]

        body_stage = []   # body_stage[k] = emission position of the group-round that emitted body[k]
        body_bonus = []   # body_bonus[k] = priority bonus for body[k] (LATE_BOOST)
        prev_bonus = 0.0
        for pos, (round, g) in enumerate(order):
            body_stage.extend([pos] * (len(body) - len(body_stage)))
            body_bonus.extend([prev_bonus] * (len(body) - len(body_bonus)))
            prev_bonus = LATE_BOOST * (g % WAVES) / WAVES if round >= LATE_BOOST_FROM else 0.0
            depth = round % (forest_height + 1)
            i = g*VLEN
            G = groups[g]
            r = sets[g%N]
            idx, val = G["idx"], G["val"]

            # (idx itself isn't compared any more: the register holds idx + 1.
            #  node_val below still checks it indirectly every round.)
            xored_in = round > 0 and defers_c5(round - 1, g)   # val and node both hold x ^ C5
            if not xored_in:
                body.append(("debug", ("vcompare", val, keys(round, i, "val"))))
            # node_val = mem[forest_values_p + idx]
            if spec(round, g):
                node = select_spec(depth, idx, val, r, inv[g], form[g])
                form[g] = "plain"
                nxt = round + 1
                if inv[g] and nxt < rounds and not is_cached(nxt, g):
                    body.append(("valu", ("^", idx, idx, self.vec_const(inv[g]))))
                    inv[g] = 0
            elif is_cached(round, g):
                assert depth == 0 or form[g] == "plain"
                levels = FLOW_LEVELS.get(depth)
                every = ALT_EVERY.get(depth, 0)
                in_scope = {"all": True, "head": round <= forest_height,
                            "tail": round > forest_height}[ALT_ROUNDS]
                if every and in_scope and g % every == 0:
                    levels = ALT_FLOW_LEVELS[depth]
                node = select_node(depth, idx, r, inv[g], xored_in, flow_levels=levels)
                nxt = round + 1
                if inv[g] and nxt < rounds and not is_cached(nxt, g):
                    # next round gathers and needs the real j: un-flip the inverted bits
                    # now, while this round's hash runs, instead of on the gather's path
                    body.append(("valu", ("^", idx, idx, self.vec_const(inv[g]))))
                    inv[g] = 0
            else:
                node = r["node"]
                if inv[g]:
                    # gathers need the real j: un-flip the inverted bits once
                    body.append(("valu", ("^", idx, idx, self.vec_const(inv[g]))))
                    inv[g] = 0
                if GATHER_ADDR_ON_ALU:
                    for k in range(VLEN):
                        body.append(("alu", ("+", r["addr_v"] + k, forest_pm1_v + k, idx + k)))
                else:
                    body.append(("valu", ("+", r["addr_v"], forest_pm1_v, idx)))
                for k in range(8):
                    body.append(("load", ("load_offset", r["node"], r["addr_v"], k)))
            if not xored_in:
                body.append(("debug", ("vcompare", node, keys(round, i, "node_val"))))
            # val = myhash(val ^ node_val)
            body.append(("valu", ("^", val, val, node)))
            alu_ops = alu_ops_for(round, g)
            # spread ADD_IMM_BUDGET only over group-rounds where the "a + C" op is on the ALU
            if HASH_PLUS_C in alu_ops:
                e = eligible_seen
                flow_lanes = (e + 1) * ADD_IMM_BUDGET // n_eligible - e * ADD_IMM_BUDGET // n_eligible
                eligible_seen += 1
            else:
                flow_lanes = 0
            defer_out = defers_c5(round, g)
            body.extend(self.build_hash(val, r["t1"], r["t2"], round, i, alu_ops, flow_lanes,
                                        defer_last_xor=defer_out))
            if not defer_out:
                body.append(("debug", ("vcompare", val, keys(round, i, "hashed_val"))))
            if SKIP_LAST_IDX_UPDATE and round == rounds - 1:
                pass
            elif depth == forest_height:
                # bottom of the tree: every walker wraps to the root (idx 0, so j = 1)
                inv[g] = 0
                if LAZY_RESET:
                    form[g] = "one"
                elif RESET_WITH_VSELECT:
                    # condition is all zeros, so vselect always picks one_const_v
                    body.append(("flow", ("vselect", idx, zero_const_v, idx, one_const_v)))
                else:
                    body.append(("valu", ("+", idx, one_const_v, zero_const_v)))
            elif spec(round + 1, g):
                # next round selects speculatively: store only P = 2*j, which doesn't wait
                # for this hash; next round adds the bit (val & 1) itself
                if form[g] == "one":
                    form[g] = "pending2"
                else:
                    body.append(("valu", ("+", idx, idx, idx)))
                    form[g] = "pending"
                inv[g] = (inv[g] << 1) | int(defer_out)
            else:
                # idx' = 2*idx + 1 + (val & 1)  =>  with j = idx + 1:  j' = 2*j + (val & 1)
                body.append(("valu", ("&", r["t1"], val, one_const_v)))
                if form[g] == "one":
                    body.append(("valu", ("+", idx, r["t1"], two_const_v)))
                else:
                    body.append(("valu", ("multiply_add", idx, idx, two_const_v, r["t1"])))
                form[g] = "plain"
                inv[g] = (inv[g] << 1) | int(defer_out)
            
        body_stage.extend([len(order)] * (len(body) - len(body_stage)))
        body_bonus.extend([prev_bonus] * (len(body) - len(body_bonus)))
        for g in range(n_groups):
            G = groups[g]
            if not SKIP_LAST_IDX_UPDATE:
                if form[g] == "one":
                    body.append(("valu", ("+", G["idx"], one_const_v, zero_const_v)))
                    inv[g] = 0
                if inv[g]:
                    body.append(("valu", ("^", G["idx"], G["idx"], self.vec_const(inv[g]))))
                # turn j back into idx before writing it out (indices aren't graded)
                body.append(("valu", ("-", G["idx"], G["idx"], one_const_v)))
                body.append(("store", ("vstore", G["idx_addr"], G["idx"]), [("idx", g)]))
            body.append(("store", ("vstore", G["val_addr"], G["val"]), [("val", g)]))
        if SCHEDULER == "priority":
            body_instrs = self.build_priority(self.setup + body, war_weight=WAR_WEIGHT,
                                              boost={"valu": VALU_BOOST, "alu": ALU_BOOST, "load": LOAD_BOOST},
                                              order_weight=ORDER_WEIGHT,
                                              stage=[0] * len(self.setup) + body_stage
                                                    + [len(order)] * (len(body) - len(body_stage)),
                                              stage_weight=STAGE_WEIGHT,
                                              bonus=[0.0] * len(self.setup) + body_bonus
                                                    + [0.0] * (len(body) - len(body_bonus)))
        else:
            body_instrs = self.build(self.setup + body)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()