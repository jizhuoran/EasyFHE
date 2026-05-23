from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class ChebyshevPSNode:
    k: int
    m: int
    divqr_q: np.ndarray
    divcs_q: np.ndarray
    s2: np.ndarray
    q_node: object | None = None
    s_node: object | None = None


@dataclass(frozen=True)
class BootstrapApproxPlan:
    secret_key_dist: str
    chebyshev_k: int
    chebyshev_m: int
    ps_root: ChebyshevPSNode
    coefficients: np.ndarray
    double_angle_iterations: int
    depth: int
    message_scaling_factor: float


KIND_C = 0
KIND_Q = 1
KIND_S = 2

SPACE_SMALL = 0
SPACE_NODE = 1

ALIGN_NONE = 0
ALIGN_C_TO_BASE = 1
ALIGN_S_DROP_TO_NOISE_ONE = 2

Q_HIGHEST_NONE = 0
Q_HIGHEST_ROOT_DOUBLE = 1
Q_HIGHEST_ROOT_REPEAT = 2
Q_HIGHEST_SCALAR = 3


@dataclass(frozen=True)
class FlatPSTailSpec:
    kind: int
    path: tuple[str, ...]
    scalar_path: tuple[str, ...]
    coefficients: tuple[float, ...]
    deg: int
    out_idx: int


@dataclass(frozen=True)
class FlatPSSmallSpec:
    kind: int
    path: tuple[str, ...]
    scalar_path: tuple[str, ...]
    root: bool
    out_idx: int
    tail_idx: int | None
    direct_t1: bool
    k: int
    m: int
    const_value: float
    align_policy: int
    q_highest_mode: int
    q_highest_repeat: int
    q_highest_scalar_value: int | None


@dataclass(frozen=True)
class FlatPSCombineSpec:
    path: tuple[str, ...]
    root: bool
    base_idx: int
    c_const_scalar_path: tuple[str, ...]
    c_ref: tuple[int, int]
    q_ref: tuple[int, int]
    s_ref: tuple[int, int]
    out_idx: int


@dataclass(frozen=True)
class FlatPSPlan:
    k: int
    m: int
    tail_specs: tuple[FlatPSTailSpec, ...]
    tail_max_deg: int
    small_specs: tuple[FlatPSSmallSpec, ...]
    combine_specs: tuple[FlatPSCombineSpec, ...]
    root_ref: tuple[int, int]
    node_count: int


def degree(lst):
    for i in range(len(lst) - 1, -1, -1):
        if lst[i] != 0:
            return i
    return 0


def _truncated_degree(coefficients, size):
    truncated = np.copy(coefficients[: int(size)])
    truncated.resize(int(size), refcheck=False)
    return degree(truncated)


def _small_coefficients_and_path(kind, node, path):
    if kind == KIND_C:
        return node.divcs_q, (*path, "c"), None
    if kind == KIND_Q:
        return node.divqr_q, (*path, "q"), node.k
    if kind == KIND_S:
        return node.s2, (*path, "s"), node.k
    raise ValueError(f"unknown flat PS small spec kind: {kind}")


def _small_align_policy(kind, root):
    if root:
        return ALIGN_NONE
    if kind == KIND_C:
        return ALIGN_C_TO_BASE
    if kind == KIND_S:
        return ALIGN_S_DROP_TO_NOISE_ONE
    return ALIGN_NONE


def _q_highest_mode_and_value(node, root, has_tail):
    if root:
        if has_tail:
            return Q_HIGHEST_ROOT_DOUBLE, 0, None
        return Q_HIGHEST_ROOT_REPEAT, int(node.divqr_q[-1]), None

    coefficient = node.divqr_q[-1] + (1.1 if has_tail else 0.0)
    scalar = int(2 ** math.floor(math.log2(coefficient)))
    return Q_HIGHEST_SCALAR, 0, scalar


def long_division_chebyshev(f, g):
    if math.isclose(f[-1], 0) or math.isclose(g[-1], 0):
        raise ValueError(
            f"Chebyshev division requires nonzero dominant coefficients, got f[-1]={f[-1]}, g[-1]={g[-1]}"
        )
    n, k = len(f) - 1, len(g) - 1

    if n < k:
        return np.array([1.0]), np.array(f)

    q = np.zeros(n - k + 1)
    r = np.copy(f)
    d = np.zeros(len(g) + n)

    while n > k:
        q[n - k] = 2 * r[-1] / g[-1]
        d = np.zeros(n + 1)
        if k == (n - k):
            d[0] = 2 * g[n - k]
            for i in range(1, 2 * k + 1):
                d[i] = g[abs(n - k - i)]
        else:
            if k > (n - k):
                d[0] = 2 * g[n - k]
                for i in range(1, k - (n - k) + 1):
                    d[i] = g[abs(n - k - i)] + g[n - k + i]
                for i in range(k - (n - k) + 1, n + 1):
                    d[i] = g[abs(i - n + k)]
            else:
                d[n - k] = g[0]
                for i in range(n - 2 * k, n + 1):
                    if i != n - k:
                        d[i] = g[abs(i - n + k)]

        r = r - d * r[-1] / g[-1]
        if len(r) > 1:
            n = degree(r)
            r.resize(n + 1, refcheck=False)

    if n == k:
        q[0] = r[-1] / g[-1]
        r = r - g * q[0]
        if len(r) > 1:
            n = degree(r)
            r.resize(n + 1, refcheck=False)

    q[0] *= 2
    return q, r


def _build_inner_ps_node(coefficients, k, m):
    k2m2k = k * (1 << (m - 1)) - k

    Tkm = np.zeros(int(k2m2k + k) + 1)
    Tkm[-1] = 1.0
    divqr_q, divqr_r = long_division_chebyshev(coefficients, Tkm)

    r2 = np.copy(divqr_r)
    if (int(k2m2k - degree(divqr_r)) <= 0):
        r2[k2m2k] -= 1
        r2.resize(degree(r2) + 1, refcheck=False)
    else:
        r2.resize(k2m2k + 1, refcheck=False)
        r2[-1] = -1

    divcs_q, divcs_r = long_division_chebyshev(r2, divqr_q)

    s2 = np.copy(divcs_r)
    s2.resize(k2m2k + 1, refcheck=False)
    s2[-1] = 1.0

    q_node = _build_inner_ps_node(divqr_q, k, m - 1) if degree(divqr_q) > k else None
    s_node = _build_inner_ps_node(s2, k, m - 1) if degree(s2) > k else None
    return ChebyshevPSNode(
        k=int(k),
        m=int(m),
        divqr_q=divqr_q,
        divcs_q=divcs_q,
        s2=s2,
        q_node=q_node,
        s_node=s_node,
    )


def _build_root_ps_node(k, m, divqr_q, divcs_q, s2):
    return ChebyshevPSNode(
        k=int(k),
        m=int(m),
        divqr_q=divqr_q,
        divcs_q=divcs_q,
        s2=s2,
        q_node=_build_inner_ps_node(divqr_q, k, m - 1) if degree(divqr_q) > k else None,
        s_node=_build_inner_ps_node(s2, k, m - 1) if degree(s2) > k else None,
    )


def compile_flat_ps_plan(root: ChebyshevPSNode) -> FlatPSPlan:
    tail_specs = []
    small_specs = []
    combine_specs = []

    def add_small(kind, node, path, root):
        coefficients, scalar_path, size = _small_coefficients_and_path(kind, node, path)
        deg = degree(coefficients) if size is None else _truncated_degree(coefficients, size)
        direct_t1 = bool(kind == KIND_C and deg == 1 and coefficients[1] == 1)
        tail_idx = None
        if deg >= 1 and not direct_t1:
            tail_idx = len(tail_specs)
            tail_specs.append(
                FlatPSTailSpec(
                    kind=int(kind),
                    path=tuple(path),
                    scalar_path=tuple(scalar_path),
                    coefficients=tuple(float(value) for value in coefficients),
                    deg=int(deg),
                    out_idx=tail_idx,
                )
            )

        q_mode = Q_HIGHEST_NONE
        q_repeat = 0
        q_scalar_value = None
        if kind == KIND_Q:
            q_mode, q_repeat, q_scalar_value = _q_highest_mode_and_value(
                node,
                root,
                tail_idx is not None,
            )

        out_idx = len(small_specs)
        small_specs.append(
            FlatPSSmallSpec(
                kind=int(kind),
                path=tuple(path),
                scalar_path=tuple(scalar_path),
                root=bool(root),
                out_idx=out_idx,
                tail_idx=tail_idx,
                direct_t1=direct_t1,
                k=int(node.k),
                m=int(node.m),
                const_value=float(coefficients[0] / 2),
                align_policy=_small_align_policy(kind, root),
                q_highest_mode=q_mode,
                q_highest_repeat=q_repeat,
                q_highest_scalar_value=q_scalar_value,
            )
        )
        return (SPACE_SMALL, out_idx)

    def compile_node(node, path, root):
        c_ref = add_small(KIND_C, node, path, root)
        if node.q_node is None:
            q_ref = add_small(KIND_Q, node, path, root)
        else:
            q_ref = compile_node(node.q_node, (*path, "q_node"), False)

        if node.s_node is None:
            s_ref = add_small(KIND_S, node, path, root)
        else:
            s_ref = compile_node(node.s_node, (*path, "s_node"), False)

        out_idx = len(combine_specs)
        combine_specs.append(
            FlatPSCombineSpec(
                path=tuple(path),
                root=bool(root),
                base_idx=int(node.m) - 1,
                c_const_scalar_path=tuple((*path, "c")),
                c_ref=c_ref,
                q_ref=q_ref,
                s_ref=s_ref,
                out_idx=out_idx,
            )
        )
        return (SPACE_NODE, out_idx)

    root_ref = compile_node(root, ("root",), True)
    return FlatPSPlan(
        k=int(root.k),
        m=int(root.m),
        tail_specs=tuple(tail_specs),
        tail_max_deg=max((spec.deg for spec in tail_specs), default=0),
        small_specs=tuple(small_specs),
        combine_specs=tuple(combine_specs),
        root_ref=root_ref,
        node_count=len(combine_specs),
    )


def describe_flat_ps_plan(flat: FlatPSPlan, bootstrap_plan=None) -> str:
    lines = [
        "Approx PS Plan",
        (
            f"k={flat.k} m={flat.m} "
            f"tails={len(flat.tail_specs)} max_deg={flat.tail_max_deg} "
            f"small={len(flat.small_specs)} combine={len(flat.combine_specs)}"
        ),
        "",
        "Tails",
    ]
    for spec in flat.tail_specs:
        lines.append(
            f"  tail[{spec.out_idx:02d}] {_kind_name(spec.kind)} "
            f"{_path_text(spec.scalar_path)} deg={spec.deg}"
        )

    lines.extend(["", "Small"])
    for spec in flat.small_specs:
        if spec.direct_t1:
            tail = "T1"
        elif spec.tail_idx is None:
            tail = "None"
        else:
            tail = f"tail[{spec.tail_idx:02d}]"
        pieces = [tail]
        pieces.append(f"+ const({_path_text(spec.scalar_path)})")
        if spec.q_highest_mode != Q_HIGHEST_NONE:
            pieces.append(f"+ {_q_highest_text(spec)}")
        if spec.kind == KIND_S:
            pieces.append("+ Tk")
        lines.append(
            f"  small[{spec.out_idx:02d}] {_kind_name(spec.kind)} "
            f"{_path_text(spec.scalar_path)} = {' '.join(pieces)}"
        )

    lines.extend(["", "Combine"])
    for spec in flat.combine_specs:
        lines.append(
            f"  node[{spec.out_idx:02d}] {_path_text(spec.path)} = "
            f"(T2[{spec.base_idx}] + {_ref_text(spec.c_ref)}) * "
            f"{_ref_text(spec.q_ref)} + {_ref_text(spec.s_ref)}"
        )

    if bootstrap_plan is not None:
        lines.extend(["", "Scalar Tables"])
        lines.append(
            f"  tail_scalar_rows={len(getattr(bootstrap_plan, 'approx_tail_scalar_names', ())) }"
        )
        lines.append(
            f"  chebyshev_neg_one={getattr(bootstrap_plan, 'chebyshev_neg_one_scalar_name', None)}"
        )
    return "\n".join(lines)


def _kind_name(kind):
    return {KIND_C: "C", KIND_Q: "Q", KIND_S: "S"}.get(kind, f"kind={kind}")


def _kind_suffix(kind):
    return {KIND_C: "c", KIND_Q: "q", KIND_S: "s"}[kind]


def _path_text(path):
    return ".".join(path)


def _ref_text(ref):
    space, idx = ref
    if space == SPACE_SMALL:
        return f"small[{idx:02d}]"
    if space == SPACE_NODE:
        return f"node[{idx:02d}]"
    return f"ref({space},{idx})"


def _q_highest_text(spec):
    if spec.q_highest_mode == Q_HIGHEST_ROOT_DOUBLE:
        return "2*Tk"
    if spec.q_highest_mode == Q_HIGHEST_ROOT_REPEAT:
        return f"{spec.q_highest_repeat}*Tk"
    if spec.q_highest_mode == Q_HIGHEST_SCALAR:
        return f"{spec.q_highest_scalar_value}*Tk"
    return "highest"


@lru_cache(maxsize=None)
def get_bootstrap_approx_plan(secret_key_dist):
    if secret_key_dist == "SPARSE_TERNARY":
        k = 7
        m = 3
        coefficients = np.array(
            [
                -0.18646470117093214, 0.036680543700430925, -0.20323558926782626, 0.029327390306199311,
                -0.24346234149506416, 0.011710240188138248, -0.27023281815251715, -0.017621188001030602,
                -0.21383614034992021, -0.048567932060728937, -0.013982336571484519, -0.051097367628344978,
                0.24300487324019346, 0.0016547743046161035, 0.23316923792642233, 0.060707936480887646,
                -0.18317928363421143, 0.0076878773048247966, -0.24293447776635235, -0.071417413140564698,
                0.37747441314067182, 0.065154496937795681, -0.24810721693607704, -0.033588418808958603,
                0.10510660697380972, 0.012045222815124426, -0.032574751830745423, -0.0032761730196023873,
                0.0078689491066424744, 0.00070965574480802061, -0.0015405394287521192, -0.00012640521062948649,
                0.00025108496615830787, 0.000018944629154033562, -0.000034753284216308228, -2.4309868106111825e-6,
                4.1486274737866247e-6, 2.7079833113674568e-7, -4.3245388569898879e-7, -2.6482744214856919e-8,
                3.9770028771436554e-8, 2.2951153557906580e-9, -3.2556026220554990e-9, -1.7691071323926939e-10,
                2.5459052150406730e-10
            ],
            dtype=np.float64,
        )
        divqr_q = np.array(
            [
                1.5737898213284949e-02,
                1.4193114896160412e-03,
                -3.0810788575042383e-03,
                -2.5281042125897298e-04,
                5.0216993231661574e-04,
                3.7889258308067125e-05,
                -6.9506568432616456e-05,
                -4.8619736212223649e-06,
                8.2972549475732493e-06,
                5.4159666227349137e-07,
                -8.6490777139797758e-07,
                -5.2965488429713838e-08,
                7.9540057542873108e-08,
                4.5902307115813160e-09,
                -6.5112052441109981e-09,
                -3.5382142647853878e-10,
                5.0918104300813460e-10,
                0.0,
                0.0,
                0.0,
                0.0,
                2.0000000000000000e00,
            ]
        )
        divcs_q = np.array(
            [
                -0.9348430720681978,
                -0.24807246365084598,
                -0.03360736343811264,
                0.10485552200765141,
                0.012171628025753911,
                -0.031034212401993305,
                -0.003985828764410408,
            ]
        )
        s2 = np.array(
            [
                -0.1788386282880803,
                0.03907676004440467,
                -0.20431837545501938,
                0.02803557926525301,
                -0.2434613107770007,
                0.012176766307331699,
                -0.2701672634577602,
                -1.0177001300236108,
                -0.2138490838407006,
                -0.048556142033877835,
                -0.013980273064777329,
                -0.05109888871842223,
                0.24300459668180657,
                0.0016549458716776414,
                0.23316927298939902,
                0.06469374592818429,
                -0.1521451141878234,
                -0.004483722738865048,
                -0.3477895670236423,
                -0.037810320617212255,
                0.6255427281424787,
                1.0,
            ]
        )
        double_angle_iterations = 3
        depth = 10
        message_scaling_factor = 1.0
    elif secret_key_dist == "UNIFORM_TERNARY":
        k = 6
        m = 4
        coefficients = np.array(
            [
                0.15421426400235561,
                -0.0037671538417132409,
                0.16032011744533031,
                -0.0034539657223742453,
                0.17711481926851286,
                -0.0027619720033372291,
                0.19949802549604084,
                -0.0015928034845171929,
                0.21756948616367638,
                0.00010729951647566607,
                0.21600427371240055,
                0.0022171399198851363,
                0.17647500259573556,
                0.0042856217194480991,
                0.086174491919472254,
                0.0054640252312780444,
                -0.046667988130649173,
                0.0047346914623733714,
                -0.17712686172280406,
                0.0016205080004247200,
                -0.22703114241338604,
                -0.0028145845916205865,
                -0.13123089730288540,
                -0.0056345646688793190,
                0.078818395388692147,
                -0.0037868875028868542,
                0.23226434602675575,
                0.0021116338645426574,
                0.13985510526186795,
                0.0059365649669377071,
                -0.13918475289368595,
                0.0018580676740836374,
                -0.23254376365752788,
                -0.0054103844866927788,
                0.056840618403875359,
                -0.0035227192748552472,
                0.25667909012207590,
                0.0055029673963982112,
                -0.073334392714092062,
                0.0027810273357488265,
                -0.24912792167850559,
                -0.0069524866497120566,
                0.21288810409948347,
                0.0017810057298691725,
                0.088760951809475269,
                0.0055957188940032095,
                -0.31937177676259115,
                -0.0087539416335935556,
                0.34748800245527145,
                0.0075378299617709235,
                -0.25116537379803394,
                -0.0047285674679876204,
                0.13970502851683486,
                0.0023672533925155220,
                -0.063649401080083698,
                -0.00098993213448982727,
                0.024597838934816905,
                0.00035553235917057483,
                -0.0082485030307578155,
                -0.00011176184313622549,
                0.0024390574829093264,
                0.000031180384864488629,
                -0.00064373524734389861,
                -7.8036008952377965e-6,
                0.00015310015145922058,
                1.7670804180220134e-6,
                -0.000033066844379476900,
                -3.6460909134279425e-7,
                6.5276969021754105e-6,
                6.8957843666189918e-8,
                -1.1842811187642386e-6,
                -1.2015133285307312e-8,
                1.9839339947648331e-7,
                1.9372045971100854e-9,
                -3.0815418032523593e-8,
                -2.9013806338735810e-10,
                4.4540904298173700e-9,
                4.0505136697916078e-11,
                -6.0104912807134771e-10,
                -5.2873323696828491e-12,
                7.5943206779351725e-11,
                6.4679566322060472e-13,
                -9.0081200925539902e-12,
                -7.4396949275292252e-14,
                1.0057423059167244e-12,
                8.1701187638005194e-15,
                -1.0611736208855373e-13,
                -8.9597492970451533e-16,
                1.1421575296031385e-14,
            ],
            dtype=np.float64,
        )
        divqr_q = np.array(
            [
                6.9497600491054290e-01,
                1.5075659923541847e-02,
                -5.0233074759606788e-01,
                -9.4571349359752407e-03,
                2.7941005703366972e-01,
                4.7345067850310439e-03,
                -1.2729880216016740e-01,
                -1.9798642689796545e-03,
                4.9195677869633810e-02,
                7.1106471834114966e-04,
                -1.6497006061515631e-02,
                -2.2352368627245097e-04,
                4.8781149658186527e-03,
                6.2360769728977259e-05,
                -1.2874704946877972e-03,
                -1.5607201790475593e-05,
                3.0620030291844116e-04,
                3.5341608360440269e-06,
                -6.6133688758953800e-05,
                -7.2921818268558850e-07,
                1.3055393804350821e-05,
                1.3791568733237984e-07,
                -2.3685622375284772e-06,
                -2.4030266570614624e-08,
                3.9678679895296662e-07,
                3.8744091942201708e-09,
                -6.1630836065047186e-08,
                -5.8027612677471620e-10,
                8.9081808596347400e-09,
                8.1010273395832156e-11,
                -1.2020982561426954e-09,
                -1.0574664739365698e-11,
                1.5188641355870345e-10,
                1.2935913264412094e-12,
                -1.8016240185107980e-11,
                -1.4879389855058450e-13,
                2.0114846118334488e-12,
                1.6340237527601039e-14,
                -2.1223472417710746e-13,
                -1.7919498594090307e-15,
                2.2843150592062770e-14,
                0.0000000000000000e00,
                2.0000000000000000e00,
            ]
        )
        divcs_q = np.array(
            [
                -7.2346249482043945e-01,
                -5.8624766264825747e-04,
                -5.0944076707358829e-02,
                1.0324286361991016e-02,
                -6.8206402964557211e-02,
                -1.6291771595364480e-02,
            ]
        )
        s2 = np.array(
            [
                3.9925918972600116e-01,
                4.1145403219203028e-03,
                -1.8235175231313211e-02,
                -1.2610653952719340e-02,
                2.8758646670247762e-01,
                7.1082491955162096e-03,
                -8.5570641881432696e-01,
                -7.9326949369856407e-03,
                2.4129727753976357e-01,
                3.3790703055063117e-03,
                2.0707743824677474e-01,
                8.0275114432424059e-04,
                1.7945499861112815e-01,
                4.8112939341447792e-03,
                8.5282908239589103e-02,
                5.2930062046256110e-03,
                -4.6427041843243977e-02,
                4.7840515232640164e-03,
                -1.7718605082654609e-01,
                1.6077363022353442e-03,
                -2.2701785866927471e-01,
                -2.8115966949872331e-03,
                -1.3123360552644656e-01,
                -5.6352028739135845e-03,
                7.8818717503706712e-02,
                -3.7867513899410860e-03,
                2.3226543877935715e-01,
                2.1115426273585532e-03,
                1.3984859250790091e-01,
                5.9369332741118662e-03,
                -1.3915168832121910e-01,
                1.8563000239210160e-03,
                -2.3269686348640464e-01,
                -5.4025808040419360e-03,
                5.7484353608334406e-02,
                -3.5538996706805082e-03,
                2.5424003264451761e-01,
                2.1906500836275766e-02,
                3.1205132805949077e-03,
                -7.8987913855752667e-03,
                -2.2278168390589351e-01,
                -5.3763068525559737e-03,
                1.0000000000000000e00,
            ]
        )
        double_angle_iterations = 6
        depth = 13
        message_scaling_factor = 512.0
    else:
        raise RuntimeError(f"unsupported bootstrap secret_key_dist: {secret_key_dist}")

    return BootstrapApproxPlan(
        secret_key_dist=str(secret_key_dist),
        chebyshev_k=int(k),
        chebyshev_m=int(m),
        ps_root=_build_root_ps_node(k, m, divqr_q, divcs_q, s2),
        coefficients=coefficients,
        double_angle_iterations=int(double_angle_iterations),
        depth=int(depth),
        message_scaling_factor=float(message_scaling_factor),
    )


def bootstrap_approx_depth(secret_key_dist):
    return get_bootstrap_approx_plan(secret_key_dist).depth
