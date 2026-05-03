import functools
import sys
from collections import defaultdict, OrderedDict
from typing import Any, DefaultDict, Literal, Mapping
import pprint


# ────────────────────────────────────────────────────────────────
# 1.  数据结构：stage → category → 累积字节数
# ────────────────────────────────────────────────────────────────
Stages      = Literal["middle", "end"]
Categories = Literal["app", "C2S", "S2C"]

_mem_table_encode: DefaultDict[Stages, DefaultDict[Categories, int]] = defaultdict(
    lambda: defaultdict(int)
)

_seen_name_table: set[tuple[Stages, Categories, str]] = set()



def encode_counter_detailed(func):
    """
    decorate encode(x, name, level, slots, is_ext, cryptoContext)
    record：
        • input :  could be of middle or end form
        • output    : should be of end form
    """
    @functools.wraps(func)
    def wrapper(x, name, level, slots, is_ext, cryptoContext):
        # ------- input -------
        if getattr(cryptoContext, 'pre_encode_type', False) == "middle" or cryptoContext.config.ENCODE_BS_FFT == False:
            log_mem_trace_encode("middle", cryptoContext, x.encoded_values, name)
        elif getattr(cryptoContext, 'pre_encode_type', False) == "end" or cryptoContext.config.ENCODE_BS_FFT == True:
            log_mem_trace_encode("end", cryptoContext, x.cv[0], name)

        # call encode func
        gpufhe_cipher = func(x, name, level, slots, is_ext, cryptoContext)

        # ------- output -------
        if getattr(cryptoContext, 'pre_encode_type', False) == "middle"  or cryptoContext.config.ENCODE_BS_FFT == False:
            log_mem_trace_encode("end", cryptoContext, gpufhe_cipher.cv[0], name)

        return gpufhe_cipher

    return wrapper




# ────────────────────────────────────────────────────────────────
# 2.  内部工具
# ────────────────────────────────────────────────────────────────
def _bytes(obj: Any) -> int:
    """优先用 .nbytes（NumPy / PyTorch），否则用 sys.getsizeof"""
    return obj.nbytes if hasattr(obj, "nbytes") else sys.getsizeof(obj)

def _infer_category(ctx: Any) -> Categories:
    """根据 cryptoContext 判定日志类别"""
    if getattr(ctx, "inBS", False):
        if getattr(ctx, "inC2S", False):
            return "C2S"
        if getattr(ctx, "inS2C", False):
            return "S2C"
    else:
        # 其余 BS 内步骤暂时也归为 app，可按需扩展
        return "app"

def _mb(bytes_cnt: int) -> float:
    return bytes_cnt / (1024 * 1024)

def _gb(bytes_cnt: int) -> float:
    return bytes_cnt / (1024 * 1024 * 1024)

# ────────────────────────────────────────────────────────────────
# 3.  对外 API
# ────────────────────────────────────────────────────────────────
def log_mem_trace_encode(stage: Stages, ctx: Any, data: Any, name: str | None = None) -> None:
    if stage not in ("middle", "end"):
        raise ValueError("stage must be 'middle' or 'end'")

    cat   = _infer_category(ctx)

    if name is not None:
        key = (stage, cat, name)
        if key in _seen_name_table:
            return
        _seen_name_table.add(key)

    bytes_ = _bytes(data)
    _mem_table_encode[stage][cat] += bytes_

# ────────────────────────────────────────────────────────────────
# 4.  查询 / 重置
# ────────────────────────────────────────────────────────────────
def get_stage_size(stage: Stages, *, category: Categories | None = None) -> float:
    """获取指定阶段（及可选类别）的统计值，单位 MB"""
    if category:
        return _mb(_mem_table_encode[stage][category])
    return _mb(sum(_mem_table_encode[stage].values()))

def reset_mem_trace_encode() -> None:
    _mem_table_encode.clear()
    _seen_name_table.clear()



# ────────────────────────────────────────────────────────────────
# 5.  pretty-print 为可复制的 dict（单位 MB）
# ────────────────────────────────────────────────────────────────


def _to_plain_dict(obj: Any) -> Any:
    """Recursively turn OrderedDict → dict so that the repr is a literal dict."""
    if isinstance(obj, OrderedDict):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, Mapping):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    return obj


def prt_dict_mem_trace_encode(
    dict_name: str,
    *,
    registry = _mem_table_encode,
    stage_order=("middle", "end"),
    cat_order=("app", "C2S", "S2C"),
    precision: int = 4,
) -> None:
    """
    打印一段 *纯字典文本*，包含:
        • middle / end 各类的累计 MB
        • total_middle / total_end 两个聚合键
    """
    snapshot = OrderedDict()

    total_mid_mb = total_end_mb = 0.0
    for stage in stage_order:
        stage_dict = OrderedDict()
        for cat in cat_order:
            if cat in registry[stage]:
                mb = get_stage_size(stage, category=cat)
                stage_dict[cat] = round(mb, precision)
        snapshot[stage] = stage_dict

        stage_total = round(get_stage_size(stage), precision)
        if stage == "middle":
            total_mid_mb = stage_total
        else:
            total_end_mb = stage_total

    snapshot["total_middle"] = total_mid_mb
    snapshot["total_end"]    = total_end_mb

    plain = _to_plain_dict(snapshot)
    print(f"{dict_name} = {pprint.pformat(plain, sort_dicts=False, width=120)}")

# ────────────────────────────────────────────────────────────────
# 6.  控制台查看 total_middle / total_end
# ────────────────────────────────────────────────────────────────
def print_totals_mb() -> None:
    """仅打印 total_middle & total_end（单位 MB / GB）"""
    mid_mb = get_stage_size("middle")
    end_mb = get_stage_size("end")
    print(f"[total_middle] {mid_mb:10.4f} MB  ({mid_mb/1024:7.4f} GB)")
    print(f"[total_end]    {end_mb:10.4f} MB  ({end_mb/1024:7.4f} GB)")
