
from decimal import Decimal, getcontext
from math import ceil, floor, log10
from typing import Callable, List
from .Point import Point

getcontext().prec = 100  # 可调整精度

def pow2(n: int) -> int:
    return 1 << n

def ceil_to_int(x: float) -> int:
    return int(ceil(x) + 0.5)

def floor_to_int(x: float) -> int:
    return int(floor(x) + 0.5)

def log2_long(n: int) -> int:
    if n > 65536 or n <= 0:
        raise ValueError("n is too large.")
    for i in range(17):
        if pow2(i) == n:
            return i
    return -1

def num_one(n: int) -> int:
    return bin(n).count('1')

def sgn(x: Decimal) -> Decimal:
    return Decimal(1) if x > 0 else Decimal(-1) if x < 0 else Decimal(0)

def fracpart(x: Decimal) -> Decimal:
    return x - x.to_integral_value()

def ReLU(x):
    if isinstance(x, Decimal):
        return x if x > 0 else Decimal(0)
    else:
        return x if x > 0 else 0.0

def eval_poly(deg: int, coeff: List[Decimal], val: Decimal, type: int, scale: Decimal) -> Decimal:
    if type == 0:
        tmp = Decimal(1)
        rtn = coeff[0] * tmp
        for i in range(1, deg + 1):
            tmp *= val
            rtn += coeff[i] * tmp
        return rtn
    elif type == 1:
        if deg == 0:
            return Decimal(1)
        tmp1 = Decimal(1)
        tmp2 = val
        rtn = coeff[0] * tmp1 + coeff[1] * tmp2
        for i in range(2, deg + 1):
            tmp3 = Decimal(2) * val * tmp2 - tmp1
            tmp1, tmp2 = tmp2, tmp3
            rtn += coeff[i] * tmp3
        return rtn
    elif type == 2:
        if deg == 0:
            return Decimal(1)
        tmp1 = Decimal(1)
        tmp2 = val / scale
        iden2 = Decimal(2) * val / scale
        rtn = coeff[0] * tmp1 + coeff[1] * tmp2
        for i in range(2, deg + 1):
            tmp3 = iden2 * tmp2 - tmp1
            tmp1, tmp2 = tmp2, tmp3
            rtn += coeff[i] * tmp3
        return rtn
    else:
        raise ValueError("Unknown polynomial type")

def find_extreme(func: Callable[[Decimal], Decimal], coeff: List[Decimal], deg: int,
                 start: Decimal, end: Decimal, prec: int, scan: Decimal, type: int,
                 scale: Decimal, is_opt_sampling: bool) -> (List[Point], Decimal):

    ext = []
    origin_sc = scan
    s = 15 if is_opt_sampling else 0
    sc = scan / Decimal(10 ** s) if is_opt_sampling else scan

    def err(x): return eval_poly(deg, coeff, x, type, scale) - func(x)

    # Start boundary
    y = err(start)
    ext.append(Point(start, y))
    ext[-1].locmm = 1 if y > 0 else -1

    scan_1 = start
    scan_2 = start + sc
    scan_y1 = err(scan_1)
    scan_y2 = err(scan_2)
    inc_2 = 1 if scan_y1 < scan_y2 else -1

    while scan_2 + sc < end:
        if is_opt_sampling:
            for i in range(s):
                check_range = Decimal(10) * origin_sc / Decimal(10**i)
                if start + check_range < scan_2 < end - check_range:
                    sc = origin_sc / Decimal(10**i)
                    break
            else:
                sc = origin_sc / Decimal(10**(s + 1))
        else:
            sc = origin_sc

        scan_prev = scan_1
        scan_1 = scan_2
        scan_2 = scan_1 + sc

        scan_y1 = scan_y2
        scan_y2 = err(scan_2)
        inc_1, inc_2 = inc_2, 1 if scan_y1 < scan_y2 else -1

        if inc_1 == 1 and inc_2 != 1 or inc_1 == -1 and inc_2 != -1:
            # Binary search
            search_start, search_end = scan_prev, scan_2
            search_sc = (search_end - search_start) / 4
            for _ in range(prec):
                intervals = [search_start + i * search_sc for i in range(5)]
                slopes = [(err(intervals[i]) < err(intervals[i + 1])) for i in range(4)]
                if slopes[0] and not slopes[1]:
                    search_end -= 2 * search_sc
                elif slopes[1] and not slopes[2]:
                    search_start += search_sc
                    search_end -= search_sc
                elif slopes[2] and not slopes[3]:
                    search_start += 2 * search_sc
                search_sc /= 2

            x_ext = (search_start + search_end) / 2
            y_ext = err(x_ext)
            pt = Point(x_ext, y_ext)
            pt.locmm = 1 if inc_1 == 1 else -1
            ext.append(pt)

    # End boundary
    y = err(end)
    pt = Point(end, y)
    pt.locmm = 1 if y > 0 else -1
    ext.append(pt)

    maxerr = max(abs(p.y) for p in ext)
    return ext, maxerr
