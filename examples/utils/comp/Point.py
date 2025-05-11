from decimal import Decimal, getcontext

# 设置足够高的精度以模拟 NTL::RR 的大数行为
getcontext().prec = 100  # 你可以根据需要调整这个精度

class Point:
    def __init__(self, x=None, y=None):
        # 默认为 None，如果给定则转换为 Decimal
        self.x = Decimal(x) if x is not None else None
        self.y = Decimal(y) if y is not None else None
        self.locmm = None  # 对应 C++ 中的 long 类型成员

