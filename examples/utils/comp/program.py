from math import log
from decimal import Decimal, getcontext
from typing import List
from PolyUpdate import Tree, EvalType
from MinicompFunc import pow2, ceil_to_int

getcontext().prec = 100

def upgrade_oddbaby(n: int, tree: Tree):
    assert n % 2 == 1, "n must be odd"
    d = ceil_to_int(log(n) / log(2))
    total_min = 10000
    min_m, min_l = 0, 0
    total_min_tree = Tree()

    for l in range(1, n + 1):
        if pow2(l) - 1 > n:
            break
        for m in range(1, n + 1):
            if pow2(m - 1) >= n:
                break

            f = [[0 if i == 1 else 10000 for _ in range(d + 1)] for i in range(n + 1)]
            G = [[Tree(EvalType.ODDBABY) for _ in range(d + 1)] for _ in range(n + 1)]
            f[1][1] = 0

            for j in range(2, d + 1):
                for i in range(1, n + 1, 2):
                    if i <= pow2(l) - 1 and i <= pow2(j - 1):
                        f[i][j] = 0
                    else:
                        min_val = 10000
                        min_tree = Tree(EvalType.ODDBABY)
                        for k in range(1, m):
                            g = pow2(k)
                            if g >= i or k >= j:
                                break
                            temp_val = f[i - g][j - 1] + f[g - 1][j] + 1
                            if temp_val < min_val:
                                min_val = temp_val
                                min_tree = Tree()
                                min_tree.merge(G[g - 1][j], G[i - g][j - 1], g)
                        f[i][j] = min_val
                        G[i][j] = min_tree

            cost = f[n][d] + pow2(l - 1) + m - 2
            if cost < total_min:
                total_min = cost
                total_min_tree = G[n][d]
                min_m = m
                min_l = l

    tree.copy(total_min_tree)
    tree.m = min_m
    tree.l = min_l


def upgrade_baby(n: int, tree: Tree):
    d = ceil_to_int(log(n + 1) / log(2))
    total_min = 10000
    min_m, min_b = 0, 0
    type_ = EvalType.BABY
    total_min_tree = Tree(type_)

    if n == 1:
        tree.copy(Tree(type_))
        tree.m = 1
        tree.b = 1
        return

    for b in range(1, n + 1):
        for m in range(1, n + 1):
            if pow2(m - 1) * b > n:
                break

            f = [[0 for _ in range(d + 1)] for _ in range(n + 1)]
            G = [[Tree(type_) for _ in range(d + 1)] for _ in range(n + 1)]

            for j in range(1, d + 1):
                for i in range(1, n + 1):
                    if i + 1 > pow2(j):
                        f[i][j] = 10000
                        G[i][j] = Tree(type_)
                    elif b == 1 and m >= 2 and i <= 2 and i <= pow2(j - 1):
                        f[i][j] = 0
                        G[i][j] = Tree(type_)
                    elif i <= b and i <= pow2(j - 1):
                        f[i][j] = 0
                        G[i][j] = Tree(type_)
                    else:
                        min_val = 10000
                        min_tree = Tree(type_)
                        for k in range(2, b + 1):
                            g = k
                            if g > pow2(j - 1) or g >= i:
                                continue
                            temp_val = f[i - g][j - 1] + f[g - 1][j] + 1
                            if temp_val < min_val:
                                min_val = temp_val
                                min_tree = Tree()
                                min_tree.merge(G[g - 1][j], G[i - g][j - 1], g)
                        for k in range(m):
                            g = pow2(k) * b
                            if g > pow2(j - 1) or g >= i:
                                continue
                            temp_val = f[i - g][j - 1] + f[g - 1][j] + 1
                            if temp_val < min_val:
                                min_val = temp_val
                                min_tree = Tree()
                                min_tree.merge(G[g - 1][j], G[i - g][j - 1], g)
                        f[i][j] = min_val
                        G[i][j] = min_tree

            cost = f[n][d] + m + b - 2
            if cost < total_min:
                total_min = cost
                total_min_tree = G[n][d]
                min_m = m
                min_b = b

    tree.copy(total_min_tree)
    tree.m = min_m
    tree.b = min_b
