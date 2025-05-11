
from enum import Enum

class EvalType(Enum):
    NONE = 0
    ODDBABY = 1
    BABY = 2

from decimal import Decimal, getcontext
import math

getcontext().prec = 100

class Tree:
    def __init__(self, ty=EvalType.NONE, a=None, b=None, g=None):
        if a is None and b is None:
            self.depth = 0
            self.type = ty
            self.tree = [-1, 0]
            self.m = 0
            self.l = 0
            self.b = 0
        else:
            if a.type != b.type:
                raise ValueError("the types of two trees are not the same")
            self.type = a.type
            self.depth = max(a.depth, b.depth) + 1
            size = 2 ** (self.depth + 1)
            self.tree = [-1] * size
            self.tree[1] = g

            for i in range(1, 2 ** (a.depth + 1)):
                temp = 2 ** int(math.log(i, 2))
                self.tree[i + temp] = a.tree[i]

            for i in range(1, 2 ** (b.depth + 1)):
                temp = 2 ** int(math.log(i, 2))
                self.tree[i + 2 * temp] = b.tree[i]

    def clear(self):
        self.depth = 0
        self.type = EvalType.NONE
        self.tree = [-1, 0]

    def print(self):
        print("depth of tree:", self.depth)
        for i in range(self.depth + 1):
            print(" ".join(str(self.tree[j]) for j in range(2 ** i, 2 ** (i + 1))))
        if self.type == EvalType.ODDBABY:
            print(f"m: {self.m}, l: {self.l}")
            nonscalar = self.m - 1 + 2 ** (self.l - 1) - 1 + sum(1 for x in self.tree if x > 0)
            print(f"nonscalar: {nonscalar}")
        elif self.type == EvalType.BABY:
            print(f"m: {self.m}, b: {self.b}")
            nonscalar = self.m + self.b - 2 + sum(1 for x in self.tree if x > 0)
            print(f"nonscalar: {nonscalar}")

    def merge(self, a, b, g):
        self.clear()
        if a.type != b.type:
            raise ValueError("the types of two trees are not the same")
        self.type = a.type
        self.depth = max(a.depth, b.depth) + 1
        self.tree = [-1] * (2 ** (self.depth + 1))
        self.tree[1] = g

        for i in range(1, 2 ** (a.depth + 1)):
            temp = 2 ** int(math.log(i, 2))
            self.tree[i + temp] = a.tree[i]
        for i in range(1, 2 ** (b.depth + 1)):
            temp = 2 ** int(math.log(i, 2))
            self.tree[i + 2 * temp] = b.tree[i]

    def copy(self, input):
        import copy
        self.depth = input.depth
        self.type = input.type
        self.tree = copy.deepcopy(input.tree[:])
        self.m = input.m
        self.l = input.l
        self.b = input.b

class Polynomial:
    def __init__(self, deg=-1, coeff=None, tag=None):
        self.deg = deg
        self.coeff = []
        self.chebcoeff = []
        if coeff is not None and tag is not None:
            if tag == "power":
                self.coeff = coeff[:]
                self.chebcoeff = [Decimal(0)] * (deg + 1)
            elif tag == "cheb":
                self.chebcoeff = coeff[:]
                self.coeff = [Decimal(0)] * (deg + 1)
        elif deg >= 0:
            self.coeff = [Decimal(0)] * (deg + 1)
            self.chebcoeff = [Decimal(0)] * (deg + 1)

    def get_coeff(self):
        return self.coeff[:]

    def copy(self, poly):
        self.deg = poly.deg
        self.coeff = poly.coeff[:]
        self.chebcoeff = poly.chebcoeff[:]

    def cheb_to_power(self):
        tmp = Polynomial(self.deg)
        for i in range(self.deg + 1):
            chebbasis = Polynomial()
            chebyshev(chebbasis, i)
            for j in range(i + 1):
                chebbasis.coeff[j] *= self.chebcoeff[i]
            addinplace(tmp, chebbasis)
        self.coeff = tmp.coeff[:]

    def evaluate(self, input_val: Decimal) -> Decimal:
        result = Decimal(0)
        power = Decimal(1)
        for i in range(self.deg + 1):
            result += self.coeff[i] * power
            power *= input_val
        return result

    def evaluate_cheb(self, input_val: Decimal) -> Decimal:
        if self.deg == 0:
            return Decimal(1)
        tmp1 = Decimal(1)
        tmp2 = input_val
        result = self.chebcoeff[0] * tmp1 + self.chebcoeff[1] * tmp2
        for i in range(2, self.deg + 1):
            tmp3 = 2 * input_val * tmp2 - tmp1
            tmp1, tmp2 = tmp2, tmp3
            result += self.chebcoeff[i] * tmp3
        return result


def mul(rtn: Polynomial, a: Polynomial, b: Polynomial):
    rtn.deg = a.deg + b.deg
    rtn.coeff = [Decimal(0)] * (rtn.deg + 1)
    rtn.chebcoeff = [Decimal(0)] * (rtn.deg + 1)
    for i in range(rtn.deg + 1):
        for j in range(i + 1):
            if j <= a.deg and (i - j) <= b.deg:
                rtn.coeff[i] += a.coeff[j] * b.coeff[i - j]

def add(rtn: Polynomial, a: Polynomial, b: Polynomial):
    if a.deg >= b.deg:
        rtn.copy(Polynomial(a.deg, a.coeff, "power"))
        for i in range(b.deg + 1):
            rtn.coeff[i] += b.coeff[i]
    else:
        rtn.copy(Polynomial(b.deg, b.coeff, "power"))
        for i in range(a.deg + 1):
            rtn.coeff[i] += a.coeff[i]

def subt(rtn: Polynomial, a: Polynomial, b: Polynomial):
    if a.deg >= b.deg:
        rtn.copy(Polynomial(a.deg, a.coeff, "power"))
        for i in range(b.deg + 1):
            rtn.coeff[i] -= b.coeff[i]
    else:
        rtn.copy(Polynomial(b.deg, b.coeff, "power"))
        for i in range(b.deg + 1):
            rtn.coeff[i] *= -1
        for i in range(a.deg + 1):
            rtn.coeff[i] += a.coeff[i]

def addinplace(a: Polynomial, b: Polynomial):
    rtn = Polynomial()
    add(rtn, a, b)
    a.copy(rtn)

def chebyshev(rtn: Polynomial, deg: int):
    if deg == 0:
        rtn.copy(Polynomial(0))
        rtn.coeff[0] = Decimal(1)
    elif deg == 1:
        rtn.copy(Polynomial(1))
        rtn.coeff[0] = Decimal(0)
        rtn.coeff[1] = Decimal(1)
    else:
        iden2 = Polynomial(1)
        iden2.coeff[0] = Decimal(0)
        iden2.coeff[1] = Decimal(2)
        tmp1 = Polynomial(0)
        tmp1.coeff[0] = Decimal(1)
        tmp2 = Polynomial(1)
        tmp2.coeff[0] = Decimal(0)
        tmp2.coeff[1] = Decimal(1)
        for i in range(2, deg + 1):
            tmp3 = Polynomial()
            mul(tmp3, iden2, tmp2)
            subt(rtn, tmp3, tmp1)
            tmp1.copy(tmp2)
            tmp2.copy(rtn)
