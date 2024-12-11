from bacteria.code_sjh.utils.Process_utils.Core import *


class norm_func(ProcessorFunction):
    # 最大最小值归一
    def __init__(self, a = 0, b = 1, ):
        super(norm_func, self).__init__("norm_func()" if (a == 0 and b == 1) else "norm_func({},{})".format(a, b))
        self.a = a
        self.b = b

    def __call__(self, y, x = None):
        y = ((self.b - self.a) * (y - min(y))) / (max(y) - min(y)) + self.a
        if x is None:
            return y
        else:
            return y, x


class area_norm_func(ProcessorFunction):
    # 曲线下面积归一
    def __init__(self, a = 1, ):
        assert a > 0, "a must be greater than 0"
        super(area_norm_func, self).__init__("area_norm_func()" if a == 1 else "area_norm_func({})".format(a))
        self.a = a

    def __call__(self, y, x = None):
        y -= y.min()
        res = self.a * y / sum(y) * len(y)
        return res if x is None else (res, x)