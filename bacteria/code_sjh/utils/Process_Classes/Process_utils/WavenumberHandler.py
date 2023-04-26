import numpy
from scipy import interpolate
# TODO：实现通过插值的波数平移
def wavenumber_translator_interpolate(new_x, dx = 0):
    """

    @param dx: 保持X不变，将窗口移动dx（正右，负左）
    @return: 重新插值函数
    """
    res_x = new_x
    new_x = res_x-dx
    def func(y,x):
        f = interpolate.interp1d(x, y, kind = "cubic")
        new_y = f(new_x)
        return new_y, new_x

    return func

if __name__ == '__main__':
    wavenumber_translator_interpolate()
