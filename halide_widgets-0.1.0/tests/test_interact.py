import halide as hl
from halide_widgets import Slider, interact

def test_interact():
    x = hl.Var()
    y = hl.Var()
    f = hl.Func()
    p = hl.Param(hl.Float(32), 0.0)
    f = hl.Func()
    f[x, y] = x + p * y
    interact(f, size=[100, 100], controls=[Slider(p, 0, 1, 0.001)])
