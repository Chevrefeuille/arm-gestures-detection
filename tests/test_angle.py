from tools import utils as ut
import math

A = (0, 0)
B = (1, 0)
C = (0, 1)

angle = ut.compute_angle(A, B, C)
print(angle, math.pi / 4)