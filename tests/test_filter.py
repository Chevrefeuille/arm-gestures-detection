from tools import utils as ut

signal = [2, 80, 6, 3]

filtered_signal = ut.median_filter(signal, 3)

print(filtered_signal)