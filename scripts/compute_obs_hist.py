import json
import matplotlib.pyplot as plt
import numpy as np
import math
from tools import utils as ut

intensities = ['0', '1', '2', '3']
observables_names = ['d', 'v_G', 'v_diff']

file_name = 'data/observables/obs.json'

pdfs = {}

with open(file_name) as json_data:
    observables = json.load(json_data)

for o in observables_names:
    pdfs[o] = {}
    for i in intensities:
        no_nan = [v for v in observables[i][o] if not math.isnan(v)]
        hist = ut.compute_histogram(o, observables[i][o])
        pdfs[o][i] = ut.compute_pdf(o, hist)
        
ut.plot_pdf(pdfs, intensities)
ut.save_pdf(pdfs, intensities)

