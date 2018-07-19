import json
import matplotlib.pyplot as plt
import numpy as np
import math

# compute the histograms from the angles computed with compute_angles.py

angle_names = ['RShoulder', 'RElbow',  'LShoulder', 'LElbow']

for camera_id in [1, 2, 4, 5]:
 for angle in angle_names:
    pdfs = {}
    for interaction in ['0', '1']:
        file_name = 'data/angles/{}_{}_{}.json'.format(camera_id, angle, interaction)
        with open(file_name) as json_data:
            pdfs[interaction] = np.array(json.load(json_data))

    max_bound = max(max(pdfs['0']), max(pdfs['1']))
    min_bound = min(min(pdfs['0']), min(pdfs['1']))
    bin_size = 5
    n_bins = math.ceil((max_bound - min_bound) / bin_size) + 1
    edges = np.linspace(min_bound, max_bound, n_bins)


    for interaction in ['0', '1']:
        hist = np.histogram(pdfs[interaction], edges)[0]
        pdfs[interaction] = hist / sum(hist) / bin_size

    edges = np.arange(min_bound, max_bound, bin_size)
    plt.title('{} PDFs for camera {}'.format(angle, camera_id))
    plt.plot(edges, pdfs['0'], label='no gesture')
    plt.plot(edges, pdfs['1'], label='gestures')
    plt.xlabel('{}'.format(angle))
    plt.ylabel('p({})'.format(angle))
    plt.legend()
    plt.show()
