import json
from tools import utils as ut
import matplotlib.pyplot as plt
import pickle
import sensei.recognition_tools as tools
from sensei.constants import *
import numpy as np


# extract amdf for figures

# no-gest
# file_name = 'data/angles/samples/101-1-RElbow'
# data_name = 'data/gnuplot/no_gest_amdf.txt'
# # strong gest
# file_name = 'data/angles/samples/277-1-LElbow'
# data_name = 'data/gnuplot/strong_gest_amdf.txt'
# # weak gest
file_name = 'data/angles/samples/499-2-LElbow'
data_name = 'data/gnuplot/weak_gest_amdf.txt'


i_min = int(0 * FPS)
i_max = i_min + int(6 * FPS)

angles = np.array(pickle.load(open(file_name, 'rb')))#[i_min:i_max]
x = tools.medfilt(angles, MEDFILT_SIZE)

t = np.array([i * 1/FPS for i in range(len(x))])


amd = tools.amdf(x)
t_amd = np.array([i * 1/FPS for i in range(len(amd))])
res = tools.fit_sin(t_amd, amd)

if res:
    np.savetxt(data_name, np.c_[t_amd, amd, res['fitfunc'](t_amd)]) 
else:
    np.savetxt(data_name, np.c_[t_amd, amd])