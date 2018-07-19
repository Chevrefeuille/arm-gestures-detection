#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 16:18:57 2018

@author: zeynep
"""

FPS = 60
DATADIR = '../data/angles/samples/'

GEST_NAMES = ['speak', 'touch', 'arms', 'gaze']
CAM_NOS = ['1', '2', '4', '5']
#LIMBS = ['LElbow', 'RElbow', 'LShoulder', 'RShoulder']
LIMBS = ['LElbow', 'RElbow']  # For now, I do the detection using only elbows

MAX_MISSING_SAMPLE_RATE = 0.20
MIN_OBSERVATION_DUR = 4
MEDFILT = 'on'
MEDFILT_SIZE = 9 # put a limit on number of saples before or after median filtering?
