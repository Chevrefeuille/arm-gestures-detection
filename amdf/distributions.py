#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 16:14:10 2018

@author: zeynep
"""


import numpy as np
import matplotlib.pyplot as plt
from constants import CAM_NOS, GEST_NAMES


def get_ngroups_by_cam(groups):
    
    ngroups_by_cam = {}
    for cam_no in CAM_NOS:
        ngroups_by_cam[ cam_no ] = 0
    
    gest_name = GEST_NAMES[0] # i check only for first gesture since all others are 1 or 0 too
    for group in groups:    
        for cam_no in CAM_NOS:        
            # -1 means group is not in cam
            if (group.gesture_gt[gest_name][ cam_no ] is not -1) : 
                ngroups_by_cam[ cam_no ] += 1
                
    print('-------------------------------')                    
    print('Ngroups by camera')    
    for cam_no in CAM_NOS:
        print(cam_no, ': ', ngroups_by_cam[ cam_no ])
    # groups[0].cams[0].isincam
    # groups[0].cams[0].gesture_gt['speak']
    
def get_ngestures(groups):
    
    ngestures = {}
    for gest_name in GEST_NAMES:
        ngestures[gest_name] = 0

    for group in groups:    
        for gest_name in GEST_NAMES:
            for cam_no in CAM_NOS:
                    if group.gesture_gt[gest_name][ cam_no ] == 1:
                        ngestures[gest_name] += 1
                        
    print('-------------------------------')    
    print('Ngestures')    
    for gest_name in GEST_NAMES:
        print(gest_name, ': ', ngestures[gest_name])
        
def get_ngest_by_cam(groups):

    ngest_by_cam = {}
    for gest_name in GEST_NAMES:
        ngest_by_cam[gest_name] = {}
        for cam_no in CAM_NOS:
            ngest_by_cam[gest_name][ cam_no ] = 0

        
    for group in groups:    
        for gest_name in GEST_NAMES:
            for cam_no in CAM_NOS:
                    if group.gesture_gt[gest_name][ cam_no ] == 1:
                        ngest_by_cam[gest_name][ cam_no ] += 1
    print('-------------------------------')    
    print('Ngestures by camera')  
    for cam_no in CAM_NOS:
        print('\t', cam_no, end='', flush=True)
    print('')
    
    
    for gest_name in GEST_NAMES:
        print(gest_name, '\t', end='', flush=True)
        for cam_no in CAM_NOS:
            print(ngest_by_cam[gest_name][ cam_no ], '\t\t', end='', flush=True)
        print('')
        
def get_ngest_concur(groups):

    ngest_concur = {}
    for gest_name1 in GEST_NAMES:
        ngest_concur[gest_name1] = {}
        for gest_name2 in GEST_NAMES:
            ngest_concur[gest_name1][gest_name2] = 0

        
    for group in groups:    
        for gest1 in GEST_NAMES:
             for gest2 in GEST_NAMES:
                 for cam_no in CAM_NOS:
                        if gest1 != gest2 and \
                        group.gesture_gt[gest1][ cam_no ] == 1 and \
                        group.gesture_gt[gest2][ cam_no ] == 1:
                            ngest_concur[gest1][gest2] += 1
    print('-------------------------------')    
    print('Ngestures concurring')  
    for gest_name in GEST_NAMES:
        print('\t', gest_name, end='', flush=True)
    print('')
    
    for gest1 in GEST_NAMES:
        print(gest1, '\t', end='', flush=True)
        for gest2 in GEST_NAMES:
            print(ngest_concur[gest1][gest2], '\t\t', end='', flush=True)
        print('')
        
def get_ngest_isolated(groups):

    ngest_isolated = {}
    for gest_name in GEST_NAMES:
        ngest_isolated[gest_name] = 0
        
    for group in groups:    
        for gest1 in GEST_NAMES:
            for cam_no in CAM_NOS:
                isisolated = True
                for gest2 in GEST_NAMES:
                        if gest1 != gest2 :
                            if group.gesture_gt[gest1][ cam_no ] == 1 and \
                            group.gesture_gt[gest2][ cam_no ] == 0:
                                isisolated = isisolated and True
                            else:
                                isisolated = isisolated and False
                # one entry for each cam, not for each group
                if isisolated:
                    ngest_isolated[gest1] += 1
                    
                                
    print('-------------------------------')    
    print('Ngestures isolated')  
    
    for gest_name in GEST_NAMES:
        print('{}\t{}'.format(gest_name, ngest_isolated[gest_name]))
        
        
def summary( groups):
         get_ngestures(groups)
         get_ngroups_by_cam(groups)
         get_ngest_by_cam(groups) 
         get_ngest_concur(groups)
         get_ngest_isolated(groups)
        
