#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 14:11:04 2018

@author: zeynep
"""
import numpy.ma as MA 
import numpy as np
import social_relation as social
from constants import CAM_NOS, GEST_NAMES

def find_between( s, first, last ):
    """
    Find the part of string s, which is between first and last
    firstfirstHERElast returns firstHERE
    If you want to go deeper use the other function find_between_r 
    (from tanakan's)
    """
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""
    
def readHeader(fname): 
    # Open the file and read the relevant lines 
    f = open(fname) 
    head = f.readlines()[:13] 
    f.close() 
        
    return (head) 


def readData(fname): 
      
    # Open file and read column names and data block 
    f = open(fname) 
    # Ignore header 
    for i in range(13):  
        f.readline() 
    # read the important part
    data_block = f.readlines() 
    f.close() 

    groups = []
        
    for line in data_block: 
        # items = re.split(r'\t+',  line) # to split multi tabs, not good for me
        items = line.split("\t")
      
        g = social.group(items[0].split('_'), int(items[-1].strip()))
        
        for c, cam_no in enumerate(CAM_NOS):
            
            items_cam = items[c*4+1:(c+1)*4+1]
            isincam = any(items_cam)
            
            if isincam:
                for i, val in enumerate(items_cam):
                    g.set_gesture_gt(GEST_NAMES[i], cam_no, val )
                
            g.load_members()
            
        groups.append(g)
        # groups[0].cams[0].isincam
        # groups[0].cams[0].gestures['speak']
    return groups


    