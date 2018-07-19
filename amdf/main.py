#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 15:58:58 2018

@author: zeynep
"""

import file_tools as file
import distributions
import consistency
from constants import CAM_NOS


if __name__ == "__main__":

     fname1 = 'annotations_hoang_v4.txt'
     fname2 = 'annotations_koyama_v1.txt'
     
     groups1 = file.readData(fname1)
     groups2 = file.readData(fname2)

     distributions.summary(groups2)
     
     consistency.gesture_consistency(groups1, groups2)
     
     
     for g in groups1: 
         g.load_members()
         for m in g.members:         
             m.estimate_gesture()
             g.set_gesture_estim()
        
     consistency.get_estim_perf(groups1)
      

     for g in groups2: 
         g.load_members()
         for m in g.members:         
             m.estimate_gesture()
             g.set_gesture_estim()
        
     consistency.get_estim_perf(groups2)                          
    # groups[0].cams[0].isincam
    # groups[0].cams[0].gesture_gt['speak']