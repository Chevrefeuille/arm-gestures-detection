#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 15:13:34 2018

@author: zeynep
"""
from constants import *
import pickle
import numpy as np
import recognition_tools as tools
import os.path
import matplotlib.pyplot as plt


class limb:
    
    def __init__(self, limb_name):
    
        self.limb_name = limb_name
        self.angles = {}
        
        self.gesture_status = {}
        for cam_no in CAM_NOS:   
            self.gesture_status[ cam_no ] = -1
        
    def load_limb(self, id):
        # print(id)
        for cam_no in CAM_NOS:  
            fname = DATADIR + str(id) + '-' + cam_no + '-' + self.limb_name
            if os.path.isfile(fname) :
                self.angles[ cam_no ] = np.array(pickle.load(open(fname, 'rb')))
                #print('angle file {} loaded'.format(fname))
            else:
                self.angles[ cam_no ] = []

        
    def get_gesture_status(self):
        """
        We have one gt of gesture for the entire group
        so we know whether there is a gesture in the group or not but we do not 
        know who did it
        """
        n_inconc = 0
        for cam_no in CAM_NOS: 
            # we permit misssing samples at not more than 25% and 
            # we require an estimation of at least 2 secs (not necessarily continuous)
            if (np.sum((np.isnan(self.angles[ cam_no ]))) <  MAX_MISSING_SAMPLE_RATE * len(self.angles[ cam_no ])):
                if (np.sum(np.invert(np.isnan(self.angles[ cam_no ]))) > MIN_OBSERVATION_DUR * FPS):
        
                    #print('Estimating limb gesture\t',end='', flush=True)
                    angles = self.angles[cam_no]
                    x = angles - np.mean(angles[np.where(np.invert(np.isnan(angles)))])
                    if (MEDFILT == 'on'):
                        x = tools.medfilt(x, MEDFILT_SIZE)

                    
                    amd = tools.amdf(x)
                    t_amd = np.array([i * 1/FPS for i in range(len(amd))])
                    res = tools.fit_sin(t_amd, amd)
                    #print(res)
                    
                    if res:
                        # print(cam_no)
                        # plt.plot(t_amd, amd, label='amdf')
                        # plt.plot(t_amd, res['fitfunc'](t_amd), '--', label='fitted sinusoid')
                        # plt.legend()
                        # plt.title('AMDF and fitted sinusoid')
                        # plt.xlabel(r'$\tau$')
                        # plt.ylabel('AMDF')
                        # plt.show()
                        rmse_amd = np.sqrt(np.mean((amd - res['fitfunc'](t_amd))**2))
                        #print('Cam_no:\t{}\tLimb:\t{}\tNo Gesture \trmse_amd: {}' .format(cam_no, l, rmse_amd))
                        self.gesture_status[cam_no] = 0
                        #print('No gesture')
                    else:
                        #print('Cam_no:\t{}\tLimb:\t{}\tGesture***'.format(cam_no, l))
                        # print(cam_no)
                        # plt.plot(t_amd, amd, label='amdf')
                        # plt.legend()
                        # plt.title('AMDF and fitted sinusoid')
                        # plt.xlabel(r'$\tau$')
                        # plt.ylabel('AMDF')
                        # plt.show()
                        self.gesture_status[cam_no] = 1
                        #print('Gesture')
                else:
                    #print('Too short samples to estimate the limb gesture')
                    #print('Inconclusive')
                    self.gesture_status[cam_no] = -1

            else:
                #print('Cam_no:\t{}\tLimb:\t{}\tInconclusive'.format(cam_no, l))
                #print('Not enough not-nan samples to estimte the limb gesture')
                #print('Inconclusive')
                n_inconc += 1
                self.gesture_status[cam_no] = -1
        print("Inconclusive: " + str(n_inconc))

    
class ped:
    
    def __init__(self, id):
        
        self.id = int(id) # maybe it is int from the begining? 
        
        self.limbs = {}
        for limb_name in LIMBS:
            if not limb_name in self.limbs:
                self.limbs[limb_name] = limb(limb_name)
 
        # 1: gesture
        # 0: no gesture
        # -1: inconclusive
        #self.gesture_status = -1
        
    def load_ped(self):
        """
        Loads the limbs
        """
        for limb_name in LIMBS:
            self.limbs[limb_name].load_limb(self.id)
                
    def estimate_gesture(self):
        for limb_name in LIMBS:
            self.limbs[limb_name].get_gesture_status()
            # if any of the limbs is positive set positive
#            
#    def set_gesture_status(self):
#        for limb_name in LIMBS:
#            for cam_no in CAM_NOS:
#                if self.limbs[limb_name][cam_no] == 1:
#                    self.gesture_status = 1
#                    break
#                elif self.limbs[limb_name][cam_no] == 0:
#                    self.gesture_status = 0
#            # if any of the limbs is positive set positive
        


class group:
    
    def __init__(self, ids=[],intensity=0):
        # individual peds have ids 
        # also the group has a list of ids 
        # i need to simplify this later
        #self.ids = list(map(int, ids)) # list of strings to list of integers
        #self.cams = []
        self.members = []
        for id in ids:
            self.members.append(ped(id))
        
        self.gesture_estim = {}
        for gest_name in GEST_NAMES: 
            self.gesture_estim[gest_name] = {}
            for cam_no in CAM_NOS:
                self.gesture_estim[gest_name][cam_no] = -1 # inconclusive
                    
        self.gesture_gt = {}
        for gest_name in GEST_NAMES:  
            self.gesture_gt[gest_name] = {}
            for cam_no in CAM_NOS:
                self.gesture_gt[gest_name][cam_no] = -1 # as not in cam 
                
        self.intensity = intensity


      
    def set_gesture_gt(self, gest_name, cam_no, val):
        if val == '1' or val == '0':
            # looks srupid but sometimes it is empty, blank or sth
            self.gesture_gt[gest_name][cam_no] = int(val)
        
        
            
    def load_members(self):      
        for m in self.members:              
            m.load_ped()
                    
    def set_gesture_estim(self):
        """
        After I get the gesture status of each lib of each member in each camera,
        I set a gesture status for the entire group for that camera
        
        Basically if any limb (LElbow or RElbow) of any member is detected to make 
        a gesture in a particular camera view, the gesture status of the group for 
        that camera is set to positive, otherwise it is left negative
        
        """
        gest = 'arms'
        
        for cam_no in CAM_NOS:
            for member in self.members:
                for limb_name in LIMBS:
                    
                    # override inconclusive or no-gesture
                    if self.gesture_estim[gest][ cam_no ] is not 1:
                        
                        # override inconclusive or no-gesture
                        if (self.gesture_estim[gest][ cam_no ] is -1 or \
                        self.gesture_estim[gest][ cam_no ] is 0 ) and\
                        member.limbs[limb_name].gesture_status[cam_no] == 1:
                            
                            self.gesture_estim[gest][ cam_no ] = 1
                            #print('Gesture xxx')
         
                        # override inconclusive
                        elif self.gesture_estim[gest][ cam_no ] is -1 and\
                        member.limbs[limb_name].gesture_status[cam_no] == 0:
                            #print('No gesture xxx')
                            self.gesture_estim[gest][ cam_no ] = 0


    
