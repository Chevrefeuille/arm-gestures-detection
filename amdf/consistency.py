#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 11:02:34 2018

@author: zeynep
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from constants import CAM_NOS, GEST_NAMES, LIMBS


def kriAlpha(labels1,labels2,scale):
    
    """
    alpha = kriAlpha(data,scale)
    calculates Krippendorff's Alpha as a measure of inter-rater agreement
    
    data: rate matrix, each row is a rater or coder, each column is a case
    scale: level of measurement, supported are 'nominal', 'ordinal', 'interval'
    missing values have to be coded as #nan or inf
    
    For details about Krippendorff's Alpha see:
    Hayes, Andrew F. & Krippendorff, Klaus (2007). Answering the call for a
    standard reliability measure for coding data. Communication Methods and
    Measures, 1, 77-89
        
    Results for the two examples below have been verified:
        
    data = \
    [#nan   #nan   #nan   #nan   #nan     3     4     1     2     1     1     3     3   #nan     3; ...
    1   #nan     2     1     3     3     4     3   #nan   #nan   #nan   #nan   #nan   #nan   #nan; ...
    #nan   #nan     2     1     3     4     4   #nan     2     1     1     3     3   #nan     4];
        
    alpha nominal: 0.6914, ordinal: 0.8067, interval: 0.8108
        
    data = \
   [[1.1000,    2.1000,    5.0000,    1.1000,    2.0000], 
   [ 2.0000,    3.1000,    4.0000,    1.9000,    2.3000], 
   [1.5000,    2.9000,    4.5000,    4.4000,    2.1000], 
   [ math.nan,    2.6000 ,   4.3000,    1.1000,    2.3000]]
    
    alpha nominal: 0.0364, ordinal: 0.5482, interval: 0.5905
    
    """
    temp1 = np.array([labels1])
    temp2 = np.array([labels2])
    data =  np.concatenate((temp1, temp2), 0)
    # get only those columns with 
    
    data = np.asanyarray(data)

    allVals  =  np.unique(data)
    allVals  =  allVals[abs(allVals) < math.inf]

    # coincidence matrix
    coinMatr =  np.ones((len(allVals), len(allVals))) * float('nan')
    for r in range(0, len(allVals)):
        for c in range(r, len(allVals)):
            val = 0
            for d in range(0, len(data[0])):
                # find number of pairs
                thisEx = data[:,d]
                
                thisEx = thisEx[abs(thisEx) < math.inf]
                numEntr = len(thisEx)
                numP = 0
                for p1 in range(0, numEntr):
                    for p2 in range(0, numEntr):
                        if p1 == p2:
                            continue                        
                        if thisEx[p1] == allVals[r] and thisEx[p2] == allVals[c]:
                            numP = numP+1
                if numP:
                    val = val+numP/(numEntr-1)

            coinMatr[r,c] = val
            coinMatr[c,r] = val

    nc = np.sum(coinMatr,axis=1)
    n = np.sum(nc)

    # expected agreement
    expMatr = np.ones((len(allVals),len(allVals))) * float('nan')
    for i in range(0, len(allVals)):
        for j in range(0, len(allVals)):
            if i == j:
                val = nc[i]*(nc[j]-1)/(n-1);
            else:
                val = nc[i]*nc[j]/(n-1);

            expMatr[i,j] = val;


    # difference matrix
    diffMatr = np.zeros((len(allVals),len(allVals)))
    for i in range(0, len(allVals)):
        for j in range(i+1, len(allVals)):
            if i != j:
                if scale is 'nominal':
                    val = 1
                elif scale is 'ordinal':
                    val = np.sum(nc[i:j+1])-nc[i]/2-nc[j]/2
                    val = np.square(val)
                elif scale is 'interval':
                    val = (allVals[j]-allVals[i]) ** 2
                else:
                    print('unknown scale: ', scale)
            else:
                val = 0
            
            diffMatr[i,j] = val
            diffMatr[j,i] = val


    # observed - expected agreement
    mydo = 0
    de = 0
    for c  in range (0, len(allVals)):
        for k in range(c+1, len(allVals)):
            if scale is 'nominal':
                mydo = mydo+coinMatr[c,k]
                de = de+nc[c]*nc[k]
            elif scale is 'ordinal':
                mydo = mydo+coinMatr[c,k]*diffMatr[c,k]
                de = de+nc[c]*nc[k]*diffMatr[c,k]
            elif scale is 'interval':
                mydo = mydo+coinMatr[c,k]*(allVals[c]-allVals[k]) ** 2
                de = de+nc[c]*nc[k]*(allVals[c]-allVals[k]) ** 2
            else:
                print('unknown scale: ', scale)

    de = 1/(n-1)*de
    alpha = 1-mydo/de
    
    return alpha

def fleissKappa(labels1, labels2):
    """
    Just for trying this, 
    consider 4 people do binary labeling task for 10 samples:
        
    sample    tom brooks chris steve
    1           1      1     1     0
    2           0      0     0     0
    3           1      0     1     0
    4           0      0     0     0
    5           1      1     1     1
    6           0      1     1     1
    7           1      1     1     1
    8           1      1     1     1
    9           0      0     0     0
    10          1      1     0     0
    
    data =[[1, 1, 1, 0], [0,0,0,0,],[1,0,1,0,],[0,0,0,0], [1,1,1,1],[0,1,1,1],[1,1,1,1],[1,1,1,1],[0,0,0,0],[1,1,0,0]]
    data = np.transpose(data)
    
    This should return a kappa of 0.529.
    """
    
    temp1 = np.array([labels1])
    temp2 = np.array([labels2])
    data =  np.concatenate((temp1, temp2), 0)
        
    data = np.asanyarray(data)
    
    n_coders = len(data) # I know this
    n_subjects = len(data[0]) # number of labellings (clips)
    classes = np.unique(data) # possible labels
    
    mat = np.zeros((n_subjects, len(classes)))
    
    for i in range(0,n_subjects):
        for c, cc in enumerate(classes):
            mat[i][c] = np.sum(data[:,i]== cc) 
            
    p = mat.sum(axis = 0) / (n_coders*n_subjects)
    
    P = 1/n_coders/(n_coders-1) * (np.square(mat).sum(axis=1) - n_coders)
    P_tot = np.sum(P)
    
    # over the whole sheet
    P_bar = P_tot / n_subjects;
    P_e = np.square(p).sum();

    kappa = (P_bar - P_e) / (1-P_e);
    
    return kappa


def cohens_kappa(labels1, labels2):
    conf_mat = np.zeros((2,2))
    
    labels1 = np.array(labels1)
    labels2 = np.array(labels2)
    
    conf_mat[0,0] = len((np.where(np.logical_and((labels1==0) , (labels2==0))))[0])
    conf_mat[0,1] = len((np.where(np.logical_and((labels1==0) , (labels2==1))))[0])
    conf_mat[1,0] = len((np.where(np.logical_and((labels1==1) , (labels2==0))))[0])
    conf_mat[1,1] = len((np.where(np.logical_and((labels1==1) , (labels2==1))))[0])
    
    # observed accuracy
    Pa =  np.sum(np.diag(conf_mat)) / np.sum(conf_mat)
    
    # expected accuracy
    Pe = (((np.sum(conf_mat[0, :]) * np.sum(conf_mat[:, 0])) / np.sum(conf_mat)) + ((np.sum(conf_mat[1, :]) * np.sum(conf_mat[:, 1])) / np.sum(conf_mat) ))/np.sum(conf_mat)
    
    kappa = (Pa - Pe)/(1 - Pe)
    
    return kappa

def gesture_consistency(groups1, groups2):
    """
    Computes interrater agreement accross all cameras
    """
    print('-------------------------------')                    
    for gest_name in GEST_NAMES:
        labels1 = []
        labels2 = []
        for cam_no in CAM_NOS:
            for g, group in enumerate(groups1):
                
                temp1 = groups1[g].gesture_gt[gest_name][ cam_no ] 
                temp2 = groups2[g].gesture_gt[gest_name][ cam_no ]
                
                if (temp1 is not -1) and (temp2 is not -1):
                    labels1.append( temp1 )
                    labels2.append( temp2 )
                elif((temp1 is not -1) and (temp2 is  -1)) or ((temp1 is -1) and (temp2 is not -1)):
                    print('*** Problem in a group with {} and size {} ***'.format(group.members[0].id, len(group.members)))
                        
        
        kappa = cohens_kappa(labels1, labels2)
        alpha = kriAlpha(labels1, labels2, 'nominal')
        
        print('Cohens kappa for ' + gest_name + ': %.2f'% kappa)
        print('Krip. alpha for ' + gest_name + ': %.2f'% alpha)
        print('-------------------------------')                    
    
    print('Krippendorf\'s alpha for intensity')  
    
    labels1 = []
    labels2 = []
    for g, group in enumerate(groups1):
        labels1.append(int(groups1[g].intensity))
        labels2.append(int(groups2[g].intensity))
    
    alpha = kriAlpha(labels1, labels2, 'ordinal')
    
    print('Krip. alpha for intensity' + ': %.2f'% alpha)

                                
def get_estim_perf(groups1):

    """
    After I get the gesture status of each lib of each member in each camera,
    I set a gesture status for the entire group for that camera
    
    Basically if any limb (LElbow or RElbow) of any member is detected to make 
    a gesture in a particular camera view, the gesture status of the group for 
    that camera is set to positive, otherwise it is left negative
    
    """
    gest = 'arms'
    n_hit, n_miss, n_incam, n_inconc, n_notincam = 0,0,0,0,0
    conf_mat = [[0, 0], [0,0]]

    for g in groups1: 
        for cam_no in CAM_NOS:
                
            if g.gesture_gt[gest][ cam_no ] is not -1:
                
                n_incam += 1
                
                if g.gesture_estim[gest][ cam_no ] is not -1:
                    if g.gesture_gt[gest][ cam_no ] == 1 and g.gesture_estim[gest][ cam_no ] == 1:
                        conf_mat[1][1] = conf_mat[1][1] + 1
                        n_hit += 1
                    elif g.gesture_gt[gest][ cam_no ] == 1 and g.gesture_estim[gest][ cam_no ] == 0:
                        conf_mat[1][0] = conf_mat[1][0] + 1
                        n_miss += 1
                    elif g.gesture_gt[gest][ cam_no ] == 0 and g.gesture_estim[gest][ cam_no ] == 1:
                        conf_mat[0][1] = conf_mat[0][1] + 1
                        print(cam_no, ', '.join([str(p.id) for p in g.members]))
                        n_miss += 1
                    elif g.gesture_gt[gest][ cam_no ] == 0 and g.gesture_estim[gest][ cam_no ] == 0:
                        conf_mat[0][0] = conf_mat[0][0] + 1
                        n_hit += 1
                    else:
                        print('impossible')
                            
                else:
                     n_inconc += 1   

                    
            else:
                n_notincam += 1
          
    cum_perf = (conf_mat[0][0] + conf_mat[1][1]) / np.sum(conf_mat)
    print('-------------------------------')
    print('Estimation performance for {} '.format(gest))
    print('Hit\t{}'.format(n_hit))
    print('Miss\t{}'.format(n_miss))
    print('Inconc\t{}'.format(n_inconc))
    print('Not-in-cam\t{}'.format(n_notincam))
    print('')
    print('\t EST=0 \t EST=1')
    print('GT=0\t{}\t{}'.format(conf_mat[0][0], conf_mat[0][1]))
    print('GT=1\t{}\t{}'.format(conf_mat[1][0], conf_mat[1][1]))
    print('')
    print('Cumulative performance '+ ': %.2f'%cum_perf)

    return conf_mat