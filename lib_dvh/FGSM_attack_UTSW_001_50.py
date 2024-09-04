
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:55:19 2019

aim: validate the training set during the DQN. 

@author: writed by Chenyang Shen, modified by Chao Wang
"""
import numpy as np
from .dqn_rule_network import DQN
import matplotlib.pyplot as plt
from .score_calcu import planIQ
import math as m
import pandas as pd
import tensorflow as tf
import random


INPUT_SIZE = 100  # DVH interval number
MAX_STEP = 31 # maximum No. of tuning parameter
# ------------- range of parmaeter -----------------
paraMax = 100000 # change in validation as well
paraMin = 0
paraMax_tPTV = 1.2
paraMin_tPTV = 1
paraMax_tOAR = 1
paraMax_VOAR = 1
paraMax_VPTV = 0.3
# ---------------------------------------------------

from lib_dvh.data_prep import loadDoseMatrix,loadMask,ProcessDmat
# from lib_dvh.data_prep_parth_complete_onceagain import loadDoseMatrix,loadMask,ProcessDmat

def numerical_gradient(f, x, mainDQN1, mainDQN2,mainDQN3,mainDQN4,mainDQN5,mainDQN6,mainDQN7,mainDQN8,mainDQN9, epsilon=1e-5):
    """
    Compute the numerical gradient of a function `f` at input `x`.

    Parameters:
    - f: function that takes a 3D numpy array as input and returns a scalar.
    - x: a 3D numpy array, the point at which the gradient is evaluated.
    - epsilon: a small perturbation used to compute the finite difference.

    Returns:
    - grad: a 3D numpy array representing the numerical gradient of `f` at `x`.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index

        # Save the original value
        original_value = x[idx]

        # Compute f(x + epsilon)
        x[idx] = original_value + epsilon
        fx_plus_epsilon = f(x, mainDQN1, mainDQN2,mainDQN3,mainDQN4,mainDQN5,mainDQN6,mainDQN7,mainDQN8,mainDQN9)

        # Compute f(x - epsilon)
        x[idx] = original_value - epsilon
        fx_minus_epsilon = f(x, mainDQN1, mainDQN2,mainDQN3,mainDQN4,mainDQN5,mainDQN6,mainDQN7,mainDQN8,mainDQN9)

        # Restore the original value
        x[idx] = original_value

        # Compute the partial derivative using the central difference formula
        grad[idx] = (fx_plus_epsilon - fx_minus_epsilon) / (2 * epsilon)

        it.iternext()

    return grad

# def numerical_gradient(f, x, mainDQN1, mainDQN2,mainDQN3,mainDQN4,mainDQN5,mainDQN6,mainDQN7,mainDQN8,mainDQN9, h=1e-5):
#     """
#     Compute the numerical gradient of a scalar function f at point x.
    
#     Parameters:
#     - f: The scalar function, which takes a vector (numpy array) as input and returns a scalar.
#     - x: The point (numpy array) at which to evaluate the gradient.
#     - h: A small perturbation used to approximate the gradient (default is 1e-5).
    
#     Returns:
#     - grad: The numerical gradient (numpy array) of the function f at point x.
#     """
#     grad = np.zeros_like(x)  # Initialize the gradient array with the same shape as x
    
#     # Iterate over all dimensions
#     for i in range(x.size):
#         x_plus_h = np.copy(x)
#         x_minus_h = np.copy(x)
        
#         # Perturb the i-th dimension by a small amount h
#         x_plus_h[i] += h
#         x_
#         x_minus_h[i] -= h
        
#         # Compute the difference in function values
#         f_plus_h = f(x_plus_h, mainDQN1, mainDQN2,mainDQN3,mainDQN4,mainDQN5,mainDQN6,mainDQN7,mainDQN8,mainDQN9)
#         f_minus_h = f(x_minus_h, mainDQN1, mainDQN2,mainDQN3,mainDQN4,mainDQN5,mainDQN6,mainDQN7,mainDQN8,mainDQN9)
        
#         # Approximate the derivative using the central difference formula
#         grad[i] = (f_plus_h - f_minus_h) / (2 * h)
    
#     return grad

def get_action(X, mainDQN1, mainDQN2,mainDQN3,mainDQN4,mainDQN5,mainDQN6,mainDQN7,mainDQN8,mainDQN9):
    TotalArray = np.zeros((9,3))
    tempoutput = mainDQN1.model.predict(X)
    # print('tempoutput', tempoutput)
    TotalArray[0] = tempoutput
    tempoutput1 = tempoutput[0,:]
    # print('tempoutput1', tempoutput)
    tempoutput = mainDQN2.model.predict(X)
    TotalArray[1] = tempoutput
    tempoutput2 = tempoutput[0, :]
    tempoutput = mainDQN3.model.predict(X)
    TotalArray[2] = tempoutput
    tempoutput3 = tempoutput[0, :]
    tempoutput = mainDQN4.model.predict(X)
    TotalArray[3] = tempoutput
    tempoutput4 = tempoutput[0, :]
    tempoutput = mainDQN5.model.predict(X)
    TotalArray[4] = tempoutput
    tempoutput5 = tempoutput[0, :]
    tempoutput = mainDQN6.model.predict(X)
    TotalArray[5] = tempoutput
    tempoutput6 = tempoutput[0,:]
    tempoutput = mainDQN7.model.predict(X)
    TotalArray[6] = tempoutput
    tempoutput7 = tempoutput[0, :]
    tempoutput = mainDQN8.model.predict(X)
    TotalArray[7] = tempoutput
    tempoutput8 = tempoutput[0, :]
    tempoutput = mainDQN9.model.predict(X)
    TotalArray[8] = tempoutput
    tempoutput9 = tempoutput[0, :]
    # print('TotalArray', TotalArray)
    des_action = np.max(TotalArray)
    # print('des_action', des_action)

    return des_action


def evalu_training(mainDQN1: DQN, mainDQN2: DQN, mainDQN3: DQN, 
             mainDQN4: DQN,  mainDQN5: DQN, mainDQN6: DQN,
             mainDQN7: DQN,  mainDQN8: DQN, mainDQN9: DQN,
             runOpt_dvh, episode,flagg,pdose,maxiter) -> None:
    test_set =['10']#['18','20','22','23','25','26','27','28','30','31','36','37','42','43','45','46','54','57','61','65','66','68','70','73','74','77','80','81','83','84','85','87','88','91','92','93','95','97','98'] #['12','17'] # training set
    for sampleid in range(1):
        id = test_set[sampleid]
        print('############################# Testing start for patient '+ id +' #################################################')
        # data_path= '/data/data/dose_deposition3/f_dijs/0'
        data_path = '/data2/tensorflow_utsw/dose_deposition/prostate_dijs/f_dijs/0'
        # data_path = '/home/mainul1/DQN/'
        # data_path2='/data/data/dose_deposition3/plostate_dijs/f_masks/0'
        data_path2 = '/home/mainul1/Actor-critic-based-treatment-planning/acer_VTPN-12-18-23/data/data/dose_deposition3/plostate_dijs/f_masks/0'
        # data_path2 = '/home/mainul1/DQN/'
        doseMatrix_test = loadDoseMatrix(data_path + id + '.hdf5')
        # doseMatrix_test = loadDoseMatrix(data_path + 'test_onceagain.hdf5')
        targetLabels_test, bladderLabeltest, rectumLabeltest, PTVLabeltest = loadMask(data_path2+ id + '.h5')
        # targetLabels_test, bladderLabeltest, rectumLabeltest, PTVLabeltest = loadMask("test_dose_mask_onceagain.h5","test_structure_mask_onceagain.h5",)

        # ---------------- generate the linear systems -----------------------------
        MPTV, MBLA, MREC, MBLA1, MREC1 = ProcessDmat(doseMatrix_test, 
                                                     targetLabels_test, 
                                                     bladderLabeltest, 
                                                     rectumLabeltest)
          # ------------------------ initial paramaters & input --------------------
        actionArray = np.empty((50,2), dtype = object)
        for rep in range(50):
            print('=========================================Iteration',rep,'=====================================')

            planScore = 8.5
            # ------------------------ initial paramaters & input --------------------
            # Lower limit of the following block(except of tPTV) is being updated =[0, 0.1, [0.5, 0.3, 0.1, 0.3, 0.3, 0.1, 0.2, 0.4], [0.7, 0.3, 0.1, 0.3, 0.3, 0.1, 0.2, 0.4], [0.3, 0.3, 0.3, 0.3, 0.3, 0.1, 0.3, 0.3]]
            while (planScore >= 6):
                # this is the general block
                epsilon = 1e-10
                tPTV = random.uniform(1 + epsilon, 1.2 - epsilon)
                tBLA = random.uniform(0.3 + epsilon, 1 - epsilon)
                tREC = random.uniform(0.3 + epsilon, 1 - epsilon)
                lambdaPTV = random.uniform(0.3 + epsilon, 1 - epsilon)
                lambdaBLA = random.uniform(0.3 + epsilon, 1 - epsilon)
                lambdaREC = random.uniform(0.3 + epsilon, 1 - epsilon)
                VPTV = random.uniform(0.1 + epsilon, 0.3 - epsilon)
                VBLA = random.uniform(0.3 + epsilon, 1 - epsilon)
                VREC = random.uniform(0.3 + epsilon, 1 - epsilon)
        # tPTV = 1
        # tBLA = 1
        # tREC = 1
        # lambdaPTV = 1
        # lambdaBLA = 2
        # lambdaREC = 2
        # VPTV = 0.1
        # VBLA = 1
        # VREC = 1
                xVec = np.ones((MPTV.shape[1],))
                gamma = np.zeros((MPTV.shape[0],))
                # --------------------- solve treatment planning optmization -----------------------------
                state_test0, iter, xVec, gamma = \
                    runOpt_dvh(MPTV, MBLA, MREC,tPTV,tBLA, tREC, 
                               lambdaPTV, lambdaBLA, lambdaREC, 
                               VPTV, VBLA, VREC, xVec,gamma,pdose,maxiter)
                 # --------------------- generate input for NN -----------------------------                                   
                
                DPTV = MPTV.dot(xVec)
                DBLA = MBLA.dot(xVec)
                DREC = MREC.dot(xVec)
                # DPTV=np.array(DPTVex)
                # DBLA=np.array(DBLAex)
                # DREC=np.array(DRECex)
                DPTV = np.sort(DPTV)
                DPTV = np.flipud(DPTV)
                DBLA = np.sort(DBLA)
                DBLA = np.flipud(DBLA)
                DREC = np.sort(DREC)
                DREC = np.flipud(DREC)
                edge_ptv = np.zeros((1000 + 1,))
                edge_ptv[1:1000 + 1] = np.linspace(0, max(DPTV), 1000)
                x_ptv = np.linspace(0.5 * max(DPTV) / 1000, max(DPTV), 1000)
                (n_ptv, b) = np.histogram(DPTV, bins=edge_ptv)
                y_ptv = 1 - np.cumsum(n_ptv / len(DPTV), axis=0)

                edge_bladder = np.zeros((1000 + 1,))
                edge_bladder[1:1000 + 1] = np.linspace(0, max(DBLA), 1000)
                x_bladder = np.linspace(0.5 * max(DBLA) / 1000, max(DBLA), 1000)
                (n_bladder, b) = np.histogram(DBLA, bins=edge_bladder)
                y_bladder = 1 - np.cumsum(n_bladder / len(DBLA), axis=0)

                edge_rectum = np.zeros((1000 + 1,))
                edge_rectum[1:1000 + 1] = np.linspace(0, max(DREC), 1000)
                x_rectum = np.linspace(0.5 * max(DREC) / 1000, max(DREC), 1000)
                (n_rectum, b) = np.histogram(DREC, bins=edge_rectum)
                y_rectum = 1 - np.cumsum(n_rectum / len(DREC), axis=0)

                Y = np.zeros((1000, 3))
                Y[:, 0] = y_ptv
                Y[:, 1] = y_bladder
                Y[:, 2] = y_rectum

                X = np.zeros((1000, 3))
                X[:, 0] = x_ptv
                X[:, 1] = x_bladder
                X[:, 2] = x_rectum
                data_result_path="/home/mainul1/DQN/ResultsUTSW001/"
                np.save(data_result_path+id+'xDVHYInitial',Y)
                np.save(data_result_path+id+'xDVHXInitial',X)

                tPTV_all = np.zeros((MAX_STEP + 1))
                tBLA_all = np.zeros((MAX_STEP + 1))
                tREC_all = np.zeros((MAX_STEP + 1))
                lambdaPTV_all = np.zeros((MAX_STEP + 1))
                lambdaBLA_all = np.zeros((MAX_STEP + 1))
                lambdaREC_all = np.zeros((MAX_STEP + 1))
                VPTV_all = np.zeros((MAX_STEP + 1))
                VBLA_all = np.zeros((MAX_STEP + 1))
                VREC_all = np.zeros((MAX_STEP + 1))
                planScore_all = np.zeros((MAX_STEP + 1))
                planScore_fine_all = np.zeros((MAX_STEP + 1))
                planScore_fine, planScore,scoreall = planIQ(MPTV, MBLA1, MREC1, xVec,pdose)
                print("Iteration_num: {}  PlanScore: {}  PlanScore_fine: {} ScoreAll: {}".format(iter, planScore, planScore_fine, scoreall))

                tPTV_all[0] = tPTV
                tBLA_all[0] = tBLA
                tREC_all[0] = tREC
                lambdaPTV_all[0] = lambdaPTV
                lambdaBLA_all[0] = lambdaBLA
                lambdaREC_all[0] = lambdaREC
                VPTV_all[0] = VPTV
                VBLA_all[0] = VBLA
                VREC_all[0] = VREC
                planScore_all[0] = planScore
                planScore_fine_all[0] = planScore_fine
            
            state_test = state_test0
            # -------------- NN ----------------------------------
            for i in range(2):
                if i ==0:
                    X = np.reshape(state_test,(1,INPUT_SIZE,3))
                # X = tf.convert_to_tensor(X, dtype=tf.float32)
                TotalArray = np.zeros((9,3))
                tempoutput = mainDQN1.model.predict(X)
                print('tempoutput', tempoutput)
                TotalArray[0] = tempoutput
                tempoutput1 = tempoutput[0,:]
                print('tempoutput1', tempoutput)
                tempoutput = mainDQN2.model.predict(X)
                TotalArray[1] = tempoutput
                tempoutput2 = tempoutput[0, :]
                tempoutput = mainDQN3.model.predict(X)
                TotalArray[2] = tempoutput
                tempoutput3 = tempoutput[0, :]
                tempoutput = mainDQN4.model.predict(X)
                TotalArray[3] = tempoutput
                tempoutput4 = tempoutput[0, :]
                tempoutput = mainDQN5.model.predict(X)
                TotalArray[4] = tempoutput
                tempoutput5 = tempoutput[0, :]
                tempoutput = mainDQN6.model.predict(X)
                TotalArray[5] = tempoutput
                tempoutput6 = tempoutput[0,:]
                tempoutput = mainDQN7.model.predict(X)
                TotalArray[6] = tempoutput
                tempoutput7 = tempoutput[0, :]
                tempoutput = mainDQN8.model.predict(X)
                TotalArray[7] = tempoutput
                tempoutput8 = tempoutput[0, :]
                tempoutput = mainDQN9.model.predict(X)
                TotalArray[8] = tempoutput
                tempoutput9 = tempoutput[0, :]
                print('TotalArray', TotalArray)
                des_action = np.max(TotalArray)
                print('des_action', des_action)
                # gradient = numerical_gradient(get_action, X, mainDQN1, mainDQN2,mainDQN3,mainDQN4,mainDQN5,mainDQN6,mainDQN7,mainDQN8,mainDQN9,epsilon=1e-5)
                # print('gradient', gradient)
                actionArray[rep, i] = np.array(np.unravel_index(np.argmax(TotalArray), TotalArray.shape))
                print('actionArray', actionArray)

                if i == 0:

                    gradient = numerical_gradient(get_action, X, mainDQN1, mainDQN2,mainDQN3,mainDQN4,mainDQN5,mainDQN6,mainDQN7,mainDQN8,mainDQN9,epsilon=1e-5)
                    print('gradient', gradient)
                    X = X-(0.001*np.sign(gradient))
                    # Xstore = np.reshape(X, )
                    print('X', X)
            
            if rep == 49:
                print('actionArray', actionArray)
                np.save(data_result_path+'actionArray0.001', actionArray)
# =========================== rule-based =============================
#                 if case==0: # Rule-based 
#                     if scoreall[0]<0.01:
#                         tempoutput4[1] = -100 # only allow lambda_ptv increases 
#                         tempoutput4[2] = -100
# 
#                         tempoutput1[0] = -100 # only allow t_ptv decreases
#                         tempoutput1[1] = -100
# 
#                         tempoutput7[0] = -100 # only allow v_ptv decreases
#                         tempoutput7[1] = -100
# 
#                         tempoutput5[0] = -100 # only allow lambda_bla decreases
#                         tempoutput5[1] = -100
# 
#                         tempoutput6[0] = -100 # only allow lambda_rec decreases
#                         tempoutput6[1] = -100
# 
#                     else:
#                         scorebladder = scoreall[1]+scoreall[2]+scoreall[3]+scoreall[4]
#                         scorerectum = scoreall[5] + scoreall[6] + scoreall[7] + scoreall[8]
#                         if scorebladder<2 and scorebladder<scorerectum: # need to improve bla
#                             tempoutput5[1] = -100 # only allow lambda_bla increases 
#                             tempoutput5[2] = -100
#     
#                             tempoutput2[0] = -100 # only allow t_bla decreases
#                             tempoutput2[1] = -100
#     
#                             tempoutput8[0] = -100 # only allow v_bla decreases
#                             tempoutput8[1] = -100
#     
#                             tempoutput4[0] = -100 # only allow lambda_ptv decreases
#                             tempoutput4[1] = -100
#     
#                             tempoutput6[0] = -100 # only allow lambda_rec decreases
#                             tempoutput6[1] = -100
#     
#                         if scorerectum<2 and scorerectum<scorebladder:
#                             tempoutput6[1] = -100 # only allow lambda_rec increases 
#                             tempoutput6[2] = -100
#     
#                             tempoutput3[0] = -100 # only allow t_rec decreases
#                             tempoutput3[1] = -100
#     
#                             tempoutput9[0] = -100 # only allow v_rec decreases
#                             tempoutput9[1] = -100
#     
#                             tempoutput4[0] = -100 # only allow lambda_ptv decreases
#                             tempoutput4[1] = -100
#     
#                             tempoutput5[0] = -100 # only allow lambda_bla decreases
#                             tempoutput5[1] = -100
# =============================================================================

# =================================The next part is commented out to try FGSM Attack====================
#                 value = np.zeros(9)
#                 value[0] = np.max(tempoutput1)
#                 value[1] = np.max(tempoutput2)
#                 value[2] = np.max(tempoutput3)
#                 value[3] = np.max(tempoutput4)
#                 value[4] = np.max(tempoutput5)
#                 value[5] = np.max(tempoutput6)
#                 value[6] = np.max(tempoutput7)
#                 value[7] = np.max(tempoutput8)
#                 value[8] = np.max(tempoutput9)
#                 print('value array', value)

#                 ######################################## tune parameter according to NN  ############################################
#                 paraidx = np.argmax(value)
#                 print('paraidx',paraidx)
#                 if paraidx == 0:
#                     action = np.argmax(tempoutput1)
#                     if action == 0:
#                         action_factor = 1.01
#                         tPTV = tPTV * action_factor
#                         if tPTV >= paraMax_tPTV:
#                             tPTV = paraMax_tPTV
#                     elif action == 1:
#                         action_factor = 1
#                         tPTV = tPTV *  action_factor
#                         if tPTV >= paraMax_tPTV:
#                             tPTV = paraMax_tPTV
#                     else:
#                         action_factor = 0.09
#                         tPTV = tPTV * action_factor
#                         if tPTV <= paraMin_tPTV:
#                             tPTV = paraMin_tPTV

#                 if paraidx == 1:
#                     action = np.argmax(tempoutput2)
#                     if action == 0:
#                         action_factor = 1.25
#                         tBLA = tBLA * action_factor
#                         if tBLA >= paraMax_tOAR:
#                             tBLA = paraMax_tOAR
#                     elif action == 1:
#                         action_factor = 1
#                         tBLA = tBLA * action_factor
#                         if tBLA >= paraMax_tOAR:
#                             tBLA = paraMax_tOAR
#                     else:
#                         action_factor = 0.8
#                         tBLA = tBLA * action_factor
#                         if tBLA <= paraMin:
#                             tBLA = paraMin

#                 if paraidx == 2:
#                     action = np.argmax(tempoutput3)
#                     if action == 0:
#                         action_factor = 1.25
#                         tREC = tREC * action_factor
#                         if tREC >= paraMax_tOAR:
#                             tREC = paraMax_tOAR
#                     elif action == 1:
#                         action_factor = 1
#                         tREC = tREC * action_factor
#                         if tREC >= paraMax_tOAR:
#                             tREC = paraMax_tOAR
#                     else:
#                         action_factor = 0.8
#                         tREC = tREC * action_factor
#                         if tREC <= paraMin:
#                             tREC = paraMin

#                 if paraidx == 3:
#                     action = np.argmax(tempoutput4)
#                     if action == 0:
#                         action_factor = m.exp(0.5)
#                         lambdaPTV = lambdaPTV * action_factor
#                         if lambdaPTV >= paraMax:
#                             lambdaPTV = paraMax
#                     elif action == 1:
#                         action_factor = 1
#                         lambdaPTV = lambdaPTV * action_factor
#                         if lambdaPTV >= paraMax:
#                             lambdaPTV = paraMax
#                     else:
#                         action_factor = m.exp(-0.5)
#                         lambdaPTV = lambdaPTV * action_factor
#                         if lambdaPTV <= paraMin:
#                             lambdaPTV = paraMin

#                 if paraidx == 4:
#                     action = np.argmax(tempoutput5)
#                     if action == 0:
#                         action_factor = m.exp(0.5)
#                         lambdaBLA = lambdaBLA * action_factor
#                         if lambdaBLA >= paraMax:
#                             lambdaBLA = paraMax
#                     elif action == 1:
#                         action_factor = 1
#                         lambdaBLA = lambdaBLA * action_factor
#                         if lambdaBLA >= paraMax:
#                             lambdaBLA = paraMax
#                     else:
#                         action_factor = m.exp(-0.5)
#                         lambdaBLA = lambdaBLA * action_factor
#                         if lambdaBLA <= paraMin:
#                             lambdaBLA = paraMin
#                 if paraidx == 5:
#                     action = np.argmax(tempoutput6)
#                     if action == 0:
#                         action_factor = m.exp(0.5)
#                         lambdaREC = lambdaREC * action_factor
#                         if lambdaREC >= paraMax:
#                             lambdaREC = paraMax
#                     elif action == 1:
#                         action_factor = 1
#                         lambdaREC = lambdaREC * action_factor
#                         if lambdaREC >= paraMax:
#                             lambdaREC = paraMax
#                     else:
#                         action_factor = m.exp(-0.5)
#                         lambdaREC = lambdaREC* action_factor
#                         if lambdaREC <= paraMin:
#                             lambdaREC = paraMin
#                 if paraidx == 6:
#                     action = np.argmax(tempoutput7)
#                     if action == 0:
#                         action_factor = 1.4
#                         VPTV = VPTV * action_factor
#                         if VPTV >= paraMax_VPTV:
#                             VPTV = paraMax_VPTV
#                     elif action == 1:
#                         action_factor = 1
#                         VPTV = VPTV * action_factor
#                         if VPTV >= paraMax:
#                             VPTV = paraMax
#                     else:
#                         action_factor = 0.6 
#                         VPTV = VPTV * action_factor
#                         if VPTV <= paraMin:
#                             VPTV = paraMin

#                 if paraidx == 7:
#                     action = np.argmax(tempoutput8)
#                     if action == 0:
#                         action_factor = 1.25
#                         VBLA = VBLA * action_factor
#                         if VBLA >= paraMax_VOAR:
#                             VBLA = paraMax_VOAR
#                     elif action == 1:
#                        action_factor = 1
#                        VBLA = VBLA * action_factor
#                        if VBLA >= paraMax_VOAR:
#                             VBLA = paraMax_VOAR
#                        else:
#                         action_factor = 0.8
#                         VBLA = VBLA * action_factor
#                         if VBLA <= paraMin:
#                             VBLA = paraMin
#                 if paraidx == 8:
#                     action = np.argmax(tempoutput9)
#                     if action == 0:
#                         action_factor = 1.25
#                         VREC = VREC * action_factor
#                         if VREC >= paraMax_VOAR:
#                             VREC = paraMax_VOAR
#                     elif action == 1:
#                         action_factor = 1
#                         VREC = VREC * action_factor
#                         if VREC >= paraMax_VOAR:
#                             VREC = paraMax_VOAR
#                     else:
#                         action_factor = 0.8
#                         VREC = VREC * action_factor
#                         if VREC <= paraMin:
#                             VREC = paraMin

#                 # --------------------- solve treatment planning optmization -----------------------------
#                 if action != 1:
#                     xVec = np.ones((MPTV.shape[1],))
#                     gamma = np.zeros((MPTV.shape[0],))
#                     state_test, iter, xVec , gamma = \
#                             runOpt_dvh(MPTV, MBLA, MREC,tPTV,tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, xVec,gamma,pdose,maxiter)

#                 planScore_fine,  planScore, scoreall = planIQ(MPTV, MBLA1, MREC1, xVec,pdose)
#                 print("Step: {} Iteration_num: {}  PlanScore: {}  PlanScore_fine: {}".format(i,iter, planScore, planScore_fine))
                
#                 # collect the result in each iteration 
#                 tPTV_all[i + 1] = tPTV
#                 tBLA_all[i + 1] = tBLA
#                 tREC_all[i + 1] = tREC
#                 lambdaPTV_all[i + 1] = lambdaPTV
#                 lambdaBLA_all[i + 1] = lambdaBLA
#                 lambdaREC_all[i + 1] = lambdaREC
#                 VPTV_all[i + 1] = VPTV
#                 VBLA_all[i + 1] = VBLA
#                 VREC_all[i + 1] = VREC
#                 planScore_all[i + 1] = planScore
#                 planScore_fine_all[i + 1] = planScore_fine
#                 DPTV = MPTV.dot(xVec)
#                 DBLA = MBLA1.dot(xVec)
#                 DREC = MREC1.dot(xVec)
#                 DPTV = np.sort(DPTV)
#                 DPTV = np.flipud(DPTV)
#                 DBLA = np.sort(DBLA)
#                 DBLA = np.flipud(DBLA)
#                 DREC = np.sort(DREC)
#                 DREC = np.flipud(DREC)
#                 edge_ptv = np.zeros((1000 + 1,))
#                 edge_ptv[1:1000 + 1] = np.linspace(0, max(DPTV), 1000)
#                 x_ptv = np.linspace(0.5 * max(DPTV) / 1000, max(DPTV), 1000)
#                 (n_ptv, b) = np.histogram(DPTV, bins=edge_ptv)
#                 y_ptv = 1 - np.cumsum(n_ptv / len(DPTV), axis=0)

#                 edge_bladder = np.zeros((1000 + 1,))
#                 edge_bladder[1:1000 + 1] = np.linspace(0, max(DBLA), 1000)
#                 x_bladder = np.linspace(0.5 * max(DBLA) / 1000, max(DBLA), 1000)
#                 (n_bladder, b) = np.histogram(DBLA, bins=edge_bladder)
#                 y_bladder = 1 - np.cumsum(n_bladder / len(DBLA), axis=0)

#                 edge_rectum = np.zeros((1000 + 1,))
#                 edge_rectum[1:1000 + 1] = np.linspace(0, max(DREC), 1000)
#                 x_rectum = np.linspace(0.5 * max(DREC) / 1000, max(DREC), 1000)
#                 (n_rectum, b) = np.histogram(DREC, bins=edge_rectum)
#                 y_rectum = 1 - np.cumsum(n_rectum / len(DREC), axis=0)

#                 Y = np.zeros((1000, 3))
#                 Y[:, 0] = y_ptv
#                 Y[:, 1] = y_bladder
#                 Y[:, 2] = y_rectum

#                 X = np.zeros((1000, 3))
#                 X[:, 0] = x_ptv
#                 X[:, 1] = x_bladder
#                 X[:, 2] = x_rectum


# ===================================================================================================
                #data_result_path2='/data/data/Results/GPU2/figuresrm75/'
                #plt.plot(x_ptv, y_ptv)
                #plt.plot(x_bladder, y_bladder)
                #plt.plot(x_rectum, y_rectum)
                #plt.legend(('ptv','bladder','rectum'))
                #plt.show(block=False)
                #plt.title('DVH'+str(episode)+'step'+str(i+1))
                #plt.savefig(data_result_path2+id+'DVH'+str(episode)+'step'+str(i+1)+'.png')
                #plt.close()

    #             if planScore==9:
    #                 print("parameter tuning is done at step: {} ".format(i+1))
    #                 break
    #         plt.plot(planScore_all)
    #         plt.show(block=False)
    #         plt.pause(5)
    #         data_result_path='/data/data/Results/GPU2/planScore180/'
    #         np.save(
    #             data_result_path+id+'planScore' + str(episode) + 'case' +str(case),
    #             planScore_all)
    #         np.save(
    #             data_result_path+id+'planScore_fine' + str(episode) + 'case' +str(case),
    #         planScore_fine_all)
          
    # path = '/data/data/Results/GPU2/planScore180/'
    # test_set = ['01','12','17','24','29','35','40','50','59','62']#['18','20','22','23','25','26','27','28','30','31','36','37','42','43','45','46','54','57','61','65','66','68','70','73','74','77','80','81','83','84','85','87','88','91','92','93','95','97','98'] #['12','17']
    # x = np.arange(31)
    # tem = 0
    # tem_inital = 0
    # for case in range(10):
    #     tem_score =  np.load(path+test_set[case]+'planScore'+str(episode)+'case0.npy')
    #     plt.plot(x,tem_score,label='$Case{case}$'.format(case=case))
    #     tem = max(tem_score) + tem
    #     tem_inital = tem_inital + tem_score[0]
    # #plt.legend(['case 1','case 2'],loc='lower right')
    # plt.legend(loc='best')
    # plt.show()
    # plt.title("episode: {} initial score: {}, mean score: {}".format(episode,round(tem_inital/2,2),round(tem/2,2)))
    # plt.savefig('/data/data/Results/GPU2/figures180/' + str(episode)+'.png')

  



