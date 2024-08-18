import numpy as np
import tensorflow as tf
import random as rnd
from collections import deque
import dqn_dvh_external_network
import h5sparse
import h5py
import time
import math as m
import scipy.io as sio
from numpy import *
import scipy.linalg
import matplotlib.pyplot as plt
import os
from tensorflow.python.framework import dtypes
from typing import List
import datetime
import pandas as pd
from numpy import linalg as LA
from scipy.sparse import vstack
from pathlib import Path

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from scipy.sparse import csr_matrix
import skcuda.linalg as linalg
import skcuda.misc as misc
import numpy.linalg as LA
# import cupy as cp

INPUT_SIZE = 100  # DVH interval number
OUTPUT_SIZE = 5  # number of actions, each lambda has three actions(+,=,-)
TRAIN_NUM = 10
DISCOUNT_RATE = 0.30
REPLAY_MEMORY = 5000
BATCH_SIZE = 16
TARGET_UPDATE_FREQUENCY = 10
MAX_EPISODES = 200
MAX_STEP = 30
load_session = 0
save_session = 1
Start = 1
IfRandom=0
paraMax = 1
paraMin = 0
LoadEpoch = 150
pdose = 1
maxiter = 40
valnum = 2
NET_NUM = 8

linalg.init()

def get_copy_var_ops(*, dest_scope_name: str, src_scope_name: str) -> List[tf.Operation]:

    op_holder = []

    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder


def replay_train(mainDQN: dqn_dvh_external_network.DQN, targetDQN1: dqn_dvh_external_network.DQN, targetDQN2: dqn_dvh_external_network.DQN, targetDQN3: dqn_dvh_external_network.DQN, targetDQN4: dqn_dvh_external_network.DQN, targetDQN5: dqn_dvh_external_network.DQN, targetDQN6: dqn_dvh_external_network.DQN, targetDQN7: dqn_dvh_external_network.DQN, targetDQN8: dqn_dvh_external_network.DQN, train_batch: list) -> float:
    states = np.vstack([x[0] for x in train_batch])
    actions = np.vstack([x[1] for x in train_batch])
    rewards = np.vstack([x[2] for x in train_batch])
    next_states = np.vstack([x[3] for x in train_batch])
    done = np.vstack([x[4] for x in train_batch])

    X = states
    Yout = zeros([BATCH_SIZE,NET_NUM*5])
    Yout[:, 0:5] = targetDQN1.predict(next_states)
    Yout[:, 5:10] = targetDQN2.predict(next_states)
    Yout[:, 10:15] = targetDQN3.predict(next_states)
    Yout[:, 15:20] = targetDQN4.predict(next_states)
    Yout[:, 20:25] = targetDQN5.predict(next_states)
    Yout[:, 25:30] = targetDQN6.predict(next_states)
    Yout[:, 30:35] = targetDQN7.predict(next_states)
    Yout[:, 35:40] = targetDQN8.predict(next_states)
    Q_target = rewards + DISCOUNT_RATE * np.max(Yout, axis=1)* ~done
    y = mainDQN.predict(X)
    y[np.arange(len(X)), actions.astype('int64')] = Q_target
    return mainDQN.update(X, y)

def ProcessDmat(doseMatrix, targetLabels):
    x = np.ones((doseMatrix.shape[1],))
    MPTVtemp = doseMatrix[targetLabels == 1, :]
    DPTV = MPTVtemp.dot(x)
    MPTV = MPTVtemp[DPTV != 0]
    MBLAtemp = doseMatrix[targetLabels == 2, :]
    DBLA = MBLAtemp.dot(x)
    MBLA = MBLAtemp[DBLA != 0]
    MRECtemp = doseMatrix[targetLabels == 3, :]
    DREC = MRECtemp.dot(x)
    MREC = MRECtemp[DREC != 0]
    return MPTV, MBLA, MREC


def planIQ(MPTV, MBLA, MREC, xVec):
    DPTV = MPTV.dot(xVec)
    DBLA = MBLA.dot(xVec)
    DREC = MREC.dot(xVec)
    DPTV = np.sort(DPTV)
    DPTV = np.flipud(DPTV)
    DBLA = np.sort(DBLA)
    DBLA = np.flipud(DBLA)
    DREC = np.sort(DREC)
    DREC = np.flipud(DREC)


    # DPTV1 = DPTV[DPTV>pdose]
    # if DPTV1.shape[0]/DPTV.shape[0]>=0.98:
    #     score1 = 1
    # else:
    #     score1 = 0
    tt = time.time()
    ind = round(0.03/0.015)-1
    a = 1/(1.07*pdose-1.1*pdose)
    b = 1-a*1.07*pdose
    score2 = a*(DPTV[ind]+DPTV[ind+1]+DPTV[ind+2]+DPTV[ind+3]+DPTV[ind-1])/5+b
    if score2>1:
        score2=2
    if score2<0:
        score2=0
    delta2 = 0.08
    # score2_fine = (1/pi*np.arctan(-((DPTV[ind]+DPTV[ind+1]+DPTV[ind+2]+DPTV[ind+3]+DPTV[ind-1])/5-(1.07*pdose+1.1*pdose)/2)/delta2)+0.5)*5
    if (DPTV[ind]+DPTV[ind+1]+DPTV[ind+2]+DPTV[ind+3]+DPTV[ind-1])/5>1.07:
        score2_fine = (1 / pi * np.arctan(-((DPTV[ind] + DPTV[ind + 1] + DPTV[ind + 2] + DPTV[ind + 3] + DPTV[ind - 1]) / 5 - (1.07 * pdose + 1.1 * pdose) / 2) / delta2) + 0.5)*8
    else:
        score2_fine=6
    # score2_fine = score2



    DBLA1 = DBLA[DBLA>=pdose*1.01]
    a = 1/(0.15-0.20)
    b = 1-a*0.15
    score3 = a*DBLA1.shape[0]/DBLA.shape[0]+b
    if score3>1:
        score3=1
    if score3<0:
        score3=0
    delta3 = 0.05
    if DBLA1.shape[0]/DBLA.shape[0]<0.2:
        score3_fine = 1 / pi * np.arctan(-(DBLA1.shape[0]/DBLA.shape[0] - (0.15 + 0.20) / 2) / delta3) + 0.5
    else:
        score3_fine=0

    DBLA2 = DBLA[DBLA >= pdose * 0.947]
    a = 1 / (0.25 - 0.30)
    b = 1 - a * 0.25
    score4 = a * DBLA2.shape[0] / DBLA.shape[0] + b
    if score4 > 1:
        score4 = 1
    if score4 < 0:
        score4 = 0
    delta4 = 0.05
    if DBLA2.shape[0] / DBLA.shape[0]<0.3:
        score4_fine = 1 / pi * np.arctan(-(DBLA2.shape[0] / DBLA.shape[0] - (0.25 + 0.30) / 2) / delta4) + 0.5
    else:
        score4_fine = 0

    DBLA3 = DBLA[DBLA >= pdose * 0.8838]
    a = 1 / (0.35 - 0.40)
    b = 1 - a * 0.35
    score5 = a * DBLA3.shape[0] / DBLA.shape[0] + b
    if score5 > 1:
        score5 = 1
    if score5 < 0:
        score5 = 0
    delta5 = 0.05
    if DBLA3.shape[0] / DBLA.shape[0]<0.4:
        score5_fine = 1 / pi * np.arctan(-(DBLA3.shape[0] / DBLA.shape[0] - (0.35 + 0.40) / 2) / delta5) + 0.5
    else:
        score5_fine = 0

    DBLA4 = DBLA[DBLA >= pdose * 0.8207]
    a = 1 / (0.5 - 0.55)
    b = 1 - a * 0.5
    score6 = a * DBLA4.shape[0] / DBLA.shape[0] + b
    if score6 > 1:
        score6 = 1
    if score6 < 0:
        score6 = 0
    delta6 = 0.05
    if DBLA4.shape[0] / DBLA.shape[0]<0.55:
        score6_fine = 1 / pi * np.arctan(-(DBLA4.shape[0] / DBLA.shape[0] - (0.5 + 0.55) / 2) / delta6) + 0.5
    else:
        score6_fine = 0



    DREC1 = DREC[DREC >= pdose * 0.947]
    a = 1 / (0.15 - 0.20)
    b = 1 - a * 0.15
    score7 = a * DREC1.shape[0] / DREC.shape[0] + b
    if score7 > 1:
        score7 = 1
    if score7 < 0:
        score7 = 0
    delta7 = 0.05
    if DREC1.shape[0] / DREC.shape[0]<0.2:
        score7_fine = 1 / pi * np.arctan(-(DREC1.shape[0] / DREC.shape[0] - (0.15 + 0.20) / 2) / delta7) + 0.5
    else:
        score7_fine = 0



    DREC2 = DREC[DREC >= pdose * 0.8838]
    a = 1 / (0.25 - 0.30)
    b = 1 - a * 0.25
    score8 = a * DREC2.shape[0] / DREC.shape[0] + b
    if score8 > 1:
        score8 = 1
    if score8 < 0:
        score8 = 0
    delta8 = 0.05
    if DREC2.shape[0] / DREC.shape[0]<0.3:
        score8_fine = 1 / pi * np.arctan(-(DREC2.shape[0] / DREC.shape[0] - (0.25 + 0.30) / 2) / delta8) + 0.5
    else:
        score8_fine = 0

    DREC3 = DREC[DREC >= pdose * 0.8207]
    a = 1 / (0.35 - 0.40)
    b = 1 - a * 0.35
    score9 = a * DREC3.shape[0] / DREC.shape[0] + b
    if score9 > 1:
        score9 = 1
    if score9 < 0:
        score9 = 0
    delta9 = 0.05
    if DREC3.shape[0] / DREC.shape[0]<0.4:
        score9_fine = 1 / pi * np.arctan(-(DREC3.shape[0] / DREC.shape[0] - (0.35 + 0.40) / 2) / delta9) + 0.5
    else:
        score9_fine = 0

    DREC4 = DREC[DREC >= pdose * 0.7576]
    a = 1 / (0.50 - 0.55)
    b = 1 - a * 0.50
    score10 = a * DREC4.shape[0] / DREC.shape[0] + b
    if score10 > 1:
        score10 = 1
    if score10 < 0:
        score10 = 0
    delta10 = 0.05
    if DREC4.shape[0] / DREC.shape[0]<0.55:
        score10_fine = 1 / pi * np.arctan(-(DREC4.shape[0] / DREC.shape[0] - (0.50 + 0.55) / 2) / delta10) + 0.5
    else:
        score10_fine = 0
    elapsedTime = time.time()-tt
    print('time:{}',format(elapsedTime))


    score = score2+score3+score4+score5+score6+score7+score8+score9+score10
    score_fine = score2_fine + score3_fine + score4_fine + score5_fine + score6_fine + score7_fine + score8_fine + score9_fine + score10_fine
    # print("score2: {}  score3: {}  score4: {}  score5: {}  score6: {}  score7: {}  score8: {} score9: {} score10: {}".format(
    #     score2, score3, score4, score5, score6, score7, score8, score9, score10))
    return score_fine, score


def bot_play(mainDQN1: dqn_dvh_external_network.DQN, mainDQN2: dqn_dvh_external_network.DQN, mainDQN3: dqn_dvh_external_network.DQN, mainDQN4: dqn_dvh_external_network.DQN,  mainDQN5: dqn_dvh_external_network.DQN, mainDQN6: dqn_dvh_external_network.DQN, mainDQN7: dqn_dvh_external_network.DQN, mainDQN8: dqn_dvh_external_network.DQN,DVHplot,MinimizeDoseOARnew1,episode) -> None:
    for val in range(valnum):
        if val==0:
            id = '12'
        if val==1:
            id = '17'
        data_path= '/data/data/dose_deposition/prostate_dijs/f_dijs/0'
        data_path2='/data/data/dose_deposition3/plostate_dijs/f_masks/0'
        doseMatrix_test = loadDoseMatrix(data_path +id+'.hdf5')
        targetLabels_test = loadMask(data_path2 +id+'.h5')
        # lambda6 = 1 # target dose PTV4.67
        lambdaPTV = 1
        lambdaBLA = 1
        lambdaREC = 1
        VPTV = 0.1
        VBLA = 1
        VREC = 1
        tPTV = 1
        tBLA = 1
        tREC = 1
        MPTV, MBLA, MREC = ProcessDmat(doseMatrix_test, targetLabels_test)
        xVec = np.ones((MPTV.shape[1],))
        y = MPTV.dot(xVec)
        gamma = np.zeros((y.shape))
        #state_test, iter, xVec, gamma = \
            #runOpt1(MPTV, MBLA, MREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, tPTV, tBLA, tREC, xVec, gamma)
        xVec, iter, converge, gamma =MinimizeDoseOARnew1(MPTV, MBLA, MREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, tPTV, tBLA, tREC, xVec, gamma)
        state_test=DVHplot(MPTV, MBLA, MREC,xVec)
        Dose = doseMatrix_test.dot(xVec)
        DPTV = MPTV.dot(xVec)
        DBLA = MBLA.dot(xVec)
        DREC = MREC.dot(xVec)
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

        Y = zeros((1000, 3))
        Y[:, 0] = y_ptv
        Y[:, 1] = y_bladder
        Y[:, 2] = y_rectum

        X = zeros((1000, 3))
        X[:, 0] = x_ptv
        X[:, 1] = x_bladder
        X[:, 2] = x_rectum

        data_result_path='/data/data/Results/GPU2/general/'
        np.save(data_result_path +id+'xDVHYInitial',
                Y)
        np.save(data_result_path +id+'xDVHXInitial',
                X)
        np.save(data_result_path +id+'xVecInitial', xVec)
        np.save(data_result_path +id+'DoseInitial', Dose)


        lambdaBLA_all = zeros((MAX_STEP + 1))
        lambdaREC_all = zeros((MAX_STEP + 1))
        VPTV_all = zeros((MAX_STEP + 1))
        VBLA_all = zeros((MAX_STEP + 1))
        VREC_all = zeros((MAX_STEP + 1))
        tPTV_all = zeros((MAX_STEP + 1))
        tBLA_all = zeros((MAX_STEP + 1))
        tREC_all = zeros((MAX_STEP + 1))

        planScore_all = zeros((MAX_STEP + 1))
        planScore_fine_all = zeros((MAX_STEP + 1))

        planScore_fine, planScore = planIQ(MPTV, MBLA, MREC, xVec)

        lambdaBLA_all[0] = lambdaBLA
        lambdaREC_all[0] = lambdaREC
        VPTV_all[0] = VPTV
        VBLA_all[0] = VBLA
        VREC_all[0] = VREC
        tPTV_all[0] = tPTV
        tBLA_all[0] = tBLA
        tREC_all[0] = tREC
        # lambda6_all[0] = lambda6
        planScore_all[0] = planScore
        planScore_fine_all[0] = planScore_fine
        print("Iteration_num: {}  PlanScore: {}  PlanScore_fine: {}".format(iter, planScore, planScore_fine))

        for i in range(MAX_STEP):
            X = state_test
            value = np.zeros(8)
            value[0] = np.max(mainDQN1.predict(X))
            value[1] = np.max(mainDQN2.predict(X))
            value[2] = np.max(mainDQN3.predict(X))
            value[3] = np.max(mainDQN4.predict(X))
            value[4] = np.max(mainDQN5.predict(X))
            value[5] = np.max(mainDQN6.predict(X))
            value[6] = np.max(mainDQN7.predict(X))
            value[7] = np.max(mainDQN8.predict(X))
            paraidx = np.argmax(value)
            if paraidx == 0:
                action = np.argmax(mainDQN1.predict(X))
                if action == 0:
                    action_factor = 1.5
                    lambdaBLA = lambdaBLA * action_factor
                    if lambdaBLA >= paraMax:
                        lambdaBLA = paraMax


                elif action == 1:
                    action_factor = 1.00
                    lambdaBLA = lambdaBLA * action_factor


                else:
                    action_factor = 0.5
                    lambdaBLA = lambdaBLA * action_factor
                    if lambdaBLA <= paraMin:
                        lambdaBLA = paraMin

            if paraidx == 1:
                action = np.argmax(mainDQN2.predict(X))
                if action == 0:
                    action_factor = 1.5
                    lambdaREC = lambdaREC * action_factor
                    if lambdaREC >= paraMax:
                        lambdaREC = paraMax

                elif action == 1:
                    action_factor = 1.00
                    lambdaREC = lambdaREC * action_factor


                else:
                    action_factor = 0.5
                    lambdaREC = lambdaREC * action_factor
                    if lambdaREC <= paraMin:
                        lambdaREC = paraMin

            if paraidx == 2:
                action = np.argmax(mainDQN3.predict(X))
                if action == 0:
                    action_factor = 1.5
                    VPTV = VPTV * action_factor
                    if VPTV >= paraMax:
                        VPTV = paraMax


                elif action == 1:
                    action_factor = 1.00
                    VPTV = VPTV * action_factor


                else:
                    action_factor = 0.5
                    VPTV = VPTV * action_factor
                    if VPTV <= paraMin:
                        VPTV = paraMin

            if paraidx == 3:
                action = np.argmax(mainDQN4.predict(X))
                if action == 0:
                    action_factor = 1.5
                    VBLA = VBLA * action_factor
                    if VBLA >= paraMax:
                        VBLA = paraMax


                elif action == 1:
                    action_factor = 1.00
                    VBLA = VBLA * action_factor


                else:
                    action_factor = 0.5
                    VBLA = VBLA * action_factor
                    if VBLA <= paraMin:
                        VBLA = paraMin

            if paraidx == 4:
                action = np.argmax(mainDQN5.predict(X))
                if action == 0:
                    action_factor = 1.5
                    VREC = VREC * action_factor
                    if VREC >= paraMax:
                        VREC = paraMax

                elif action == 1:
                    action_factor = 1.00
                    VREC = VREC * action_factor


                else:
                    action_factor = 0.5
                    VREC = VREC * action_factor
                    if VREC <= paraMin:
                        VREC = paraMin

            if paraidx == 5:
                action = np.argmax(mainDQN6.predict(X))
                if action == 0:
                    action_factor = 1.05
                    tPTV = tPTV * action_factor
                    if tPTV >= 1.5:
                        tPTV = 1.5


                elif action == 1:
                    action_factor = 1.00
                    tPTV = tPTV * action_factor



                else:
                    action_factor = 0.95
                    tPTV = tPTV * action_factor
                    if tPTV <= 1:
                        tPTV = 1

            if paraidx == 6:
                action = np.argmax(mainDQN7.predict(X))
                if action == 0:
                    action_factor = 1.5
                    tBLA = tBLA * action_factor
                    if tBLA >= paraMax:
                        tBLA = paraMax


                elif action == 1:
                    action_factor = 1.00
                    tBLA = tBLA * action_factor


                else:
                    action_factor = 0.5
                    tBLA = tBLA * action_factor
                    if tBLA <= paraMin:
                        tBLA = paraMin

            if paraidx == 7:
                action = np.argmax(mainDQN8.predict(X))
                if action == 0:
                    action_factor = 1.5
                    tREC = tREC * action_factor
                    if tREC >= paraMax:
                        tREC = paraMax


                elif action == 1:
                    action_factor = 1.00
                    tREC = tREC * action_factor


                else:
                    action_factor = 0.5
                    tREC = tREC * action_factor
                    if tREC <= paraMin:
                        tREC = paraMin
            if action !=1:
                xVec, iter, converge, gamma =MinimizeDoseOARnew1(MPTV, MBLA, MREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, tPTV, tBLA, tREC, xVec, gamma)
                state_test=DVHplot(MPTV, MBLA, MREC,xVec)
            #if action != 1:
              #  state_test, iter, xVec, gamma = runOpt1(MPTV, MBLA, MREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, tPTV, tBLA, tREC, xVec, gamma)

            planScore_fine,  planScore = planIQ(MPTV, MBLA, MREC, xVec)

            lambdaBLA_all[i+1] = lambdaBLA
            lambdaREC_all[i+1] = lambdaREC
            VPTV_all[i+1] = VPTV
            VBLA_all[i+1] = VBLA
            VREC_all[i+1] = VREC
            tPTV_all[i+1] = tPTV
            tBLA_all[i+1] = tBLA
            tREC_all[i+1] = tREC
            planScore_all[i + 1] = planScore
            planScore_fine_all[i + 1] = planScore_fine

            if paraidx == 0:
                print("Step: {}  Iteration: {}  Action: {}  LambdaBLA: {}  PlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter, action,
                                                                                                lambdaBLA, planScore_fine, planScore))
            if paraidx == 1:
                print("Step: {}  Iteration: {}  Action: {}  lambdaREC: {}  PlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter, action,
                                                                                                lambdaREC, planScore_fine, planScore))
            if paraidx == 2:
                print("Step: {}  Iteration: {}  Action: {}  VPTV: {}  PlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter, action,
                                                                                                VPTV, planScore_fine, planScore))
            if paraidx == 3:
                print(
                    "Step: {}  Iteration: {}  Action: {}  VBLA: {}  PlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter, action, VBLA,
                                                                                                                  planScore_fine,
                                                                                                                  planScore))

            if paraidx == 4:
                print("Step: {}  Iteration: {}  Action: {}  VREC: {}  PlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter, action,
                                                                                                VREC, planScore_fine, planScore))

            if paraidx == 5:
                print("Step: {}  Iteration: {}  Action: {}  tPTV: {}  PlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter, action,
                                                                                                tPTV, planScore_fine, planScore))

            if paraidx == 6:
                print("Step: {}  Iteration: {}  Action: {}  tBLA: {}  PlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter, action,
                                                                                                tBLA, planScore_fine, planScore))

            if paraidx == 7:
                print("Step: {}  Iteration: {}  Action: {}  tREC: {}  PlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter, action,
                                                                                                tREC, planScore_fine, planScore))
            # if paraidx == 5:
            #     print("Step: {}  Iteration: {}  Action: {}  Lambda6: {}  PlanScore: {} ".format(i + 1, iter, action,
            #                                                                                     lambda6, planScore))
            data_result_path='/data/data/Results/GPU2/general/'
            Dose = doseMatrix_test.dot(xVec)
            np.save(data_result_path +id+'xVec' + str(episode) + 'step' + str(i + 1), xVec)
            np.save(data_result_path +id+'xDose' + str(episode) + 'step' + str(i + 1), Dose)

            DPTV = MPTV.dot(xVec)
            DBLA = MBLA.dot(xVec)
            DREC = MREC.dot(xVec)
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

            Y = zeros((1000, 3))
            Y[:, 0] = y_ptv
            Y[:, 1] = y_bladder
            Y[:, 2] = y_rectum

            X = zeros((1000, 3))
            X[:, 0] = x_ptv
            X[:, 1] = x_bladder
            X[:, 2] = x_rectum

            np.save(data_result_path +id+'08xDVHY' + str(episode) + 'step' + str(i + 1),
                    Y)
            np.save(data_result_path +id+'xDVHX' + str(episode) + 'step' + str(i + 1),
                    X)
            data_result_path2='/data/data/Results/GPU2/figures/'
            plt.plot(x_ptv, y_ptv)
            plt.plot(x_bladder, y_bladder)
            plt.plot(x_rectum, y_rectum)
            plt.legend(('ptv','bladder','rectum'))
            plt.show(block=False)
            plt.title('DVH'+str(episode)+'step'+str(i+1))
            plt.savefig(data_result_path2+id+'DVH'+str(episode)+'step'+str(i+1)+'.png')
            plt.close()
        np.save(
            data_result_path +id+'lambdaBLA' + str(episode),
            lambdaBLA_all)
        np.save(
            data_result_path +id+'lambdaREC' + str(episode),
            lambdaREC_all)
        np.save(
            data_result_path +id+'VPTV' + str(episode),
            VPTV_all)
        np.save(
            data_result_path +id+'VBLA' + str(episode),
            VBLA_all)
        np.save(
            data_result_path +id+'VREC' + str(episode),
            VREC_all)
        np.save(
            data_result_path + id + 'tPTV' + str(episode),
            tPTV_all)
        np.save(
            data_result_path + id + 'tBLA' + str(episode),
            tBLA_all)
        np.save(
            data_result_path + id + 'tREC' + str(episode),
            tREC_all)
        np.save(
            data_result_path +id+'planScore' + str(episode),
            planScore_all)
        np.save(
            data_result_path +id+'planScore_fine' + str(episode),
            planScore_fine_all)
    #if 8 in planScore_all:
        plt.plot(planScore_all)
        plt.plot(planScore_fine_all)
        plt.legend(('planScore','planScoreFine'))
        plt.show(block=False)
        plt.savefig(data_result_path2+id+'planscore'+str(episode)+'.png')
        plt.close()


def Conjgrad(A, b, x0, maxiter1):
	x = x0
	# print(x)
	temp1 = np.matmul(A, x)
	r = b - temp1
	p = r
	rsold = np.matmul(r.transpose(), r)

	if rsold < 1e-5:
		return

	for i in range(maxiter1):
		Ap = np.matmul(A, p)

		pAp = np.matmul(p.transpose(), Ap)
		alpha = rsold / pAp
		x = x + alpha * p

		for j in range(A.shape[1]):
			if x[j] < 0:
				x[j] = 0
		r = r - alpha * Ap
		rsnew = np.matmul(r.transpose(), r)
		if sqrt(rsnew) < 1e-5:
			break
		p = r + (rsnew / rsold) * p
		rsold = rsnew
	# print(i)
	return x

def DVHplot(MPTV, MBLA, MREC,xVec):

    DPTV = MPTV.dot(xVec)
    DBLA = MBLA.dot(xVec)
    DREC = MREC.dot(xVec)
    DPTV = np.sort(DPTV)
    DPTV = np.flipud(DPTV)
    DBLA = np.sort(DBLA)
    DBLA = np.flipud(DBLA)
    DREC = np.sort(DREC)
    DREC = np.flipud(DREC)

    ## Plot DVH curve for optimized plan
    edge_ptv = np.zeros((INPUT_SIZE+1,))
    edge_ptv[1:INPUT_SIZE+1] = np.linspace(pdose,pdose*1.15, INPUT_SIZE)
    (n_ptv, b) = np.histogram(DPTV, bins=edge_ptv)
    y_ptv = 1 - np.cumsum(n_ptv / len(DPTV), axis=0)

    edge_bladder = np.zeros((INPUT_SIZE+1,))
    edge_bladder[1:INPUT_SIZE+1] = np.linspace(0.6*pdose, 1.1*pdose, INPUT_SIZE)
    (n_bladder, b) = np.histogram(DBLA, bins=edge_bladder)
    y_bladder = 1 - np.cumsum(n_bladder / len(DBLA), axis=0)

    edge_rectum = np.zeros((INPUT_SIZE+1,))
    edge_rectum[1:INPUT_SIZE+1] = np.linspace(0.6*pdose, 1.1*pdose, INPUT_SIZE)
    (n_rectum, b) = np.histogram(DREC, bins=edge_rectum)
    y_rectum = 1 - np.cumsum(n_rectum / len(DREC), axis=0)

    Y = np.zeros((INPUT_SIZE,3))
    Y[:, 0] = y_ptv
    Y[:, 1] = y_bladder
    Y[:, 2] = y_rectum

    Y = np.reshape(Y,(100*3,),order = 'F')


    return Y


def MinimizeDoseOARnew1(MPTV, MBLA, MREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, tPTV, tBLA, tREC, xVec, gamma):
    beta=1
    #the current MPTV is stored in csc format, so here we transfer it to csr format. The csc format will be used as MPTVT for dot product in cuda later
    MPTVcsr=csr_matrix(MPTV)
    MBLAcsr=csr_matrix(MBLA)
    MRECcsr=csr_matrix(MREC)
    #test
    #print(MPTVcsr.data.shape[0])
    #for i in range(10):
    #    print("MPTV.indices, MPTV.indptr",MPTV.indices[i],MPTV.indptr[i],MPTVcsr.indices[i],MPTVcsr.indptr[i])
    #print(MPTVcsr.indptr.shape[0])
    print(MPTVcsr.shape[0])
    #print(MPTVcsr.indptr[0],MPTVcsr.indptr[1])
    print(MPTVcsr.dtype)

    mod=SourceModule("""
    #include <thrust/sort.h>
    #include <thrust/binary_search.h>
    #include <thrust/device_vector.h>
    #include <thrust/execution_policy.h>

    struct sparseStru{
        int nrows;
        float *data;
        int *indices, *indptr, *nrowS;
    };
    struct xVecAll{
 //       float *new,*old,*xVecPTV1,*xVecPTV2, *xVecBLA, *xVecREC, *xVecR, *xVecP, *xVecAP;
       int nrows;
    };
    struct dosePTV{
        float *d, *y, *dy_sorted, *gamma;
        int *dy_indices, posi;
    };
    struct doseOAR{
        float *d;
        int *d_indices, posi;
    };
    struct para{
        float lamPTV,lamBLA,lamREC,vPTV,vBLA,vREC,tPTV,tBLA,tREC;
    };
    extern "C"{
    __global__ void dotsortDose(float *data,int *indices, int *indptr, int nrows,float *xVec,float *d,int flag, float *d_sorted)
    {
    /****** d=M.dot(xVec) and d_sorted=sort(d)with d_indices ordered accordingly ****/
    /** flag =0, do the sorting; flag=1, no sorting; flag=2, only sorting. **/
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < nrows)
        {
            const int row_start = indptr[row];
            const int row_end = indptr[row + 1];
            //test struct copy to GPU
           /* if (row==0)
            {
                printf("%d, %d \\n", row_start, row_end);
            }*/
            float sum =0;
            //int i=0;
            for (int element = row_start; element < row_end; element++)
            {
                sum += data[element] * xVec[indices[element]];
                // test if data correctly copyed to GPU
                /*if (row==0 & i<5)
                {
                    printf("position=%d, M_data=%f, M_indptr=%d \\n", element,data[element],indptr[element]);
                    i+=1;
                }*/
            }
            d[row] = sum;
            if (flag==0)
            {
                d_sorted[row]=sum;
            }
        }
        __syncthreads();

        if (row== 0)
        {
            if (flag==0)
            {
               // thrust::stable_sort_by_key(thrust::device, d_sorted, d_sorted+nrows,d_indices);
                thrust::stable_sort(thrust::device, d_sorted, d_sorted+nrows);
              //  printf("d_sorted_0=%f, d_indices=%d, flag=%d\\n", d_sorted[0],d_indices[0], flag);
            }
         /*   else if (flag==1)
            {
                for (int i=0;i<1;i++)
                    printf("hello=====================, no sort, d=%f, ind=%d,flag=%d\\n", d[i],i,flag);
            }*/
        }
        __syncthreads();
    }
    __global__ void scaleDose(int nrows, float *xVec, int posi, float *d_sorted,float pdose, int flag, int nrows2, float *d, float *y,float *xVec_old)
    {
    /** scale the dose to D95=1, update xVec. If flag=1, update d, d_sorted, y as well***/
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        __shared__ float factor;
        if (threadIdx.x==0)
            factor=pdose/d_sorted[posi];
        __syncthreads();
        if (row<nrows)
        {
            xVec[row]*=factor;
            xVec_old[row]=xVec[row];
        }

        //if (row==0)
          //  printf("before scale: d_sorted_0=%f, d_0=%f, d_95=%f, scaling_facotr=%f, flag=%d\\n",d_sorted[0], d[0], d_sorted[posi], factor,flag);
        if (flag==1)
        {
            if (row<nrows2)
            {
                d[row]*=factor;
                d_sorted[row]*=factor;
                y[row]=d[row];
            }
            __syncthreads();
            /*if (row==20)
            {
                printf("after scale:factor=%f, d_lens=%d, posi=%d, d_sorted_0=%f, d_0=%f, d_95=%f, flag=%d\\n",factor,nrows2, posi, d_sorted[0], d[0], d_sorted[posi], flag);
                printf("hello=================\\n");
            }*/
        }
        __syncthreads();

    }
    __global__ void nrowOverDose(int nrows, int *d_indices, int *rowShort, float *d,float posD)
    {
    /** to get how many rows are over given dose criteria. Total count will be returned, with corresponding row indices be sorted from small to large and returned. **/
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row<nrows)
        {
            if (posD<d[row])
            {
              //  printf("row, %d\\n", row);

               d_indices[row]=1;
               atomicAdd(rowShort, 1);
            }
            else
                d_indices[row]=0;
        }
        __syncthreads();
       /* if (row==0)
        {
            printf("total rows =%d, rows exceeding dose criteria=%d\\n", nrows,*nrowS);
            for (int i=0;i<*nrowS&i<100;i++)
            {
                if (d_indices[i]!=0)
                     printf("d_indices_over=%d,i=%d\\n", d_indices[i],i);
            }
        }*/

    }

    __global__ void overDose(float *r,float *r2,int flag, int nrows,int ncols, int *ncol2,int *d_indices,float *data,int *indices, int *indptr,float *y, float *d,float lam,float t)
    {
    /** to obtain the total over dose. Flag=1, r=r+MPTV_overT.dot(pdose*t-D_over)*lam/nrows;flag=2, r=r+MPTVT.dot(y-d)*lam/ncols; flag=0, temp=MPTV_over*p_over**/
        int row = blockIdx.x * blockDim.x + threadIdx.x;
       // if (row==0)
         //   printf("flag=%d,ncols=%d, ncol2=%d,d_indices_start=%d, d_indices_end=%d,lam=%f\\n", flag,ncols, *ncol2, d_indices[ncols-(*ncol2)],d_indices[ncols-1],lam);

        if (flag==0)
        {
            if (row<nrows)
            {
                if (d_indices[row]>0)
                {
                    const int row_start=indptr[row];
                    const int row_end=indptr[row+1];
                    float sum=0;
                    for (int element = row_start; element < row_end; element++)
                        sum += data[element] * y[indices[element]];
                    r[row]=r2[row]+sum*lam;
                 }

                else
                    r[row]=0;
            }
        }
        else if (flag==1|flag==2)
        {
            if (row<nrows)
            {
                const int row_start = indptr[row];
                const int row_end = indptr[row + 1];
                float sum =0;
                int i=0;
                for (int element = row_start; element < row_end; element++)
                {
                    if (flag==1)
                    {
                       if(d_indices[indices[element]]>0)
                       {
                            sum += data[element] * (t*y[indices[element]]-d[indices[element]]);

       /*                     if (row==0&i<5)
                            {
                                printf("element=%d,indices[element]=%d,data=%f,y=%f,d=%f \\n", element,indices[element],data[element],t*y[indices[element]],d[indices[element]]);
                                i+=1;
                            }*/
                        }
                    }
                    else if (flag==2)
                    {
                        sum += data[element] * (y[indices[element]]-d[indices[element]]);
                    }
                }
                if (flag==1)
                    r[row]=r2[row]+sum*lam/(*ncol2);
                else if (flag==2)
                    r[row]=r2[row]+sum*lam/ncols;
            }
        }
        __syncthreads();
    }

    __global__ void adding(int nrows, float *x, float *y, float *z, float t,int flag,float *xx)
    {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
       // if (row==0)
      //      printf("x=%f,y=%f,z=%f,t=%f,flag=%d\\n",x[0],y[0],z[0],t,flag);
        if (row<nrows)
        {
            x[row]=y[row]+t*z[row];
            if (flag==1)
                x[row]=x[row]>=0?x[row]:0;
            if (flag==2)
            {
                xx[row]=x[row];
            }
        }
        __syncthreads();

        if (flag == 2)
        {
            if (row==0)
            {
                thrust::stable_sort(thrust::device, xx, xx+nrows);
            }
        }
       /* if (row==0)
        {
            printf("after:x=%f,y=%f,z=%f,t=%f,flag=%d\\n",x[0],y[0],z[0],t,flag);
        }*/
        __syncthreads();
    }
    __global__ void copying(int nrows, float *x, float *y)
    {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row<nrows)
        {
            x[row]=y[row];
        }
        __syncthreads();
    }
    __global__ void ydose(int nrows, float *y, float d1,float d2)
    {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
       // if (row==0)
         //   printf("nrows=%d,nrow2=%d, ind[0]=%d,d_indices_start=%d, d_indices_end=%d,d1=%f,d2=%f\\n", nrows,nrow2, ind[0], ind[nrow2],ind[nrows-1],d1,d2);
        if (row<nrows)
        {
            if (y[row]>=d1 &y[row]<d2)
                y[row]=d2;
        }
        __syncthreads();
    }

    }
        """,no_extern_c=True)

 ## transfer data from python to GPU with arrays and structures
    class sparseStru:
        def __init__(self,array):
            self.data = gpuarray.to_gpu(array.data.astype(np.float32))
            self.indices = gpuarray.to_gpu(array.indices.astype(np.int32))
            self.indptr =gpuarray.to_gpu(array.indptr.astype(np.int32))
            self.nrows=np.int32(array.indptr.shape[0]-1) # this is to accomodate both csr and csc formats
    class xVecAll:
        def __init__(self,xVec):
            self.nrows=np.int32(xVec.shape)
            self.new = gpuarray.to_gpu(xVec.astype(np.float32))
            self.old = gpuarray.to_gpu(xVec.astype(np.float32))
            self.PTV1 =gpuarray.to_gpu(np.empty_like(xVec).astype(np.float32))
            self.PTV2 =gpuarray.to_gpu(np.empty_like(xVec).astype(np.float32))
            self.BLA = gpuarray.to_gpu(np.empty_like(xVec).astype(np.float32))
            self.REC =gpuarray.to_gpu(np.empty_like(xVec).astype(np.float32))
            self.R = gpuarray.to_gpu(np.empty_like(xVec).astype(np.float32))
            self.P =gpuarray.to_gpu(np.empty_like(xVec).astype(np.float32))
            self.AP =gpuarray.to_gpu(np.empty_like(xVec).astype(np.float32))
    class dosePTV:
        def __init__(self,gamma):
            self.d = gpuarray.to_gpu(np.empty_like(gamma).astype(np.float32))
            self.y =gpuarray.to_gpu(np.empty_like(gamma).astype(np.float32))
            self.dy_sorted =gpuarray.to_gpu(np.empty_like(gamma).astype(np.float32))
            self.dy_indices = gpuarray.to_gpu(np.empty_like(gamma).astype(np.int32))
            self.y_sorted =gpuarray.to_gpu(np.empty_like(gamma).astype(np.float32))
            self.y_indices = gpuarray.to_gpu(np.empty_like(gamma).astype(np.int32))
            self.gamma =gpuarray.to_gpu(gamma.astype(np.float32))
            self.posi=np.int32(np.round(0.05*np.float32(gamma.shape)))
    class doseOAR:
        def __init__(self,OAR):
            self.d = gpuarray.to_gpu(np.zeros(OAR.shape[0]).astype(np.float32))
            self.d_sorted = gpuarray.to_gpu(np.zeros(OAR.shape[0]).astype(np.float32))
            self.d_indices =gpuarray.to_gpu(np.zeros(OAR.shape[0]).astype(np.int32))
            self.posi=np.int32(np.round(0.05*np.float32(OAR.shape[0])))
    class para:
        def __init__(self,lambdaPTV,lambdaBLA,lambdaREC,VPTV,VBLA,VREC,tPTV,tBLA,tREC):
            self.lamPTV=np.float32(lambdaPTV)
            self.lamBLA=np.float32(lambdaBLA)
            self.lamREC=np.float32(lambdaREC)
            self.vPTV=np.float32(VPTV)
            self.vBLA=np.float32(VBLA)
            self.vREC=np.float32(VREC)
            self.tPTV=np.float32(tPTV)
            self.tBLA=np.float32(tBLA)
            self.tREC=np.float32(tREC)
            self.beta=np.float32(beta)

    # start here to initiate all GPU memories
    MPTV_gpu = sparseStru(MPTVcsr)
    MPTVT_gpu= sparseStru(MPTV)
    MBLA_gpu = sparseStru(MBLAcsr)
    MBLAT_gpu= sparseStru(MBLA)
    MREC_gpu = sparseStru(MRECcsr)
    MRECT_gpu= sparseStru(MREC)
    xVec_gpu=xVecAll(xVec)
    #test
    #for i in range(MPTVcsr.indptr[0],MPTVcsr.indptr[1]):
     #   print(MPTVcsr.data[i],xVec[MPTVcsr.indices[i]])
    PTV_gpu=dosePTV(gamma)
    BLA_gpu=doseOAR(MBLA)
    REC_gpu=doseOAR(MREC)

    #container to store temporaty dose
    num=math.ceil(max(MPTV.shape[0],MBLA.shape[0],MREC.shape[0]))
    tempD_gpu=gpuarray.to_gpu(np.zeros(num).astype(np.float32))
    tempDzero_gpu=gpuarray.to_gpu(np.zeros(num).astype(np.float32))
    pdoseA_gpu=gpuarray.to_gpu((np.ones(num)*pdose).astype(np.float32))
    #iter, converge pointers
    iter=0
    iter_gpu = np.int32(iter)
    converge=0
    converge_gpu = np.int32(converge)
    # all other constant parameters
    para_gpu=para(lambdaPTV,lambdaBLA,lambdaREC,VPTV,VBLA,VREC,tPTV,tBLA,tREC)
    pdose_gpu=np.float32(pdose)

    # initialize threads and blocks
    ThreadNum=int(1024)
    BlockNumPTV=math.ceil(MPTV.shape[0]/ThreadNum)
    BlockNumBLA=math.ceil(MBLA.shape[0]/ThreadNum)
    BlockNumREC=math.ceil(MREC.shape[0]/ThreadNum)

    #dose scaling
    BlockNum1=math.ceil(max(MPTV.shape[0],MPTV.shape[1])/ThreadNum)
    BlockNum0=math.ceil(MPTV.shape[1]/ThreadNum)
    # define a value to scale the over dose values, since the r is too small represented in float32 format
    Sca=1000;

    # load GPU kernels
    func_dose = mod.get_function("dotsortDose")
    func_sPTV=mod.get_function("scaleDose")
    func_nrowOver=mod.get_function("nrowOverDose")
    func_overDose=mod.get_function("overDose")
    func_adding=mod.get_function("adding")
    func_copying=mod.get_function("copying")
    func_ydose=mod.get_function("ydose")
    stream1=cuda.Stream()
    stream2=cuda.Stream()
    stream3=cuda.Stream()
    # run kernels, start the GPU computation
 #   print("nrows= ", MPTV_gpu.nrows,"ThreadNum= ", ThreadNum, "BlockNumPTV=", BlockNumPTV)
    func_dose(MPTV_gpu.data,MPTV_gpu.indices, MPTV_gpu.indptr, MPTV_gpu.nrows,xVec_gpu.new,PTV_gpu.d,np.int32(0),PTV_gpu.dy_sorted,block = (ThreadNum, 1, 1), grid=(BlockNumPTV, 1))
  #  print("PTVdose[0]=", PTV_gpu.d[0], "PTVdose_sorted[0]= ", PTV_gpu.dy_sorted[0])
    func_sPTV(xVec_gpu.nrows, xVec_gpu.new,PTV_gpu.posi,PTV_gpu.dy_sorted,np.float32(pdose),np.int32(1), MPTV_gpu.nrows,PTV_gpu.d,PTV_gpu.y,xVec_gpu.old,block = (ThreadNum, 1, 1), grid=(BlockNum1, 1))

    start_time = time.time()
    #start the iteration
    for iter in range(maxiter):
        print("iteration====================================================================",iter)
        func_copying(MPTVT_gpu.nrows,xVec_gpu.old,xVec_gpu.new,block =(ThreadNum,1,1),grid=(BlockNum0,1))
        # DBLA=MBLA.dot(xVec) DBLA_sorted=sort(DBLA)
        func_dose(MBLA_gpu.data,MBLA_gpu.indices, MBLA_gpu.indptr, MBLA_gpu.nrows,xVec_gpu.new,BLA_gpu.d,np.int32(0),BLA_gpu.d_sorted,block = (ThreadNum, 1, 1), grid=(BlockNumBLA, 1))
           # DBLA=MREC.dot(xVec) DREC_sorted=sort(DREC)
        func_dose(MREC_gpu.data,MREC_gpu.indices, MREC_gpu.indptr, MREC_gpu.nrows,xVec_gpu.new,REC_gpu.d,np.int32(0),REC_gpu.d_sorted,block = (ThreadNum, 1, 1), grid=(BlockNumREC, 1))
       # r=-MPTVT.dot(gamma)
        rowShort=gpuarray.to_gpu(np.array(0).astype(np.int32))
        func_overDose(xVec_gpu.R,tempDzero_gpu,np.int32(2),MPTVT_gpu.nrows,MPTV_gpu.nrows,rowShort,PTV_gpu.dy_indices,MPTVT_gpu.data,MPTVT_gpu.indices,MPTVT_gpu.indptr,PTV_gpu.gamma,tempDzero_gpu,np.float32(-Sca),np.float32(1),block=(ThreadNum,1,1), grid=(BlockNum0,1))
       # r=r+MPTVT.dot((y-DPTV)*beta/y.shape[0])
    #__global__ void overDose(float *r,int flag, int nrows,int ncols, int *ncol2,int *d_indices,float *data,int *indices, int *indptr,float *y, float *d,float lam,float t)
        func_overDose(xVec_gpu.R,xVec_gpu.R,np.int32(2),MPTVT_gpu.nrows,MPTV_gpu.nrows,rowShort,PTV_gpu.dy_indices,MPTVT_gpu.data,MPTVT_gpu.indices,MPTVT_gpu.indptr,PTV_gpu.y,PTV_gpu.d,np.float32(beta*Sca),np.float32(1),block=(ThreadNum,1,1), grid=(BlockNum0,1))
 #       print("PTV_dmax", PTV_gpu.dy_sorted[-1].get(),pdose*tPTV)
        if PTV_gpu.dy_sorted[-1].get()>pdose*tPTV:
            posi = int(round((1 - VPTV) * MPTV.shape[0]))-1
            if posi < 0:
                posi = 0
            DPTVV = max(PTV_gpu.dy_sorted[posi].get(), pdose * tPTV)
            rowShortP=gpuarray.to_gpu(np.array(0).astype(np.int32))
            func_nrowOver(MPTV_gpu.nrows,PTV_gpu.dy_indices,rowShortP,PTV_gpu.d,np.float32(DPTVV), block = (ThreadNum, 1, 1), grid=(BlockNumPTV, 1))

            print("rowShortP",rowShortP)
            #r=r+lambdaPTV*MPTV1T.dot(pdose*tPTV*np.ones((DPTVs.shape)) -DPTVs) / DPTVs.shape
            func_overDose(xVec_gpu.R,xVec_gpu.R,np.int32(1),MPTVT_gpu.nrows,MPTV_gpu.nrows,rowShortP,PTV_gpu.dy_indices,MPTVT_gpu.data, MPTVT_gpu.indices, MPTVT_gpu.indptr,pdoseA_gpu,PTV_gpu.d,np.float32(Sca*lambdaPTV),para_gpu.tPTV,block=(ThreadNum,1,1),grid=(BlockNum0,1))

        #  print("para",para_gpu.lamPTV,pdose_gpu,para_gpu.tPTV)

  #      print("BLA_dmax",BLA_gpu.d_sorted[-1].get(),pdose*tBLA)
        if BLA_gpu.d_sorted[-1].get()>pdose*tBLA:
            posi = int(round((1 - VBLA) * MBLA.shape[0]))-1
            if posi<0:
                posi=0
            DBLAV = max(BLA_gpu.d_sorted[posi].get(),pdose * tBLA)
            rowShortB=gpuarray.to_gpu(np.array(0).astype(np.int32))
            func_nrowOver(MBLA_gpu.nrows,BLA_gpu.d_indices,rowShortB, BLA_gpu.d,np.float32(DBLAV), block = (ThreadNum,1,1), grid=(BlockNumBLA, 1))
            print("rowShortB",rowShortB)
           # r = r+lambdaBLA * MBLA1T.dot(pdose*tBLA*np.ones((DBLAs.shape,))-DBLAs) / DBLAs.shape
            func_overDose(xVec_gpu.R,xVec_gpu.R,np.int32(1),MBLAT_gpu.nrows,MBLA_gpu.nrows,rowShortB,BLA_gpu.d_indices,MBLAT_gpu.data, MBLAT_gpu.indices, MBLAT_gpu.indptr, pdoseA_gpu,BLA_gpu.d,np.float32(lambdaBLA*Sca),para_gpu.tBLA,block=(ThreadNum,1,1),grid=(BlockNum0,1))

   #     print("REC_dmax",REC_gpu.d_sorted[-1].get(),pdose*tBLA)
        if REC_gpu.d_sorted[-1].get()>pdose*tREC:
            posi = int(round((1 - VREC) * MREC.shape[0]))-1
            if posi<0:
             posi=0
            DRECV = max(REC_gpu.d_sorted[posi].get(),pdose * tREC)
            rowShortR=gpuarray.to_gpu(np.array(0).astype(np.int32))
            func_nrowOver(MREC_gpu.nrows,REC_gpu.d_indices,rowShortR,REC_gpu.d,np.float32(DRECV),block = (ThreadNum, 1, 1), grid=(BlockNumREC, 1))

            print("rowShortR",rowShortR)
            #r = r+lambdaREC * MREC1T.dot(pdose*tREC*np.ones((DRECs.shape))-DRECs) / DRECs.shape
            func_overDose(xVec_gpu.R,xVec_gpu.R,np.int32(1),MRECT_gpu.nrows,MREC_gpu.nrows,rowShortR,REC_gpu.d_indices,MRECT_gpu.data,MRECT_gpu.indices,MRECT_gpu.indptr,pdoseA_gpu,REC_gpu.d,np.float32(lambdaREC*Sca),para_gpu.tREC,block = (ThreadNum,1,1), grid=(BlockNum0,1))

        func_copying(MPTVT_gpu.nrows,xVec_gpu.P,xVec_gpu.R,block =(ThreadNum,1,1),grid=(BlockNum0,1))
        xVecR=xVec_gpu.R.get()
        rsold=np.inner(xVecR,xVecR)/Sca/Sca
        #rsold=linalg.dot(xVec_gpu.R.get(), xVec_gpu.R.get())/Sca/Sca
    #    for i in range(5):
     #       print("xVec_gpu.R[i]=",xVec_gpu.R[i],rsold)# this contains the scale factor "Sca"
        print("rsold=",rsold, "iter=", iter, "=========================")# this not

        if rsold>1e-10:
            for iloop in range (3):
                #tempD=MPTV.dot(p)
                func_dose(MPTV_gpu.data,MPTV_gpu.indices, MPTV_gpu.indptr, MPTV_gpu.nrows,xVec_gpu.P,tempD_gpu,PTV_gpu.dy_indices,np.int32(1),PTV_gpu.dy_sorted,block = (ThreadNum, 1, 1), grid=(BlockNumPTV, 1))
                #Ap=beta*MPTVT.dot(tempD)/y.shape[0]
                func_overDose(xVec_gpu.AP,tempDzero_gpu,np.int32(2),MPTVT_gpu.nrows,MPTV_gpu.nrows,rowShort,PTV_gpu.dy_indices,MPTVT_gpu.data,MPTVT_gpu.indices,MPTVT_gpu.indptr,tempD_gpu,tempDzero_gpu,np.float32(beta),np.float32(1),block =(ThreadNum,1,1),grid=(BlockNum0,1))
                if PTV_gpu.dy_sorted[-1].get()>pdose*tPTV:
                 #   temp = MPTV1.dot(p)
        #            print("BlockNum_overDose",BlockNumPO)
                    func_overDose(tempD_gpu,tempDzero_gpu,np.int32(0),MPTV_gpu.nrows,MPTV_gpu.nrows,rowShortP,PTV_gpu.dy_indices,MPTV_gpu.data,MPTV_gpu.indices,MPTV_gpu.indptr,xVec_gpu.P,tempDzero_gpu,np.float32(1),np.float32(1),block =(ThreadNum,1,1),grid=(BlockNumPTV,1))
        #            Ap = Ap+MPTV1T.dot(temp)* lambdaPTV / MPTV1.shape[0]
                    func_overDose(xVec_gpu.AP,xVec_gpu.AP,np.int32(1),MPTVT_gpu.nrows,MPTV_gpu.nrows,rowShortP,PTV_gpu.dy_indices,MPTVT_gpu.data,MPTVT_gpu.indices,MPTVT_gpu.indptr,tempD_gpu,tempDzero_gpu,np.float32(lambdaPTV),np.float32(1),block =(ThreadNum,1,1),grid=(BlockNum0,1))

                if BLA_gpu.d_sorted[-1].get()>pdose*tBLA:
         #           print("start Ap for BLA")
         #           temp = MBLA1.dot(p)
          #          print("BlockNum_overDoseBLA",BlockNumBO)
                    func_overDose(tempD_gpu,tempDzero_gpu,np.int32(0),MBLA_gpu.nrows,MBLA_gpu.nrows,rowShortB,BLA_gpu.d_indices,MBLA_gpu.data,MBLA_gpu.indices,MBLA_gpu.indptr,xVec_gpu.P,tempDzero_gpu,np.float32(1),np.float32(1),block =(ThreadNum,1,1),grid=(BlockNumBLA,1))
          #          Ap = Ap+MBLA1T.dot(temp)* lambdaBLA / MBLA1.shape[0]
                    func_overDose(xVec_gpu.AP,xVec_gpu.AP,np.int32(1),MBLAT_gpu.nrows,MBLA_gpu.nrows,rowShortB,BLA_gpu.d_indices,MBLAT_gpu.data,MBLAT_gpu.indices,MBLAT_gpu.indptr,tempD_gpu,tempDzero_gpu,np.float32(lambdaBLA),np.float32(1),block =(ThreadNum,1,1),grid=(BlockNum0,1))

                if REC_gpu.d_sorted[-1].get()>pdose*tREC:
                #    print("start Ap for REC")
           #         temp = MREC1.dot(p)
                    func_overDose(tempD_gpu,tempDzero_gpu,np.int32(0),MREC_gpu.nrows,MREC_gpu.nrows,rowShortR,REC_gpu.d_indices,MREC_gpu.data,MREC_gpu.indices,MREC_gpu.indptr,xVec_gpu.P,tempDzero_gpu,np.float32(1),np.float32(1),block =(ThreadNum,1,1),grid=(BlockNumREC,1))
                    #Ap = Ap+MREC1T.dot(temp)* lambdaREC / MREC1.shape[0]
                    func_overDose(xVec_gpu.AP,xVec_gpu.AP,np.int32(1),MRECT_gpu.nrows,MREC_gpu.nrows,rowShortR,REC_gpu.d_indices,MRECT_gpu.data,MRECT_gpu.indices,MRECT_gpu.indptr,tempD_gpu,tempDzero_gpu,np.float32(lambdaREC),np.float32(1),block =(ThreadNum,1,1),grid=(BlockNum0,1))
                P=xVec_gpu.P.get()
                AP=xVec_gpu.AP.get()
                pAp=np.inner(P,AP)/Sca/Sca
                #pAp = linalg.dot(xVec_gpu.P, xVec_gpu.AP)/Sca/Sca
      #          for i in range(5):
       #             print("xVec_gpu.AP[i]=",xVec_gpu.AP[i])
                #pAp = linalg.dot(xVec_gpu.P, xVec_gpu.AP)/Sca/Sca
                alpha = rsold / pAp
               # xVec = xVec + alpha * p
               # xVec[xVec<0]=0
                func_adding(MPTVT_gpu.nrows,xVec_gpu.new,xVec_gpu.new,xVec_gpu.P,np.float32(alpha/Sca),np.int32(1),block =(ThreadNum,1,1),grid=(BlockNum0,1))
               # r = r - alpha * Ap
                func_adding(MPTVT_gpu.nrows,xVec_gpu.R,xVec_gpu.R,xVec_gpu.AP,np.float32(-alpha),np.int32(0),block =(ThreadNum,1,1),grid=(BlockNum0,1))
                xVecR=xVec_gpu.R.get()
                rsnew=np.inner(xVecR,xVecR)/Sca/Sca
                #rsnew=linalg.dot(xVec_gpu.R, xVec_gpu.R)/Sca/Sca
                print("rnew=",rsnew,"iloop",iloop)
                if math.sqrt(rsnew) < 1e-5:
                    break
                #p = r + (rsnew / rsold) * p
                func_adding(MPTVT_gpu.nrows,xVec_gpu.P,xVec_gpu.R,xVec_gpu.P,np.float32(rsnew/rsold),np.int32(0),block =(ThreadNum,1,1),grid=(BlockNum0,1))
                rsold = rsnew

        func_dose(MPTV_gpu.data,MPTV_gpu.indices, MPTV_gpu.indptr, MPTV_gpu.nrows,xVec_gpu.new,PTV_gpu.d,np.int32(0),PTV_gpu.dy_sorted,block = (ThreadNum, 1, 1), grid=(BlockNumPTV, 1))
        func_adding(MPTV_gpu.nrows,PTV_gpu.y,PTV_gpu.d,PTV_gpu.gamma,np.float32(MPTV.shape[0]/beta),np.int32(2),PTV_gpu.y_sorted,block =(ThreadNum,1,1),grid=(BlockNumPTV,1))
        posi=int(0.05*float(MPTV.shape[0]))
        D95=PTV_gpu.y_sorted[posi].get()
        print("posi",posi,"D95_y",D95,"nrows",MPTV_gpu.nrows)
        if D95<pdose:
             #y[(y>=D95)&(y<pdose)]=pdose
            func_ydose(MPTV_gpu.nrows, PTV_gpu.y,np.float32(D95),np.float32(pdose),block =(ThreadNum,1,1),grid=(BlockNumPTV,1))
        func_adding(MPTV_gpu.nrows,PTV_gpu.gamma,PTV_gpu.gamma,PTV_gpu.d,np.float32(beta/MPTV.shape[0]),np.int32(0),block =(ThreadNum,1,1),grid=(BlockNumPTV,1))
        func_adding(MPTV_gpu.nrows,PTV_gpu.gamma,PTV_gpu.gamma,PTV_gpu.y,np.float32(-beta/MPTV.shape[0]),np.int32(0),block =(ThreadNum,1,1),grid=(BlockNumPTV,1))

        xVec=xVec_gpu.new.get()
        xVec_old=xVec_gpu.old.get()
        print( "LA.norm(xVec- xVec_old)", LA.norm(xVec- xVec_old),"LA.norm(xVec)",LA.norm(xVec),"LA.norm(xVec_old)",LA.norm(xVec_old))
        if LA.norm(xVec- xVec_old)/LA.norm(xVec)<5e-3:
            break;

    func_dose(MPTV_gpu.data,MPTV_gpu.indices, MPTV_gpu.indptr, MPTV_gpu.nrows,xVec_gpu.new,PTV_gpu.d,PTV_gpu.dy_indices,np.int32(0),PTV_gpu.dy_sorted,block = (ThreadNum, 1, 1), grid=(BlockNumPTV, 1))
    func_sPTV(xVec_gpu.nrows, xVec_gpu.new,PTV_gpu.posi,PTV_gpu.dy_sorted,np.float32(pdose),np.int32(0), MPTV_gpu.nrows,PTV_gpu.d,PTV_gpu.y,xVec_gpu.old,block = (ThreadNum, 1, 1), grid=(BlockNum1, 1))
    converge=1
    if iter == maxiter - 1:
        converge = 0
    print("optimization time: {} seconds for {} iterrations".format((time.time() - start_time), iter))

    #copy back to CPU
    xVec=xVec_gpu.new.get()
    gamma=PTV_gpu.gamma.get()

    return xVec, iter, converge, gamma


def loadDoseMatrix(filename):
    test = h5sparse.File(filename,'r')
    Dmat = test['Dij']['000'].value
    temp = test['Dij']['032'].value
    Dmat = vstack([Dmat, temp])
    temp = test['Dij']['064'].value
    Dmat = vstack([Dmat, temp])
    temp = test['Dij']['096'].value
    Dmat = vstack([Dmat, temp])
    temp = test['Dij']['296'].value
    Dmat = vstack([Dmat, temp])
    temp = test['Dij']['264'].value
    Dmat = vstack([Dmat, temp])
    temp = test['Dij']['328'].value
    Dmat = vstack([Dmat, temp])
    Dmat = Dmat.transpose()
    return Dmat

def loadMask(filename):
    mask = h5py.File(filename,'r')
    dosemask = mask['oar_ptvs']['dose']
    dosemask = np.reshape(dosemask, (dosemask.shape[0] * dosemask.shape[1] * dosemask.shape[2],), order='C')
    PTVtemp = mask['oar_ptvs']['ptv']
    PTVtemp = np.reshape(PTVtemp, (PTVtemp.shape[0] * PTVtemp.shape[1] * PTVtemp.shape[2],), order='C')
    PTV = PTVtemp[np.nonzero(dosemask)]
    bladdertemp = mask['oar_ptvs']['bladder']
    bladdertemp = np.reshape(bladdertemp, (bladdertemp.shape[0] * bladdertemp.shape[1] * bladdertemp.shape[2],), order='C')
    bladder = bladdertemp[np.nonzero(dosemask)]
    rectumtemp = mask['oar_ptvs']['rectum']
    rectumtemp = np.reshape(rectumtemp, (rectumtemp.shape[0] * rectumtemp.shape[1] * rectumtemp.shape[2],), order='C')
    rectum = rectumtemp[np.nonzero(dosemask)]
    targetLabelFinal = np.zeros((PTV.shape))
    targetLabelFinal[np.nonzero(PTV)] = 1
    targetLabelFinal[np.nonzero(bladder)] = 2
    targetLabelFinal[np.nonzero(rectum)] = 3
    return targetLabelFinal


def main():
    id='07'
    data_path='/data/data/dose_deposition/prostate_dijs/f_dijs/0'
    data_path2='/data/data/dose_deposition3/plostate_dijs/f_masks/0'
    doseMatrix_1 = loadDoseMatrix(data_path +id+'.hdf5')
    targetLabels_1 = loadMask(data_path2 +id+'.h5')
    print(doseMatrix_1.shape)
    print(targetLabels_1.shape)

    id='08'
    doseMatrix_2 = loadDoseMatrix(data_path +id+'.hdf5')
    targetLabels_2 = loadMask(data_path2 +id+'.h5')
    print(doseMatrix_2.shape)
    print(targetLabels_2.shape)

    id='09'
    doseMatrix_3 = loadDoseMatrix(data_path +id+'.hdf5')
    targetLabels_3 = loadMask(data_path2 +id+'.h5')
    print(doseMatrix_3.shape)
    print(targetLabels_3.shape)

    id='10'
    doseMatrix_4 = loadDoseMatrix(data_path +id+'.hdf5')
    targetLabels_4 = loadMask(data_path2 +id+'.h5')
    print(doseMatrix_4.shape)
    print(targetLabels_4.shape)

    id='11'
    doseMatrix_5 = loadDoseMatrix(data_path +id+'.hdf5')
    targetLabels_5 = loadMask(data_path2 +id+'.h5')
    print(doseMatrix_5.shape)
    print(targetLabels_5.shape)

    id='12'
    doseMatrix_6 = loadDoseMatrix(data_path +id+'.hdf5')
    targetLabels_6 = loadMask(data_path2 +id+'.h5')
    print(doseMatrix_6.shape)
    print(targetLabels_6.shape)

    id='13'
    doseMatrix_7 = loadDoseMatrix(data_path +id+'.hdf5')
    targetLabels_7 = loadMask(data_path2 +id+'.h5')
    print(doseMatrix_7.shape)
    print(targetLabels_7.shape)

    id='14'
    doseMatrix_8 = loadDoseMatrix(data_path +id+'.hdf5')
    targetLabels_8 = loadMask(data_path2 +id+'.h5')
    print(doseMatrix_8.shape)
    print(targetLabels_8.shape)

    id='15'
    doseMatrix_9 = loadDoseMatrix(data_path +id+'.hdf5')
    targetLabels_9 = loadMask(data_path2 +id+'.h5')
    print(doseMatrix_9.shape)
    print(targetLabels_9.shape)

    id='16'
    doseMatrix_0 = loadDoseMatrix(data_path +id+'.hdf5')
    targetLabels_0 = loadMask(data_path2 +id+'.h5')
    print(doseMatrix_0.shape)
    print(targetLabels_0.shape)



    session_folder='/data/data/Results/GPU2/session/'
    save_session_name = session_folder +'dqn_dvh_external_fine_new.ckpt'
    session_folder2='/data/data//Results/GPU2/session/'
    session_load_name = session_folder2 +'dqn_dvh_external_fine_new.ckpt-'

    # store the previous observations in replay memory
    replay_buffer1= deque(maxlen=REPLAY_MEMORY)
    replay_buffer2 = deque(maxlen=REPLAY_MEMORY)
    replay_buffer3 = deque(maxlen=REPLAY_MEMORY)
    replay_buffer4 = deque(maxlen=REPLAY_MEMORY)
    replay_buffer5 = deque(maxlen=REPLAY_MEMORY)
    replay_buffer6 = deque(maxlen=REPLAY_MEMORY)
    replay_buffer7 = deque(maxlen=REPLAY_MEMORY)
    replay_buffer8 = deque(maxlen=REPLAY_MEMORY)
    with tf.device('/device:GPU:2'):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True ,gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            mainDQN1 = dqn_dvh_external_network.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main1")
            targetDQN1 = dqn_dvh_external_network.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="target1")
            mainDQN2 = dqn_dvh_external_network.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main2")
            targetDQN2 = dqn_dvh_external_network.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="target2")
            mainDQN3 = dqn_dvh_external_network.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main3")
            targetDQN3 = dqn_dvh_external_network.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="target3")
            mainDQN4 = dqn_dvh_external_network.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main4")
            targetDQN4 = dqn_dvh_external_network.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="target4")
            mainDQN5 = dqn_dvh_external_network.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main5")
            targetDQN5 = dqn_dvh_external_network.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="target5")
            mainDQN6 = dqn_dvh_external_network.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main6")
            targetDQN6 = dqn_dvh_external_network.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="target6")
            mainDQN7 = dqn_dvh_external_network.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main7")
            targetDQN7 = dqn_dvh_external_network.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="target7")
            mainDQN8 = dqn_dvh_external_network.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main8")
            targetDQN8 = dqn_dvh_external_network.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="target8")

            sess.run(tf.global_variables_initializer())

            copy_ops1 = get_copy_var_ops(dest_scope_name="target1",
                                        src_scope_name="main1")

            copy_ops2 = get_copy_var_ops(dest_scope_name="target2",
                                        src_scope_name="main2")

            copy_ops3 = get_copy_var_ops(dest_scope_name="target3",
                                        src_scope_name="main3")

            copy_ops4 = get_copy_var_ops(dest_scope_name="target4",
                                        src_scope_name="main4")

            copy_ops5 = get_copy_var_ops(dest_scope_name="target5",
                                        src_scope_name="main5")

            copy_ops6 = get_copy_var_ops(dest_scope_name="target6",
                                        src_scope_name="main6")

            copy_ops7 = get_copy_var_ops(dest_scope_name="target7",
                                        src_scope_name="main7")

            copy_ops8 = get_copy_var_ops(dest_scope_name="target8",
                                        src_scope_name="main8")

            sess.run(copy_ops1)

            sess.run(copy_ops2)

            sess.run(copy_ops3)

            sess.run(copy_ops4)

            sess.run(copy_ops5)

            sess.run(copy_ops6)

            sess.run(copy_ops7)

            sess.run(copy_ops8)

            saver = tf.train.Saver(max_to_keep=40)
            if load_session == 1:
                saver.restore(sess, session_load_name+str(LoadEpoch))
            if Start == 1:
                total_step=0
                reward_check = zeros((MAX_EPISODES))
                q_check = zeros((MAX_EPISODES))
                step_count1 = 0
                step_count2 = 0
                step_count3 = 0
                step_count4 = 0
                step_count5 = 0
                step_count6 = 0
                step_count7 = 0
                step_count8 = 0
                for episode in range(MAX_EPISODES):
                    reward_sum_total = 0
                    qvalue_sum = 0
                    num_q = 0
                    if load_session!=0:
                        e = 0.999 / ((episode / 500) + 1)
                    else:
                        e = 0.999 / ((episode / 500) + 1)
                    if e < 0.1:
                        e = 0.1
                    for testcase in range(TRAIN_NUM):
                        if testcase % TRAIN_NUM  == 0:
                            doseMatrix = doseMatrix_1
                            targetLabels= targetLabels_1
                        if testcase % TRAIN_NUM  == 1:
                            doseMatrix = doseMatrix_2
                            targetLabels= targetLabels_2
                        if testcase % TRAIN_NUM  == 2:
                            doseMatrix = doseMatrix_3
                            targetLabels= targetLabels_3
                        if testcase % TRAIN_NUM  == 3:
                            doseMatrix = doseMatrix_4
                            targetLabels= targetLabels_4
                        if testcase % TRAIN_NUM  == 4:
                            doseMatrix = doseMatrix_5
                            targetLabels= targetLabels_5
                        if testcase % TRAIN_NUM == 5:
                            doseMatrix = doseMatrix_6
                            targetLabels = targetLabels_6
                        if testcase % TRAIN_NUM == 6:
                            doseMatrix = doseMatrix_7
                            targetLabels = targetLabels_7
                        if testcase % TRAIN_NUM == 7:
                            doseMatrix = doseMatrix_8
                            targetLabels = targetLabels_8
                        if testcase % TRAIN_NUM == 8:
                            doseMatrix = doseMatrix_9
                            targetLabels = targetLabels_9
                        if testcase % TRAIN_NUM == 9:
                            doseMatrix = doseMatrix_0
                            targetLabels = targetLabels_0


                        done = False
                        step_count = 0
                        lambdaPTV = 1
                        lambdaBLA = 1
                        lambdaREC = 1
                        VPTV = 1
                        VBLA = 1
                        VREC = 1
                        tPTV = 1
                        tBLA = 1
                        tREC = 1


                        MPTV, MBLA, MREC = ProcessDmat(doseMatrix, targetLabels)

                        xVec = np.ones((MPTV.shape[1],))
                        y = MPTV.dot(xVec)
                        gamma = np.zeros((y.shape))

                        xVec,iter,converge,gamma=MinimizeDoseOARnew1(MPTV, MBLA, MREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, tPTV, tBLA, tREC, xVec, gamma)
                        state= DVHplot(MPTV, MBLA, MREC, xVec)

                        Score_fine, Score = planIQ(MPTV, MBLA, MREC, xVec)

                        while not done:
                            flag = np.random.rand()
                            if flag < e:
                                parasel = np.random.randint(8, size=1)
                                actiona = np.random.randint(OUTPUT_SIZE*2+1, size=1)
                                if actiona>=9:
                                    action = 4
                                if actiona >5 and actiona<9:
                                    action = 3
                                if actiona == 5:
                                    action = 2
                                if actiona >1 and actiona<5:
                                    action = 1
                                if actiona <=1:
                                    action = 0


                            else:
                                # Choose an action greedily from the Q-network
                                value = np.zeros(8)
                                value[0] = np.max(mainDQN1.predict(state))
                                value[1] = np.max(mainDQN2.predict(state))
                                value[2] = np.max(mainDQN3.predict(state))
                                value[3] = np.max(mainDQN4.predict(state))
                                value[4] = np.max(mainDQN5.predict(state))
                                value[5] = np.max(mainDQN6.predict(state))
                                value[6] = np.max(mainDQN7.predict(state))
                                value[7] = np.max(mainDQN8.predict(state))
                                parasel = np.argmax(value)
                                if parasel == 0:
                                    action = np.argmax(mainDQN1.predict(state))
                                    qvalue_sum = qvalue_sum + value[0]
                                if parasel == 1:
                                    action = np.argmax(mainDQN2.predict(state))
                                    qvalue_sum = qvalue_sum + value[1]
                                if parasel == 2:
                                    action = np.argmax(mainDQN3.predict(state))
                                    qvalue_sum = qvalue_sum + value[2]
                                if parasel == 3:
                                    action = np.argmax(mainDQN4.predict(state))
                                    qvalue_sum = qvalue_sum + value[3]
                                if parasel == 4:
                                    action = np.argmax(mainDQN5.predict(state))
                                    qvalue_sum = qvalue_sum + value[4]
                                if parasel == 5:
                                    action = np.argmax(mainDQN6.predict(state))
                                    qvalue_sum = qvalue_sum + value[5]
                                if parasel == 6:
                                    action = np.argmax(mainDQN7.predict(state))
                                    qvalue_sum = qvalue_sum + value[6]
                                if parasel == 7:
                                    action = np.argmax(mainDQN8.predict(state))
                                    qvalue_sum = qvalue_sum + value[7]
                                num_q += 1

                            if action == 1:
                                reward = 0
                                next_state = state
                                if step_count >= MAX_STEP-1:  # 29
                                    done = True
                                step_count += 1
                                if parasel==0:
                                    replay_buffer1.append((state, action, reward, next_state, done))

                                if parasel==1:
                                    replay_buffer2.append((state, action, reward, next_state, done))

                                if parasel==2:
                                    replay_buffer3.append((state, action, reward, next_state, done))


                                if parasel==3:
                                    replay_buffer4.append((state, action, reward, next_state, done))


                                if parasel==4:
                                    replay_buffer5.append((state, action, reward, next_state, done))


                                if parasel==5:
                                    replay_buffer6.append((state, action, reward, next_state, done))

                                if parasel==6:
                                    replay_buffer7.append((state, action, reward, next_state, done))

                                if parasel==7:
                                    replay_buffer8.append((state, action, reward, next_state, done))




                                if len(replay_buffer1) > BATCH_SIZE:
                                    if len(replay_buffer1) > int(REPLAY_MEMORY/2):
                                        train_num_episode = 20
                                    else:
                                        train_num_episode = 1
                                    for train_in_episode in range(train_num_episode):
                                        minibatch = rnd.sample(replay_buffer1, BATCH_SIZE)
                                        loss, _ = replay_train(mainDQN1, targetDQN1, targetDQN2, targetDQN3, targetDQN4, targetDQN5, targetDQN6,targetDQN7,targetDQN8,minibatch)
                                    step_count1 += 1
                                    if load_session==0:
                                        print(
                                        "Episode: {}  steps: {}  Loss: {}  iteration_num: {} PlanScoreFine: {} PlanScore: {}  action: {}  lambdaBLA: {}".format(episode + 1, step_count, loss, iter, Score_fine, Score, action, lambdaBLA))
                                    else:
                                        print(
                                            "Episode: {}  steps: {}  Loss: {}  iteration_num: {} PlanScoreFine: {} PlanScore: {}  action: {}  lambdaBLA: {}".format(
                                                episode + LoadEpoch + 1, step_count, loss, iter, Score_fine, Score, action, lambdaBLA))

                                    if step_count1 % TARGET_UPDATE_FREQUENCY == 0:
                                        sess.run(copy_ops1)

                                if len(replay_buffer2) > BATCH_SIZE:
                                    if len(replay_buffer2) > int(REPLAY_MEMORY/2):
                                        train_num_episode = 20
                                    else:
                                        train_num_episode = 1
                                    for train_in_episode in range(train_num_episode):
                                        minibatch = rnd.sample(replay_buffer2, BATCH_SIZE)
                                        loss, _ = replay_train(mainDQN2, targetDQN1, targetDQN2, targetDQN3, targetDQN4, targetDQN5, targetDQN6,targetDQN7,targetDQN8,minibatch)
                                    step_count2 += 1
                                    if load_session == 0:
                                        print(
                                        "Episode: {}  steps: {}  Loss: {}  iteration_num: {}  PlanScoreFine: {} PlanScore: {}  action: {}  lambdaREC: {}".format(episode + 1, step_count, loss, iter, Score_fine, Score, action, lambdaREC))
                                    else:
                                        print(
                                            "Episode: {}  steps: {}  Loss: {}  iteration_num: {}  PlanScoreFine: {} PlanScore: {}  action: {}  lambdaREC: {}".format(
                                                episode + LoadEpoch + 1, step_count, loss, iter, Score_fine, Score, action, lambdaREC))

                                    if step_count2 % TARGET_UPDATE_FREQUENCY == 0:
                                        sess.run(copy_ops2)

                                if len(replay_buffer3) > BATCH_SIZE:
                                    if len(replay_buffer3) > int(REPLAY_MEMORY/2):
                                        train_num_episode = 20
                                    else:
                                        train_num_episode = 1
                                    for train_in_episode in range(train_num_episode):
                                        minibatch = rnd.sample(replay_buffer3, BATCH_SIZE)
                                        loss, _ = replay_train(mainDQN3, targetDQN1, targetDQN2, targetDQN3, targetDQN4, targetDQN5, targetDQN6,targetDQN7,targetDQN8,minibatch)
                                    step_count3 += 1
                                    if load_session == 0:
                                        print("Episode: {}  steps: {}  Loss: {}  iteration_num: {}  PlanScoreFine: {} PlanScore: {}  action: {}  VPTV: {}".format(episode + 1, step_count, loss, iter, Score_fine, Score, action, VPTV))
                                    else:
                                        print(
                                            "Episode: {}  steps: {}  Loss: {}  iteration_num: {}  PlanScoreFine: {} PlanScore: {}  action: {}  VPTV: {}".format(
                                                episode + LoadEpoch + 1, step_count, loss, iter, Score_fine, Score, action, VPTV))
                                    if step_count3 % TARGET_UPDATE_FREQUENCY == 0:
                                        sess.run(copy_ops3)

                                if len(replay_buffer4) > BATCH_SIZE:
                                    if len(replay_buffer4) > int(REPLAY_MEMORY/2):
                                        train_num_episode = 20
                                    else:
                                        train_num_episode = 1
                                    for train_in_episode in range(train_num_episode):
                                        minibatch = rnd.sample(replay_buffer4, BATCH_SIZE)
                                        loss, _ = replay_train(mainDQN4, targetDQN1, targetDQN2, targetDQN3, targetDQN4, targetDQN5, targetDQN6,targetDQN7,targetDQN8,minibatch)
                                    step_count4 += 1
                                    if load_session == 0:
                                        print("Episode: {}  steps: {}  Loss: {}  iteration_num: {}  PlanScoreFine: {} PlanScore: {}  action: {}  VBLA: {}".format(episode + 1, step_count, loss, iter, Score_fine, Score, action, VBLA))
                                    else:
                                        print(
                                            "Episode: {}  steps: {}  Loss: {}  iteration_num: {}  PlanScoreFine: {} PlanScore: {}  action: {}  VBLA: {}".format(
                                                episode + LoadEpoch + 1, step_count, loss, iter, Score_fine, Score, action, VBLA))

                                    if step_count4 % TARGET_UPDATE_FREQUENCY == 0:
                                        sess.run(copy_ops4)

                                if len(replay_buffer5) > BATCH_SIZE:
                                    if len(replay_buffer5) > int(REPLAY_MEMORY/2):
                                        train_num_episode = 20
                                    else:
                                        train_num_episode = 1
                                    for train_in_episode in range(train_num_episode):
                                        minibatch = rnd.sample(replay_buffer5, BATCH_SIZE)
                                        loss, _ = replay_train(mainDQN5, targetDQN1, targetDQN2, targetDQN3, targetDQN4, targetDQN5, targetDQN6,targetDQN7,targetDQN8,minibatch)
                                    step_count5 += 1
                                    if load_session == 0:
                                        print("Episode: {}  steps: {}  Loss: {}  iteration_num: {}  PlanScoreFine: {} PlanScore: {}  action: {}  VREC: {}".format(episode + 1, step_count, loss, iter, Score_fine, Score, action, VREC))
                                    else:
                                        print(
                                            "Episode: {}  steps: {}  Loss: {}  iteration_num: {}  PlanScoreFine: {} PlanScore: {}  action: {}  VREC: {}".format(
                                                episode + LoadEpoch + 1, step_count, loss, iter, Score_fine, Score, action, VREC))
                                    if step_count5 % TARGET_UPDATE_FREQUENCY == 0:
                                        sess.run(copy_ops5)

                                if len(replay_buffer6) > BATCH_SIZE:
                                    if len(replay_buffer6) > int(REPLAY_MEMORY / 2):
                                        train_num_episode = 20
                                    else:
                                        train_num_episode = 1
                                    for train_in_episode in range(train_num_episode):
                                        minibatch = rnd.sample(replay_buffer6, BATCH_SIZE)
                                        loss, _ = replay_train(mainDQN6, targetDQN1, targetDQN2, targetDQN3, targetDQN4,
                                                            targetDQN5, targetDQN6, targetDQN7, targetDQN8,
                                                            minibatch)
                                    step_count6 += 1
                                    if load_session == 0:
                                        print(
                                            "Episode: {}  steps: {}  Loss: {}  iteration_num: {}  PlanScoreFine: {} PlanScore: {}  action: {}  tPTV: {}".format(
                                                episode + 1, step_count, loss, iter, Score_fine, Score, action, tPTV))
                                    else:
                                        print(
                                            "Episode: {}  steps: {}  Loss: {}  iteration_num: {}  PlanScoreFine: {} PlanScore: {}  action: {}  tPTV: {}".format(
                                                episode + LoadEpoch + 1, step_count, loss, iter, Score_fine, Score,
                                                action, tPTV))
                                    if step_count6 % TARGET_UPDATE_FREQUENCY == 0:
                                        sess.run(copy_ops6)

                                if len(replay_buffer7) > BATCH_SIZE:
                                    if len(replay_buffer7) > int(REPLAY_MEMORY / 2):
                                        train_num_episode = 20
                                    else:
                                        train_num_episode = 1
                                    for train_in_episode in range(train_num_episode):
                                        minibatch = rnd.sample(replay_buffer7, BATCH_SIZE)
                                        loss, _ = replay_train(mainDQN7, targetDQN1, targetDQN2, targetDQN3,
                                                            targetDQN4,
                                                            targetDQN5, targetDQN6, targetDQN7, targetDQN8,
                                                            minibatch)
                                    step_count7 += 1
                                    if load_session == 0:
                                        print(
                                            "Episode: {}  steps: {}  Loss: {}  iteration_num: {}  PlanScoreFine: {} PlanScore: {}  action: {}  tBLA: {}".format(
                                                episode + 1, step_count, loss, iter, Score_fine, Score, action,
                                                tBLA))
                                    else:
                                        print(
                                            "Episode: {}  steps: {}  Loss: {}  iteration_num: {}  PlanScoreFine: {} PlanScore: {}  action: {}  tBLA: {}".format(
                                                episode + LoadEpoch + 1, step_count, loss, iter, Score_fine, Score,
                                                action, tBLA))
                                    if step_count7 % TARGET_UPDATE_FREQUENCY == 0:
                                        sess.run(copy_ops7)

                                if len(replay_buffer8) > BATCH_SIZE:
                                    if len(replay_buffer8) > int(REPLAY_MEMORY / 2):
                                        train_num_episode = 20
                                    else:
                                        train_num_episode = 1
                                    for train_in_episode in range(train_num_episode):
                                        minibatch = rnd.sample(replay_buffer8, BATCH_SIZE)
                                        loss, _ = replay_train(mainDQN8, targetDQN1, targetDQN2, targetDQN3,
                                                            targetDQN4,
                                                            targetDQN5, targetDQN6, targetDQN7, targetDQN8,
                                                            minibatch)
                                    step_count8 += 1
                                    if load_session == 0:
                                        print(
                                            "Episode: {}  steps: {}  Loss: {}  iteration_num: {}  PlanScoreFine: {} PlanScore: {}  action: {}  tREC: {}".format(
                                                episode + 1, step_count, loss, iter, Score_fine, Score, action,
                                                tREC))
                                    else:
                                        print(
                                            "Episode: {}  steps: {}  Loss: {}  iteration_num: {}  PlanScoreFine: {} PlanScore: {}  action: {}  tREC: {}".format(
                                                episode + LoadEpoch + 1, step_count, loss, iter, Score_fine, Score,
                                                action, tREC))
                                    if step_count8 % TARGET_UPDATE_FREQUENCY == 0:
                                        sess.run(copy_ops8)

                                continue


                            if action != 1:
                                if parasel == 0:
                                    if action == 0:
                                        action_factor = 1.5
                                        lambdaBLA = lambdaBLA * action_factor
                                        if lambdaBLA >= paraMax:
                                            lambdaBLA = paraMax

                                    elif action == 2:
                                        action_factor = 0.90
                                        lambdaBLA = lambdaBLA * action_factor
                                        if lambdaBLA <= paraMin:
                                            lambdaBLA = paraMin

                                    else:
                                        action_factor = 0.5
                                        lambdaBLA = lambdaBLA * action_factor
                                        if lambdaBLA <= paraMin:
                                            lambdaBLA = paraMin

                                if parasel == 1:
                                    if action == 0:
                                        action_factor = 1.5
                                        lambdaREC = lambdaREC * action_factor
                                        if lambdaREC >= paraMax:
                                            lambdaREC = paraMax


                                    elif action == 2:
                                        action_factor = 0.9
                                        lambdaREC = lambdaREC * action_factor
                                        if lambdaREC <= paraMin:
                                            lambdaREC = paraMin

                                    else:
                                        action_factor = 0.5
                                        lambdaREC = lambdaREC * action_factor
                                        if lambdaREC <= paraMin:
                                            lambdaREC = paraMin

                                if parasel == 2:
                                    if action == 0:
                                        action_factor = 1.5
                                        VPTV = VPTV * action_factor
                                        if VPTV >= paraMax:
                                            VPTV = paraMax

                                    elif action == 2:
                                        action_factor = 0.9
                                        VPTV = VPTV * action_factor
                                        if VPTV <= paraMin:
                                            VPTV = paraMin

                                    else:
                                        action_factor = 0.5
                                        VPTV = VPTV * action_factor
                                        if VPTV <= paraMin:
                                            VPTV = paraMin

                                if parasel == 3:
                                    if action == 0:
                                        action_factor = 1.5
                                        VBLA = VBLA * action_factor
                                        if VBLA >= paraMax:
                                            VBLA = paraMax


                                    elif action == 2:
                                        action_factor = 0.9
                                        VBLA = VBLA * action_factor
                                        if VBLA <= paraMin:
                                            VBLA = paraMin

                                    else:
                                        action_factor = 0.5
                                        VBLA = VBLA * action_factor
                                        if VBLA <= paraMin:
                                            VBLA = paraMin

                                if parasel == 4:
                                    if action == 0:
                                        action_factor = 1.5
                                        VREC = VREC * action_factor
                                        if VREC >= paraMax:
                                            VREC = paraMax

                                    elif action == 2:
                                        action_factor = 0.9
                                        VREC = VREC * action_factor
                                        if VREC <= paraMin:
                                            VREC = paraMin

                                    else:
                                        action_factor = 0.5
                                        VREC = VREC * action_factor
                                        if VREC <= paraMin:
                                            VREC = paraMin

                                if parasel == 5:
                                    if action == 0:
                                        action_factor = 1.05
                                        tPTV = tPTV * action_factor
                                        if tPTV >= 1.5:
                                            tPTV = 1.5

                                    elif action == 2:
                                        action_factor = 0.99
                                        tPTV = tPTV * action_factor
                                        if tPTV <= 1:
                                            tPTV = 1

                                    else:
                                        action_factor = 0.95
                                        tPTV = tPTV * action_factor
                                        if tPTV <= 1:
                                            tPTV = 1

                                if parasel == 6:
                                    if action == 0:
                                        action_factor = 1.5
                                        tBLA = tBLA * action_factor
                                        if tBLA >= paraMax:
                                            tBLA = paraMax

                                    elif action == 2:
                                        action_factor = 0.9
                                        tBLA = tBLA * action_factor
                                        if tBLA <= paraMin:
                                            tBLA = paraMin

                                    else:
                                        action_factor = 0.5
                                        tBLA = tBLA * action_factor
                                        if tBLA <= paraMin:
                                            tBLA = paraMin

                                if parasel == 7:
                                    if action == 0:
                                        action_factor = 1.5
                                        tREC = tREC * action_factor
                                        if tREC >= paraMax:
                                            tREC = paraMax

                                    elif action == 2:
                                        action_factor = 0.9
                                        tREC = tREC * action_factor
                                        if tREC <= paraMin:
                                            tREC = paraMin

                                    else:
                                        action_factor = 0.5
                                        tREC = tREC * action_factor
                                        if tREC <= paraMin:
                                            tREC = paraMin

                                print(
                                    "MPTV: {} MBLA: {}  MREC: {}  lambdaBLA: {}  lambdaREC: {}  VPTV: {}  VBLA: {} VREC: {} tPTV: {} tBLA: {} tREC: {}".format(
                                        MPTV.shape[0], MBLA.shape[0], MREC.shape[0], lambdaBLA, lambdaREC, VPTV, VBLA, VREC, tPTV, tBLA, tREC))

                                xVec,iter,converge,gamma=MinimizeDoseOARnew1(MPTV, MBLA, MREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, tPTV, tBLA, tREC, xVec, gamma)
                                next_state= DVHplot(MPTV, MBLA, MREC, xVec)

                                Score_fine1, Score = planIQ(MPTV, MBLA, MREC, xVec)
                                reward = Score_fine1-Score_fine
                                Score_fine = Score_fine1




                                # print(reward)
                                if step_count >= MAX_STEP - 1:
                                    done = True
                                step_count += 1


                                if flag >= e:
                                    futureReward = zeros((8))
                                    futureReward[0] = np.max(targetDQN1.predict(next_state))
                                    futureReward[1] = np.max(targetDQN2.predict(next_state))
                                    futureReward[2] = np.max(targetDQN3.predict(next_state))
                                    futureReward[3] = np.max(targetDQN4.predict(next_state))
                                    futureReward[4] = np.max(targetDQN5.predict(next_state))
                                    futureReward[5] = np.max(targetDQN6.predict(next_state))
                                    futureReward[6] = np.max(targetDQN7.predict(next_state))
                                    futureReward[7] = np.max(targetDQN8.predict(next_state))
                                    if done != True:
                                        reward_sum_total = reward_sum_total + reward + DISCOUNT_RATE * np.max(futureReward)
                                    else:
                                        reward_sum_total = reward_sum_total + reward


                                if parasel==0:
                                    replay_buffer1.append((state, action, reward, next_state, done))

                                if parasel==1:
                                    replay_buffer2.append((state, action, reward, next_state, done))

                                if parasel==2:
                                    replay_buffer3.append((state, action, reward, next_state, done))

                                if parasel==3:
                                    replay_buffer4.append((state, action, reward, next_state, done))

                                if parasel==4:
                                    replay_buffer5.append((state, action, reward, next_state, done))

                                if parasel==5:
                                    replay_buffer6.append((state, action, reward, next_state, done))

                                if parasel==6:
                                    replay_buffer7.append((state, action, reward, next_state, done))

                                if parasel==7:
                                    replay_buffer8.append((state, action, reward, next_state, done))




                                if len(replay_buffer1) > BATCH_SIZE:
                                    if len(replay_buffer1) > int(REPLAY_MEMORY/2):
                                        train_num_episode = 20
                                    else:
                                        train_num_episode = 1
                                    for train_in_episode in range(train_num_episode):
                                        minibatch = rnd.sample(replay_buffer1, BATCH_SIZE)
                                        loss, _ = replay_train(mainDQN1, targetDQN1, targetDQN2, targetDQN3, targetDQN4, targetDQN5, targetDQN6,targetDQN7,targetDQN8,minibatch)
                                    step_count1 += 1
                                    if load_session==0:
                                        print(
                                        "Episode: {}  steps: {}  Loss: {}  iteration_num: {} PlanScoreFine: {} PlanScore: {}  action: {}  lambdaBLA: {}".format(episode + 1, step_count, loss, iter, Score_fine, Score, action, lambdaBLA))
                                    else:
                                        print(
                                            "Episode: {}  steps: {}  Loss: {}  iteration_num: {} PlanScoreFine: {} PlanScore: {}  action: {}  lambdaBLA: {}".format(
                                                episode + LoadEpoch + 1, step_count, loss, iter, Score_fine, Score, action, lambdaBLA))

                                    if step_count1 % TARGET_UPDATE_FREQUENCY == 0:
                                        sess.run(copy_ops1)

                                if len(replay_buffer2) > BATCH_SIZE:
                                    if len(replay_buffer2) > int(REPLAY_MEMORY/2):
                                        train_num_episode = 20
                                    else:
                                        train_num_episode = 1
                                    for train_in_episode in range(train_num_episode):
                                        minibatch = rnd.sample(replay_buffer2, BATCH_SIZE)
                                        loss, _ = replay_train(mainDQN2, targetDQN1, targetDQN2, targetDQN3, targetDQN4, targetDQN5, targetDQN6,targetDQN7,targetDQN8,minibatch)
                                    step_count2 += 1
                                    if load_session == 0:
                                        print(
                                        "Episode: {}  steps: {}  Loss: {}  iteration_num: {}  PlanScoreFine: {} PlanScore: {}  action: {}  lambdaREC: {}".format(episode + 1, step_count, loss, iter, Score_fine, Score, action, lambdaREC))
                                    else:
                                        print(
                                            "Episode: {}  steps: {}  Loss: {}  iteration_num: {}  PlanScoreFine: {} PlanScore: {}  action: {}  lambdaREC: {}".format(
                                                episode + LoadEpoch + 1, step_count, loss, iter, Score_fine, Score, action, lambdaREC))

                                    if step_count2 % TARGET_UPDATE_FREQUENCY == 0:
                                        sess.run(copy_ops2)

                                if len(replay_buffer3) > BATCH_SIZE:
                                    if len(replay_buffer3) > int(REPLAY_MEMORY/2):
                                        train_num_episode = 20
                                    else:
                                        train_num_episode = 1
                                    for train_in_episode in range(train_num_episode):
                                        minibatch = rnd.sample(replay_buffer3, BATCH_SIZE)
                                        loss, _ = replay_train(mainDQN3, targetDQN1, targetDQN2, targetDQN3, targetDQN4, targetDQN5, targetDQN6,targetDQN7,targetDQN8,minibatch)
                                    step_count3 += 1
                                    if load_session == 0:
                                        print("Episode: {}  steps: {}  Loss: {}  iteration_num: {}  PlanScoreFine: {} PlanScore: {}  action: {}  VPTV: {}".format(episode + 1, step_count, loss, iter, Score_fine, Score, action, VPTV))
                                    else:
                                        print(
                                            "Episode: {}  steps: {}  Loss: {}  iteration_num: {}  PlanScoreFine: {} PlanScore: {}  action: {}  VPTV: {}".format(
                                                episode + LoadEpoch + 1, step_count, loss, iter, Score_fine, Score, action, VPTV))
                                    if step_count3 % TARGET_UPDATE_FREQUENCY == 0:
                                        sess.run(copy_ops3)

                                if len(replay_buffer4) > BATCH_SIZE:
                                    if len(replay_buffer4) > int(REPLAY_MEMORY/2):
                                        train_num_episode = 20
                                    else:
                                        train_num_episode = 1
                                    for train_in_episode in range(train_num_episode):
                                        minibatch = rnd.sample(replay_buffer4, BATCH_SIZE)
                                        loss, _ = replay_train(mainDQN4, targetDQN1, targetDQN2, targetDQN3, targetDQN4, targetDQN5, targetDQN6,targetDQN7,targetDQN8,minibatch)
                                    step_count4 += 1
                                    if load_session == 0:
                                        print("Episode: {}  steps: {}  Loss: {}  iteration_num: {}  PlanScoreFine: {} PlanScore: {}  action: {}  VBLA: {}".format(episode + 1, step_count, loss, iter, Score_fine, Score, action, VBLA))
                                    else:
                                        print(
                                            "Episode: {}  steps: {}  Loss: {}  iteration_num: {}  PlanScoreFine: {} PlanScore: {}  action: {}  VBLA: {}".format(
                                                episode + LoadEpoch + 1, step_count, loss, iter, Score_fine, Score, action, VBLA))

                                    if step_count4 % TARGET_UPDATE_FREQUENCY == 0:
                                        sess.run(copy_ops4)

                                if len(replay_buffer5) > BATCH_SIZE:
                                    if len(replay_buffer5) > int(REPLAY_MEMORY/2):
                                        train_num_episode = 20
                                    else:
                                        train_num_episode = 1
                                    for train_in_episode in range(train_num_episode):
                                        minibatch = rnd.sample(replay_buffer5, BATCH_SIZE)
                                        loss, _ = replay_train(mainDQN5, targetDQN1, targetDQN2, targetDQN3, targetDQN4, targetDQN5, targetDQN6,targetDQN7,targetDQN8,minibatch)

                                    step_count5 += 1
                                    if load_session == 0:
                                        print("Episode: {}  steps: {}  Loss: {}  iteration_num: {}  PlanScoreFine: {} PlanScore: {}  action: {}  VREC: {}".format(episode + 1, step_count, loss, iter, Score_fine, Score, action, VREC))
                                    else:
                                        print(
                                            "Episode: {}  steps: {}  Loss: {}  iteration_num: {}  PlanScoreFine: {} PlanScore: {}  action: {}  VREC: {}".format(
                                                episode + LoadEpoch + 1, step_count, loss, iter, Score_fine, Score, action, VREC))
                                    if step_count5 % TARGET_UPDATE_FREQUENCY == 0:
                                        sess.run(copy_ops5)

                                if len(replay_buffer6) > BATCH_SIZE:
                                    if len(replay_buffer6) > int(REPLAY_MEMORY / 2):
                                        train_num_episode = 20
                                    else:
                                        train_num_episode = 1
                                    for train_in_episode in range(train_num_episode):
                                        minibatch = rnd.sample(replay_buffer6, BATCH_SIZE)
                                        loss, _ = replay_train(mainDQN6, targetDQN1, targetDQN2, targetDQN3, targetDQN4,
                                                            targetDQN5, targetDQN6, targetDQN7, targetDQN8,
                                                            minibatch)
                                    step_count6 += 1
                                    if load_session == 0:
                                        print(
                                            "Episode: {}  steps: {}  Loss: {}  iteration_num: {}  PlanScoreFine: {} PlanScore: {}  action: {}  tPTV: {}".format(
                                                episode + 1, step_count, loss, iter, Score_fine, Score, action, tPTV))
                                    else:
                                        print(
                                            "Episode: {}  steps: {}  Loss: {}  iteration_num: {}  PlanScoreFine: {} PlanScore: {}  action: {}  tPTV: {}".format(
                                                episode + LoadEpoch + 1, step_count, loss, iter, Score_fine, Score,
                                                action, tPTV))
                                    if step_count6 % TARGET_UPDATE_FREQUENCY == 0:
                                        sess.run(copy_ops6)

                                if len(replay_buffer7) > BATCH_SIZE:
                                    if len(replay_buffer7) > int(REPLAY_MEMORY / 2):
                                        train_num_episode = 20
                                    else:
                                        train_num_episode = 1
                                    for train_in_episode in range(train_num_episode):
                                        minibatch = rnd.sample(replay_buffer7, BATCH_SIZE)
                                        loss, _ = replay_train(mainDQN7, targetDQN1, targetDQN2, targetDQN3,
                                                            targetDQN4,
                                                            targetDQN5, targetDQN6, targetDQN7, targetDQN8,
                                                            minibatch)
                                    step_count7 += 1
                                    if load_session == 0:
                                        print(
                                            "Episode: {}  steps: {}  Loss: {}  iteration_num: {}  PlanScoreFine: {} PlanScore: {}  action: {}  tBLA: {}".format(
                                                episode + 1, step_count, loss, iter, Score_fine, Score, action,
                                                tBLA))
                                    else:
                                        print(
                                            "Episode: {}  steps: {}  Loss: {}  iteration_num: {}  PlanScoreFine: {} PlanScore: {}  action: {}  tBLA: {}".format(
                                                episode + LoadEpoch + 1, step_count, loss, iter, Score_fine, Score,
                                                action, tBLA))
                                    if step_count7 % TARGET_UPDATE_FREQUENCY == 0:
                                        sess.run(copy_ops7)

                                if len(replay_buffer8) > BATCH_SIZE:
                                    if len(replay_buffer8) > int(REPLAY_MEMORY / 2):
                                        train_num_episode = 20
                                    else:
                                        train_num_episode = 1
                                    for train_in_episode in range(train_num_episode):
                                        minibatch = rnd.sample(replay_buffer8, BATCH_SIZE)
                                        loss, _ = replay_train(mainDQN8, targetDQN1, targetDQN2, targetDQN3,
                                                            targetDQN4,
                                                            targetDQN5, targetDQN6, targetDQN7, targetDQN8,
                                                            minibatch)
                                    step_count8 += 1
                                    if load_session == 0:
                                        print(
                                            "Episode: {}  steps: {}  Loss: {}  iteration_num: {}  PlanScoreFine: {} PlanScore: {}  action: {}  tREC: {}".format(
                                                episode + 1, step_count, loss, iter, Score_fine, Score, action,
                                                tREC))
                                    else:
                                        print(
                                            "Episode: {}  steps: {}  Loss: {}  iteration_num: {}  PlanScoreFine: {} PlanScore: {}  action: {}  tREC: {}".format(
                                                episode + LoadEpoch + 1, step_count, loss, iter, Score_fine, Score,
                                                action, tREC))
                                    if step_count8 % TARGET_UPDATE_FREQUENCY == 0:
                                        sess.run(copy_ops8)

                                state = next_state

                    if num_q !=0:
                        reward_check[episode]=reward_sum_total/num_q
                    else:
                        reward_check[episode]=0

                    if num_q!=0:
                        q_check[episode] = qvalue_sum/num_q
                    else:
                        q_check[episode]=0

                    print("Episode: {}  Reward: {}  Q-value: {} ".format
                                        (episode + 1, reward_check[episode], q_check[episode]))



                    if save_session == 1 and (episode+1)% 5 == 0:
                        saver.save(sess, save_session_name, global_step=episode+1)
                        data_out='/data/data/Results/GPU2/checks/'
                        np.save(data_out+'reward_check.npy'+str(episode+1),reward_check)
                        np.save(data_out+'q_check.npy' + str(episode + 1), q_check)
                    if (episode+1)% 5 == 0:
                        bot_play(mainDQN1, mainDQN2, mainDQN3, mainDQN4, mainDQN5, mainDQN6, mainDQN7, mainDQN8,DVHplot,MinimizeDoseOARnew1,episode+1)

            print("Training done!, Test start!")

            bot_play(mainDQN1, mainDQN2, mainDQN3, mainDQN4, mainDQN5,  mainDQN6, mainDQN7, mainDQN8, DVHplot,MinimizeDoseOARnew1,LoadEpoch)

            # bot_play(mainDQN1,mainDQN2,mainDQN3,mainDQN4,mainDQN5,runOpt1,episode+1)



if __name__ == "__main__":
	main()
