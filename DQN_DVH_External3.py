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
# import cupy as cp

INPUT_SIZE = 100  # DVH interval number
OUTPUT_SIZE = 5  # number of actions, each lambda has three actions(+,=,-)
TRAIN_NUM = 10
DISCOUNT_RATE = 0.30
REPLAY_MEMORY = 5000
BATCH_SIZE = 16
TARGET_UPDATE_FREQUENCY = 10
MAX_EPISODES = 250
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


def bot_play(mainDQN1: dqn_dvh_external_network.DQN, mainDQN2: dqn_dvh_external_network.DQN, mainDQN3: dqn_dvh_external_network.DQN, mainDQN4: dqn_dvh_external_network.DQN,  mainDQN5: dqn_dvh_external_network.DQN, mainDQN6: dqn_dvh_external_network.DQN, mainDQN7: dqn_dvh_external_network.DQN, mainDQN8: dqn_dvh_external_network.DQN, runOpt1, episode) -> None:
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
        state_test, iter, xVec, gamma = \
            runOpt1(MPTV, MBLA, MREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, tPTV, tBLA, tREC, xVec, gamma)
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

            if action != 1:
                state_test, iter, xVec, gamma = runOpt1(MPTV, MBLA, MREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, tPTV, tBLA, tREC, xVec, gamma)

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

def MinimizeDoseOARnew1(MPTV, MBLA, MREC, tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, xVec,gamma):
    # treatment planning optimization in DVH-based scheme
    beta = 2
    lambdaBLA = lambdaBLA/lambdaPTV
    lambdaREC = lambdaREC/lambdaPTV
    # xVec = np.ones((MPTV.shape[1],))
    DPTV = MPTV.dot(xVec)
    DPTV1 = np.sort(DPTV)
    posi = int(round(0.05 * DPTV1.shape[0]))
    D95 = DPTV1[posi]
#    MPTV95 = MPTV[DPTV>=D95,:]
    factor = pdose / D95
    xVec = xVec * factor
    y = MPTV.dot(xVec)
    MPTVT = MPTV.transpose()
    DPTV = MPTV.dot(xVec)
    for iter in range(maxiter):
        # loop in iteration
        xVec_old = xVec
        DPTV1 = np.sort(DPTV)
        posi = int(round((1 - VPTV) * DPTV1.shape[0]))-1
        if posi < 0:###### what was missing
            posi = 0###### what was missing
        DPTVV = DPTV1[posi]
        DBLA = MBLA.dot(xVec)
        DBLA1 = sort(DBLA)
        posi = int(round((1 - VBLA) * DBLA1.shape[0]))-1
        if posi < 0:
            posi = 0
        DBLAV = DBLA1[posi]
        DREC = MREC.dot(xVec)
        DREC1 = sort(DREC)
        posi = int(round((1 - VREC) * DREC1.shape[0]))-1
        if posi < 0:
            posi = 0
        DRECV = DREC1[posi]

        MPTVV =  MPTV[DPTV>=DPTVV,:]
        temp= DPTV[DPTV>=DPTVV]
        if np.max(temp) > pdose* tPTV:
            MPTV1 = MPTVV[temp > pdose*tPTV, :]
            targetPTV1 = pdose*tPTV*np.ones((MPTV1.shape[0],))
            MPTV1T = MPTV1.transpose()
            temp1 = MPTV1.dot(xVec)
            temp1 = MPTV1T.dot(temp1)
            temp1 = temp1 * 1/MPTV1.shape[0]
            b1 = MPTV1T.dot(targetPTV1) / MPTV1.shape[0]
        else:
            temp1 = zeros((xVec.shape))
            b1 = zeros((xVec.shape))
            tempp1 = zeros((xVec.shape))
        tempptv = temp

        temp2 = MPTV.dot(xVec)
        temp2 = beta*MPTVT.dot(temp2)/y.shape[0]
        b2 =  beta*MPTVT.dot(y)/y.shape[0]


        MBLAV = MBLA[DBLA >= DBLAV,:]
        temp = DBLA[DBLA >= DBLAV]
        if np.max(temp) > pdose * tBLA:
            MBLA1 = MBLAV[temp > pdose * tBLA, :]
            targetBLA1 = pdose*tBLA*np.ones((MBLA1.shape[0],))
            MBLA1T = MBLA1.transpose()
            temp3 = MBLA1.dot(xVec)
            temp3 = MBLA1T.dot(temp3)
            temp3 = temp3 * lambdaBLA/MBLA1.shape[0]
            b3 = lambdaBLA * MBLA1T.dot(targetBLA1) / max(MBLA1.shape[0], 1)
        else:
            temp3 = zeros((xVec.shape))
            b3 = zeros((xVec.shape))
            tempp3 = zeros((xVec.shape))
        tempbla = temp

        MRECV = MREC[DREC >= DRECV, :]
        temp = DREC[DREC >= DRECV]
        if np.max(temp) > pdose * tREC:
            MREC1 = MRECV[temp > pdose * tREC, :]
            targetREC1 = pdose*tREC*np.ones((MREC1.shape[0],))
            MREC1T = MREC1.transpose()
            temp4 = MREC1.dot(xVec)
            temp4 = MREC1T.dot(temp4)
            temp4 = temp4 * lambdaREC/MREC1.shape[0]
            b4 = lambdaREC * MREC1T.dot(targetREC1) / MREC1.shape[0]
        else:
            temp4 = zeros((xVec.shape))
            b4 = zeros((xVec.shape))
            tempp4 = zeros((xVec.shape))
        temprec = temp

        templhs = temp1+temp2+temp3+temp4
        b = b1+b2+b3+b4-MPTVT.dot(gamma)
        r = b - templhs
        p = r
        rsold = np.inner(r,r)

        if rsold>1e-10:
            # CG for solving linear systems
            for i in range(3):
                if np.max(tempptv) > pdose*tPTV :
                    tempp1 = MPTV1.dot(p)
                    tempp1 = MPTV1T.dot(tempp1)
                    tempp1 = tempp1 * 1 / MPTV1.shape[0]

                tempp2 = MPTV.dot(p)
                tempp2 = beta * MPTVT.dot(tempp2)/y.shape[0]

                if np.max(tempbla) > pdose * tBLA:
                    tempp3 = MBLA1.dot(p)
                    tempp3 = MBLA1T.dot(tempp3)
                    tempp3 = tempp3 * lambdaBLA / MBLA1.shape[0]

                if np.max(temprec) > pdose * tREC:
                    tempp4 = MREC1.dot(p)
                    tempp4 = MREC1T.dot(tempp4)
                    tempp4 = tempp4 * lambdaREC / MREC1.shape[0]

                Ap = tempp1 + tempp2 + tempp3 + tempp4
                pAp = np.inner(p, Ap)
                alpha = rsold / pAp
                xVec = xVec + alpha * p
                xVec[xVec<0]=0
                r = r - alpha * Ap
                rsnew = np.inner(r, r)
                if sqrt(rsnew) < 1e-5:
                    break
                p = r + (rsnew / rsold) * p
                rsold = rsnew
        DPTV = MPTV.dot(xVec)
        y = (DPTV * beta/y.shape[0] + gamma) / (beta/y.shape[0])
        Dy = np.sort(y)
        posi = int(round(0.05 * Dy.shape[0]))
        D95 = Dy[posi]
        temp = np.zeros(y.shape)
        temp[y>=D95] = y[y>=D95]
        temp[temp<pdose] = pdose
        y[y>=D95] = temp[y>=D95]
        gamma = gamma + beta * (MPTV.dot(xVec)-y)/y.shape[0]

        if LA.norm(xVec - xVec_old, 2) / LA.norm(xVec_old, 2) < 5e-3:
            break
    DPTV = MPTV.dot(xVec)
    DPTV1 = np.sort(DPTV)
    posi = int(round(0.05 * DPTV1.shape[0]))
    D95 = DPTV1[posi]
    factor = pdose / D95 # thresholidng
    xVec = xVec * factor
    converge = 1
    if iter == maxiter - 1:
        converge = 0


    #print("optimization time: {} seconds for {} iterrations factor {}".format((time.time() - start_time), iter,factor))
    print("DAMON LOOK HERE",converge,iter,factor)
    return xVec, iter, converge, gamma



def runOpt1(MPTV, MBLA, MREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, tPTV, tBLA, tREC, xVec, gamma):

    # xVec, iter, converge = MinimizeDoseOAR(MPTV, MBLA, MREC, lambda1, lambdaREC, VPTV, VBLA, VREC)
    xVec, iter, converge, gamma = MinimizeDoseOARnew1(MPTV, MBLA, MREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, tPTV, tBLA, tREC, xVec, gamma)
    j = 0
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

    Y = zeros((INPUT_SIZE,3))
    Y[:, 0] = y_ptv
    Y[:, 1] = y_bladder
    Y[:, 2] = y_rectum

    Y = np.reshape(Y,(100*3,),order = 'F')


    return Y, iter, xVec, gamma

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
    session_folder2='/data/data/Results/GPU2/session/'
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

                        state, iter, xVec, gamma = \
                            runOpt1(MPTV, MBLA, MREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, tPTV, tBLA, tREC, xVec, gamma)
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
                                next_state, iter, xVec, gamma = \
                                    runOpt1(MPTV, MBLA, MREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, tPTV, tBLA, tREC, xVec, gamma)
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



                    if save_session == 1 and (episode+1):
                        saver.save(sess, save_session_name, global_step=episode+1)
                        data_out='/data/data/Results/GPU2/checks/'
                        np.save(data_out+'reward_check.npy'+str(episode+1),reward_check)
                        np.save(data_out+'q_check.npy' + str(episode + 1), q_check)
                    if (episode+1)% 5 == 0:
                        bot_play(mainDQN1, mainDQN2, mainDQN3, mainDQN4, mainDQN5, mainDQN6, mainDQN7, mainDQN8, runOpt1,episode+1)

            print("Training done!, Test start!")

            bot_play(mainDQN1, mainDQN2, mainDQN3, mainDQN4, mainDQN5,  mainDQN6, mainDQN7, mainDQN8, runOpt1,LoadEpoch)

            # bot_play(mainDQN1,mainDQN2,mainDQN3,mainDQN4,mainDQN5,runOpt1,episode+1)



if __name__ == "__main__":
	main()
