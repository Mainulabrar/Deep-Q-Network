
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:55:19 2019

aim: DQN for dvh-based treatment planning parameter selection

@author: writed by Chenyang Shen, modified by Chao Wang
"""

import logging
import numpy as np
import tensorflow as tf
import random as rnd
from collections import deque
from typing import List
#import dqn_rule_network
import math as m
from lib_dvh.dqn_rule_network import DQN
from numpy import zeros
import matplotlib.pyplot as plt
from timeit import Timer
import time
# ===================== back up  ================================
# import matplotlib.pyplot as plt
# import os
# import time
# import scipy.io as sio
# from numpy import *
# from numpy import zeros, sqrt
# import scipy.linalg
# from tensorflow.python.framework import dtypes
# import datetime
# import pandas as pd
# from numpy import linalg as LA
# from scipy.sparse import vstack
# =============================================================================
from lib_dvh.data_prep import loadDoseMatrix,loadMask,ProcessDmat
from lib_dvh.score_calcu import planIQ
from lib_dvh.TP_DVH_algo1 import runOpt_dvh
from lib_dvh.validation import bot_play
from lib_dvh.exalu_training import evalu_training
from termcolor import colored
logging.basicConfig(filename = '/data/GPU6results/logstuff/',level = logging.INFO,
                    format = '%(asctime)s:%(message)s')

# ------------------------- setting for training and testing  ------------------------------
INPUT_SIZE = 100  # DVH interval number
OUTPUT_SIZE = 3  # number of actions, each lambda has three actions(+,=,-)
TRAIN_NUM = 1 # number of training set
DISCOUNT_RATE = 0.70
REPLAY_MEMORY = 125000
BATCH_SIZE = 16
TARGET_UPDATE_FREQUENCY = 500
MAX_EPISODES = 1
MAX_STEP = 30
load_session = 1# 1 for loading weight from #LoadEpoch; 0 for starting NN from randomn weight
save_session = 1 # 1 for saving the output
Start = 0 # 1 for training and 0 for testing
LoadEpoch = 180 # if load_session is 1, then loading the weight from LoadEpoch
pdose = 1 # target dose for PTV
maxiter = 40 # maximum iteration number for treatment planing optimization
# ------------- range of parmaeter -----------------
paraMax = 100000 # change in validation as well
paraMin = 0
paraMax_tPTV = 1.2
paraMin_tPTV = 1
paraMax_tOAR = 1
paraMax_VOAR = 1

paraMax_VPTV = 0.3
# ---------------------------------------------------


def main():


     save_session_name = '/data/data/sessionrm125/'
     #save_session_name2 = '/data/data/Results/GPU2/sessionHz2501/'
   # ---------------store the previous observations in replay memory ------------
     replay_buffer1= deque(maxlen=REPLAY_MEMORY)
     replay_buffer2 = deque(maxlen=REPLAY_MEMORY)
     replay_buffer3 = deque(maxlen=REPLAY_MEMORY)
     replay_buffer4 = deque(maxlen=REPLAY_MEMORY)
     replay_buffer5 = deque(maxlen=REPLAY_MEMORY)
     replay_buffer6 = deque(maxlen=REPLAY_MEMORY)
     replay_buffer7 = deque(maxlen=REPLAY_MEMORY)
     replay_buffer8 = deque(maxlen=REPLAY_MEMORY)
     replay_buffer9 = deque(maxlen=REPLAY_MEMORY)
   # ---------------------------------------------------------------------------
     with tf.device('/device:GPU:2'):
         with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True,gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
             ## ------------------------------- initial 9 NNs for all parameters ----------------------------------------------------
            mainDQN1 = DQN()
            mainDQN1.build_and_compile_DQN(img_x = INPUT_SIZE, channels_in = 3, channels_out=1, dropout_rate=0, learningRate = 1e-5)
            targetDQN1 = DQN()
            targetDQN1.build_and_compile_DQN(img_x = INPUT_SIZE, channels_in = 3, channels_out=1, dropout_rate=0, learningRate = 1e-5)

            mainDQN2 = DQN()
            mainDQN2.build_and_compile_DQN(img_x = INPUT_SIZE, channels_in = 3, channels_out=1, dropout_rate=0, learningRate = 1e-5)
            targetDQN2 = DQN()
            targetDQN2.build_and_compile_DQN(img_x = INPUT_SIZE, channels_in = 3, channels_out=1, dropout_rate=0, learningRate = 1e-5)

            mainDQN3 = DQN()
            mainDQN3.build_and_compile_DQN(img_x = INPUT_SIZE, channels_in = 3, channels_out=1, dropout_rate=0, learningRate = 1e-5)
            targetDQN3 = DQN()
            targetDQN3.build_and_compile_DQN(img_x = INPUT_SIZE, channels_in = 3, channels_out=1, dropout_rate=0, learningRate = 1e-5)

            mainDQN4 = DQN()
            mainDQN4.build_and_compile_DQN(img_x = INPUT_SIZE, channels_in =3, channels_out=1, dropout_rate=0, learningRate = 1e-5)
            targetDQN4 = DQN()
            targetDQN4.build_and_compile_DQN(img_x = INPUT_SIZE, channels_in = 3, channels_out=1, dropout_rate=0, learningRate = 1e-5)

            mainDQN5 = DQN()
            mainDQN5.build_and_compile_DQN(img_x = INPUT_SIZE, channels_in = 3, channels_out=1, dropout_rate=0, learningRate = 1e-5)
            targetDQN5 = DQN()
            targetDQN5.build_and_compile_DQN(img_x = INPUT_SIZE, channels_in = 3, channels_out=1, dropout_rate=0, learningRate = 1e-5)

            mainDQN6 = DQN()
            mainDQN6.build_and_compile_DQN(img_x = INPUT_SIZE, channels_in = 3, channels_out=1, dropout_rate=0, learningRate = 1e-5)
            targetDQN6 = DQN()
            targetDQN6.build_and_compile_DQN(img_x = INPUT_SIZE, channels_in = 3, channels_out=1, dropout_rate=0, learningRate = 1e-5)

            mainDQN7 = DQN()
            mainDQN7.build_and_compile_DQN(img_x = INPUT_SIZE, channels_in = 3, channels_out=1, dropout_rate=0, learningRate = 1e-5)
            targetDQN7 = DQN()
            targetDQN7.build_and_compile_DQN(img_x = INPUT_SIZE, channels_in = 3, channels_out=1, dropout_rate=0, learningRate = 1e-5)

            mainDQN8 = DQN()
            mainDQN8.build_and_compile_DQN(img_x = INPUT_SIZE, channels_in =3, channels_out=1, dropout_rate=0, learningRate = 1e-5)
            targetDQN8 = DQN()
            targetDQN8.build_and_compile_DQN(img_x = INPUT_SIZE, channels_in = 3, channels_out=1, dropout_rate=0, learningRate = 1e-5)

            mainDQN9 = DQN()
            mainDQN9.build_and_compile_DQN(img_x = INPUT_SIZE, channels_in = 3, channels_out=1, dropout_rate=0, learningRate = 1e-5)
            targetDQN9 = DQN()
            targetDQN9.build_and_compile_DQN(img_x = INPUT_SIZE, channels_in = 3, channels_out=1, dropout_rate=0, learningRate = 1e-5)

            targetDQN1.model.set_weights(mainDQN1.model.get_weights())
            targetDQN2.model.set_weights(mainDQN2.model.get_weights())
            targetDQN3.model.set_weights(mainDQN3.model.get_weights())
            targetDQN4.model.set_weights(mainDQN4.model.get_weights())
            targetDQN5.model.set_weights(mainDQN5.model.get_weights())
            targetDQN6.model.set_weights(mainDQN6.model.get_weights())
            targetDQN7.model.set_weights(mainDQN7.model.get_weights())
            targetDQN8.model.set_weights(mainDQN8.model.get_weights())
            targetDQN9.model.set_weights(mainDQN9.model.get_weights())

            if load_session == 1:
                # loading the weight
                mainDQN1.model.load_weights(save_session_name + 'mainDQN1_episode_' + str(LoadEpoch)+'.h5')
                mainDQN2.model.load_weights(save_session_name + 'mainDQN2_episode_' + str(LoadEpoch)+'.h5')
                mainDQN3.model.load_weights(save_session_name + 'mainDQN3_episode_' + str(LoadEpoch)+'.h5')
                mainDQN4.model.load_weights(save_session_name + 'mainDQN4_episode_' + str(LoadEpoch)+'.h5')
                mainDQN5.model.load_weights(save_session_name + 'mainDQN5_episode_' + str(LoadEpoch)+'.h5')
                mainDQN6.model.load_weights(save_session_name + 'mainDQN6_episode_' + str(LoadEpoch)+'.h5')
                mainDQN7.model.load_weights(save_session_name + 'mainDQN7_episode_' + str(LoadEpoch)+'.h5')
                mainDQN8.model.load_weights(save_session_name + 'mainDQN8_episode_' + str(LoadEpoch)+'.h5')
                mainDQN9.model.load_weights(save_session_name + 'mainDQN9_episode_' + str(LoadEpoch)+'.h5')

                targetDQN1.model.load_weights(save_session_name + 'targetDQN1_episode_' + str(LoadEpoch)+'.h5')
                targetDQN2.model.load_weights(save_session_name + 'targetDQN2_episode_' + str(LoadEpoch)+'.h5')
                targetDQN3.model.load_weights(save_session_name + 'targetDQN3_episode_' + str(LoadEpoch)+'.h5')
                targetDQN4.model.load_weights(save_session_name + 'targetDQN4_episode_' + str(LoadEpoch)+'.h5')
                targetDQN5.model.load_weights(save_session_name + 'targetDQN5_episode_' + str(LoadEpoch)+'.h5')
                targetDQN6.model.load_weights(save_session_name + 'targetDQN6_episode_' + str(LoadEpoch)+'.h5')
                targetDQN7.model.load_weights(save_session_name + 'targetDQN7_episode_' + str(LoadEpoch)+'.h5')
                targetDQN8.model.load_weights(save_session_name + 'targetDQN8_episode_' + str(LoadEpoch)+'.h5')
                targetDQN9.model.load_weights(save_session_name + 'targetDQN9_episode_' + str(LoadEpoch)+'.h5')
            if Start == 1:
                  # --------------------------------------load matrix and mask -------------------------------------------------------------------

                id='07'
                data_path='/home/exx/dose_deposition_full/prostate_dijs/f_dijs/0'#'/data/data/dose_deposition3/f_dijs/0'
                data_path2='/data/data/dose_deposition3/plostate_dijs/f_masks/0'
                doseMatrix_1 = loadDoseMatrix(data_path+id+'.hdf5')
                targetLabels_1, bladderLabel1, rectumLabel1, PTVLabel1 = loadMask(data_path2+id+'.h5')
                print(doseMatrix_1.shape)

                # id='08'
                # doseMatrix_2 = loadDoseMatrix(data_path+id+'.hdf5')
                # targetLabels_2, bladderLabel2, rectumLabel2, PTVLabel2 = loadMask(data_path2+id+'.h5')
                # print(doseMatrix_2.shape)

                # id='09'
                # doseMatrix_3 = loadDoseMatrix(data_path+id+'.hdf5')
                # targetLabels_3,  bladderLabel3, rectumLabel3, PTVLabel3 = loadMask(data_path2+id+'.h5')
                # print(doseMatrix_3.shape)

                # id='10'
                # doseMatrix_4 = loadDoseMatrix(data_path+id+'.hdf5')
                # targetLabels_4, bladderLabel4, rectumLabel4, PTVLabel4 = loadMask(data_path2+id+'.h5')
                # print(doseMatrix_4.shape)

                # id='11'
                # doseMatrix_5 = loadDoseMatrix(data_path+id+'.hdf5')
                # targetLabels_5, bladderLabel5, rectumLabel5, PTVLabel5 = loadMask(data_path2+id+'.h5')
                # print(doseMatrix_5.shape)

                # id='12'
                # doseMatrix_6 = loadDoseMatrix(data_path+id+'.hdf5')
                # targetLabels_6, bladderLabel6, rectumLabel6, PTVLabel6 = loadMask(data_path2+id+'.h5')
                # print(doseMatrix_6.shape)

                # id='13'
                # doseMatrix_7 = loadDoseMatrix(data_path+id+'.hdf5')
                # targetLabels_7, bladderLabel7, rectumLabel7, PTVLabel7 = loadMask(data_path2+id+'.h5')
                # print(doseMatrix_7.shape)

                # id='14'
                # doseMatrix_8 = loadDoseMatrix(data_path+id+'.hdf5')
                # targetLabels_8, bladderLabel8, rectumLabel8, PTVLabel8 = loadMask(data_path2+id+'.h5')
                # print(doseMatrix_8.shape)

                # id='15'
                # doseMatrix_9 = loadDoseMatrix(data_path+id+'.hdf5')
                # targetLabels_9, bladderLabel9, rectumLabel9, PTVLabel9 = loadMask(data_path2+id+'.h5')
                # print(doseMatrix_9.shape)

                # id='16'
                # doseMatrix_0 = loadDoseMatrix(data_path+id+'.hdf5')
                # targetLabels_0, bladderLabel0, rectumLabel0, PTVLabel0 = loadMask(data_path2+id+'.h5')
                # print(doseMatrix_0.shape)

                reward_check = zeros((MAX_EPISODES))
                q_check = zeros((MAX_EPISODES))
                loss_check = zeros((MAX_EPISODES))
                step_count1 = 0
                step_count2 = 0
                step_count3 = 0
                step_count4 = 0
                step_count5 = 0
                step_count6 = 0
                step_count7 = 0
                step_count8 = 0
                step_count9 = 0
                vali_num = 0
                ## -------------------- loop for each episode -------------------------------------------
                for episode in range(MAX_EPISODES):
                    reward_sum_total = 0
                    qvalue_sum = 0
                    loss_sum=0
                    num_q = 0
                    if load_session!=0:
                        e = 0.999 / (((episode+LoadEpoch) / 100) + 1)
                    else:
                        e = 0.999 / ((episode / 100) + 1)
                    if e < 0.1:
                        e = 0.1
                    ## -------------------- loop for each patient case  -------------------------------------------
                    for testcase in range(TRAIN_NUM):
                        logging.info('---------Training: Episode {}, Patient {}'.format(episode,testcase)+'-------------')
                        if testcase == 0:
                            doseMatrix = doseMatrix_1
                            targetLabels= targetLabels_1
                            bladderLabel =  bladderLabel1
                            rectumLabel = rectumLabel1
                            PTVLabel = PTVLabel1
                        if testcase == 1:
                            doseMatrix = doseMatrix_2
                            targetLabels= targetLabels_2
                            bladderLabel = bladderLabel2
                            rectumLabel = rectumLabel2
                            PTVLabel = PTVLabel2
                        if testcase == 2:
                            doseMatrix = doseMatrix_3
                            targetLabels= targetLabels_3
                            bladderLabel = bladderLabel3
                            rectumLabel = rectumLabel3
                            PTVLabel = PTVLabel3
                        if testcase == 3:
                            doseMatrix = doseMatrix_4
                            targetLabels= targetLabels_4
                            bladderLabel = bladderLabel4
                            rectumLabel = rectumLabel4
                            PTVLabel = PTVLabel4
                        if testcase == 4:
                            doseMatrix = doseMatrix_5
                            targetLabels= targetLabels_5
                            bladderLabel = bladderLabel5
                            rectumLabel = rectumLabel5
                            PTVLabel = PTVLabel5
                        if testcase == 5:
                            doseMatrix = doseMatrix_6
                            targetLabels = targetLabels_6
                            bladderLabel = bladderLabel6
                            rectumLabel = rectumLabel6
                            PTVLabel = PTVLabel6
                        if testcase == 6:
                            doseMatrix = doseMatrix_7
                            targetLabels = targetLabels_7
                            bladderLabel = bladderLabel7
                            rectumLabel = rectumLabel7
                            PTVLabel = PTVLabel7
                        if testcase == 7:
                            doseMatrix = doseMatrix_8
                            targetLabels = targetLabels_8
                            bladderLabel = bladderLabel8
                            rectumLabel = rectumLabel8
                            PTVLabel = PTVLabel8
                        if testcase == 8:
                            doseMatrix = doseMatrix_9
                            targetLabels = targetLabels_9
                            bladderLabel = bladderLabel9
                            rectumLabel = rectumLabel9
                            PTVLabel = PTVLabel9
                        if testcase == 9:
                            doseMatrix = doseMatrix_0
                            targetLabels = targetLabels_0
                            bladderLabel = bladderLabel0
                            rectumLabel = rectumLabel0
                            PTVLabel = PTVLabel0

                        done = False
                        step_count = 0
                        # ------------------------ initial paramaters & input --------------------
                        tPTV = 1
                        tBLA = 1
                        tREC = 1
                        lambdaPTV = 1
                        lambdaBLA = 1
                        lambdaREC = 1
                        VPTV = 0.1
                        VBLA = 1
                        VREC = 1
                        # --------------------- solve treatment planning optmization -----------------------------
                        MPTV, MBLA, MREC, MBLA1, MREC1 = ProcessDmat(doseMatrix, targetLabels,bladderLabel, rectumLabel)
                        print("MPTV.shape: MBLA.shape: MREC.shape: MBLA1.shape: MREC1.shape ",MPTV.shape, MBLA.shape,MREC.shape,MBLA1.shape, MREC1.shape)
                        xVec = np.ones((MPTV.shape[1],))
                        gamma = np.zeros((MPTV.shape[0],))
                        state, iter, xVec, gamma = \
                           runOpt_dvh(MPTV, MBLA, MREC,tPTV,tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC,
                                      VPTV, VBLA, VREC, xVec,gamma,pdose,maxiter)
                        Score_fine, Score,scoreall = planIQ(MPTV, MBLA1, MREC1, xVec,pdose)
                        logging.info('------------------- initial parameter --------------------')
                        logging.info(
                            " tPTV: {}  tBLA: {}  tREC: {}  lambdaPTV: {} lambdaBLA: {} lambdaREC: {} vPTV: {}  vBLA: {}  vREC: {} \n planScore: {} planScore_fine: {}".format(
                                tPTV,tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, Score, Score_fine))


                        ## -------------------- loop for tuning parameter  -------------------------------------------
                        while not done:
                            print('---------Training: Episode {}, Patient {}, Step {}'.format(episode+1,testcase+1,step_count+1)+'-------------')
                            tempoutput = mainDQN1.model.predict(np.reshape(state,(1,INPUT_SIZE,3)))
                            print(colored(tempoutput.shape,'red'))
                            tempoutput1 = tempoutput[0, :]
                            tempoutput = mainDQN2.model.predict(np.reshape(state,(1,INPUT_SIZE,3)))
                            tempoutput2 = tempoutput[0, :]
                            tempoutput = mainDQN3.model.predict(np.reshape(state,(1,INPUT_SIZE,3)))
                            tempoutput3 = tempoutput[0, :]
                            tempoutput = mainDQN4.model.predict(np.reshape(state,(1,INPUT_SIZE,3)))
                            tempoutput4 = tempoutput[0, :]
                            tempoutput = mainDQN5.model.predict(np.reshape(state,(1,INPUT_SIZE,3)))
                            tempoutput5 = tempoutput[0, :]
                            tempoutput = mainDQN6.model.predict(np.reshape(state,(1,INPUT_SIZE,3)))
                            tempoutput6 = tempoutput[0, :]
                            tempoutput = mainDQN7.model.predict(np.reshape(state,(1,INPUT_SIZE,3)))
                            tempoutput7 = tempoutput[0, :]
                            tempoutput = mainDQN8.model.predict(np.reshape(state,(1,INPUT_SIZE,3)))
                            tempoutput8 = tempoutput[0, :]
                            tempoutput = mainDQN9.model.predict(np.reshape(state,(1,INPUT_SIZE,3)))
                            tempoutput9 = tempoutput[0, :]
                            scorebladder = scoreall[1] + scoreall[2] + scoreall[3] + scoreall[4]
                            scorerectum = scoreall[5] + scoreall[6] + scoreall[7] + scoreall[8]
                            flagsub = np.random.rand()
                            flag = np.random.rand()
    # =============================================================================
    #                         ##---------------------------------- rule-based ------------------------------------
    #                         if flagsub<0.7*e and scoreall[0]<0.01: # ptv has a low score
    #                             dice = np.random.rand()
    #                             if dice <= 0.4:
    #                                 parasel = 3
    #                                 action = 0
    #                             else:
    #                                 parasel = 6
    #                                 action = 2
    #                         else:
    #                             if flagsub<0.7*e and scorebladder<2 and scorebladder<=scorerectum: # bla has a low score
    #                                 dice = np.random.rand()
    #                                 if dice <= 0.5:
    #                                     parasel = 1
    #                                     action = 2
    #                                 else:
    #                                     parasel = 7
    #                                     action = 2
    #                             else:
    #                                 if flagsub<0.7*e and scorerectum<2 and scorerectum<=scorebladder:
    #                                    dice = np.random.rand()
    #                                    if dice <= 0.5:
    #                                        parasel = 2
    #                                        action = 2
    #                                    else:
    #                                       parasel = 8
    #                                       action = 2
    #                                 else:
    # =============================================================================
                            if flag < e:
                                ##--------------------- random seaching but skipping nonchange action (1) ---
                                parasel = np.random.randint(9, size=1)
                                action = np.random.randint(2, size=1)
                                if action == 1:
                                    action = 2
                            else:
                                 ##--------------------- action decided by NN  ---------------------------
                                value = np.zeros((9))
                                value[0] = np.max(tempoutput1)
                                value[1] = np.max(tempoutput2)
                                value[2] = np.max(tempoutput3)
                                value[3] = np.max(tempoutput4)
                                value[4] = np.max(tempoutput5)
                                value[5] = np.max(tempoutput6)
                                value[6] = np.max(tempoutput7)
                                value[7] = np.max(tempoutput8)
                                value[8] = np.max(tempoutput9)
                                parasel = np.argmax(value)
                                if parasel == 0:
                                    action = np.argmax(tempoutput1)
                                    qvalue_sum = qvalue_sum + value[0]
                                    # loss1=history1.history['loss']
                                    # loss1=loss1[-1]
                                    # loss_sum = loss_sum+loss1
                                if parasel == 1:
                                    action = np.argmax(tempoutput2)
                                    qvalue_sum = qvalue_sum + value[1]
                                    # loss2=history2.history['loss']
                                    # loss2=loss2[-1]
                                    #loss_sum = loss_sum+loss2
                                if parasel == 2:
                                    action = np.argmax(tempoutput3)
                                    qvalue_sum = qvalue_sum + value[2]
                                    #loss3=history3.history['loss']
                                    #loss3=loss3[-1]
                                    #loss_sum = loss_sum+loss3
                                if parasel == 3:
                                    action = np.argmax(tempoutput4)
                                    qvalue_sum = qvalue_sum + value[3]
                                    #loss4=history4.history['loss']
                                    #loss4=loss4[-1]
                                    #loss_sum = loss_sum+loss4
                                if parasel == 4:
                                    action = np.argmax(tempoutput5)
                                    qvalue_sum = qvalue_sum + value[4]
                                    #loss5=history5.history['loss']
                                    #loss5=loss5[-1]
                                    #loss_sum = loss_sum+loss5
                                if parasel == 5:
                                    action = np.argmax(tempoutput6)
                                    qvalue_sum = qvalue_sum + value[5]
                                    #loss6=history6.history['loss']
                                    #loss6=loss6[-1]
                                    #loss_sum = loss_sum+loss6
                                if parasel == 6:
                                    action = np.argmax(tempoutput7)
                                    qvalue_sum = qvalue_sum + value[6]
                                    #loss7=history7.history['loss']
                                    #loss7=loss7[-1]
                                    #loss_sum = loss_sum+loss7
                                if parasel == 7:
                                    action = np.argmax(tempoutput8)
                                    qvalue_sum = qvalue_sum + value[7]
                                    #loss8=history8.history['loss']
                                    #loss8=loss8[-1]
                                    #loss_sum = loss_sum+loss8
                                if parasel == 8:
                                    action = np.argmax(tempoutput9)
                                    qvalue_sum = qvalue_sum + value[8]
                                    #loss9=history9.history['loss']
                                    #loss9=loss9[-1]
                                    #loss_sum = loss_sum+loss9
                                num_q += 1
                            ## tune parameter
                            if action == 1:
                                # unchange parameter
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
                                if parasel==8:
                                    replay_buffer9.append((state, action, reward, next_state, done))

                                # ----------------------- training NN 1 ----------------------------------------
                                if len(replay_buffer1) > BATCH_SIZE:
                                    if len(replay_buffer1) > int(REPLAY_MEMORY/10):
                                        train_num_episode = 5
                                    elif len(replay_buffer1) > int(REPLAY_MEMORY/5):
                                        train_num_episode = 20
                                    else:
                                        train_num_episode = 1
                                    for train_in_episode in range(train_num_episode):
                                        minibatch = rnd.sample(replay_buffer1, BATCH_SIZE)
                                        minibatch_states = np.reshape(np.vstack([x[0] for x in minibatch]),(BATCH_SIZE,INPUT_SIZE,3))
                                        minibatch_actions = np.vstack([x[1] for x in minibatch])
                                        minibatch_rewards = np.vstack([x[2] for x in minibatch])
                                        minibatch_next_states = np.reshape(np.vstack([x[3] for x in minibatch]),(BATCH_SIZE, INPUT_SIZE, 3))
                                        minibatch_done = np.vstack([x[4] for x in minibatch])

                                        X = minibatch_states
                                        Yout = zeros([BATCH_SIZE, OUTPUT_SIZE * 9])
                                        Yout[:, 0:3] = targetDQN1.model.predict(minibatch_next_states)
                                        Yout[:, 3:6] = targetDQN2.model.predict(minibatch_next_states)
                                        Yout[:, 6:9] = targetDQN3.model.predict(minibatch_next_states)
                                        Yout[:, 9:12] = targetDQN4.model.predict(minibatch_next_states)
                                        Yout[:, 12:15] = targetDQN5.model.predict(minibatch_next_states)
                                        Yout[:, 15:18] = targetDQN6.model.predict(minibatch_next_states)
                                        Yout[:, 18:21] = targetDQN7.model.predict(minibatch_next_states)
                                        Yout[:, 21:24] = targetDQN8.model.predict(minibatch_next_states)
                                        Yout[:, 24:27] = targetDQN9.model.predict(minibatch_next_states)
                                        Q_target = minibatch_rewards[:,0]  + DISCOUNT_RATE * np.max(Yout,axis=1) * ~minibatch_done[:,0]
                                        y = mainDQN1.model.predict(X)
                                        y[np.arange(len(X)), minibatch_actions[:,0].astype('int64')] = Q_target[np.arange(len(X))]
                                        #y[np.arange(len(X)), 1] = 0
                                        history1=mainDQN1.model.fit(X, y)
                                    step_count1 += 1
                                    if load_session==0:
                                        logging.info(
                                        "Episode: {}  steps: {} iteration_num: {} \nPlanScoreFine: {} PlanScore: {} Score_ptv: {} Score_bla:{} Score_rec: {} \naction: {}  tPTV: {}  reward: {}".format(
                                                episode + 1, step_count, iter, Score_fine, Score, scoreall[0], scorebladder, scorerectum, action, tPTV, reward))
                                    else:
                                        logging.info(
                                            "Episode: {}  steps: {} iteration_num: {} \nPlanScoreFine: {} PlanScore: {}  Score_ptv: {} Score_bla:{} Score_rec: {} \naction: {}  tPTV: {}  reward: {}".format(
                                                episode + LoadEpoch + 1, step_count, iter, Score_fine, Score,scoreall[0], scorebladder, scorerectum, action, tPTV, reward))

                                # ----------------------- training NN 2 ----------------------------------------
                                if len(replay_buffer2) > BATCH_SIZE:
                                    if len(replay_buffer2) > int(REPLAY_MEMORY/10):
                                        train_num_episode = 5
                                    elif len(replay_buffer2) > int(REPLAY_MEMORY/5):
                                        train_num_episode = 20
                                    else:
                                        train_num_episode = 1
                                    for train_in_episode in range(train_num_episode):
                                        minibatch = rnd.sample(replay_buffer2, BATCH_SIZE)
                                        minibatch_states = np.reshape(np.vstack([x[0] for x in minibatch]),
                                                                      (BATCH_SIZE, INPUT_SIZE, 3))
                                        minibatch_actions = np.vstack([x[1] for x in minibatch])
                                        minibatch_rewards = np.vstack([x[2] for x in minibatch])
                                        minibatch_next_states = np.reshape(np.vstack([x[3] for x in minibatch]),
                                                                           (BATCH_SIZE, INPUT_SIZE,3))
                                        minibatch_done = np.vstack([x[4] for x in minibatch])
                                        X = minibatch_states
                                        Yout = zeros([BATCH_SIZE, OUTPUT_SIZE * 9])
                                        Yout[:, 0:3] = targetDQN1.model.predict(minibatch_next_states)
                                        Yout[:, 3:6] = targetDQN2.model.predict(minibatch_next_states)
                                        Yout[:, 6:9] = targetDQN3.model.predict(minibatch_next_states)
                                        Yout[:, 9:12] = targetDQN4.model.predict(minibatch_next_states)
                                        Yout[:, 12:15] = targetDQN5.model.predict(minibatch_next_states)
                                        Yout[:, 15:18] = targetDQN6.model.predict(minibatch_next_states)
                                        Yout[:, 18:21] = targetDQN7.model.predict(minibatch_next_states)
                                        Yout[:, 21:24] = targetDQN8.model.predict(minibatch_next_states)
                                        Yout[:, 24:27] = targetDQN9.model.predict(minibatch_next_states)
                                        Q_target = minibatch_rewards[:,0]  + DISCOUNT_RATE * np.max(Yout,axis=1) * ~minibatch_done[:,0]
                                        y = mainDQN2.model.predict(X)
                                        y[np.arange(len(X)), minibatch_actions[:,0].astype('int64')] = Q_target[np.arange(len(X))]
                                        #y[np.arange(len(X)), 1] = 0
                                        history2=mainDQN2.model.fit(X, y)
                                    step_count2 += 1
                                    if load_session==0:
                                        logging.info(
                                        "Episode: {}  steps: {} iteration_num: {} \nPlanScoreFine: {} PlanScore: {} Score_ptv: {} Score_bla:{} Score_rec: {} \naction: {}  tPTV: {}  reward: {}".format(
                                                episode + 1, step_count, iter, Score_fine, Score, scoreall[0], scorebladder, scorerectum, action, tPTV, reward))
                                    else:
                                        logging.info(
                                            "Episode: {}  steps: {} iteration_num: {} \nPlanScoreFine: {} PlanScore: {}  Score_ptv: {} Score_bla:{} Score_rec: {} \naction: {}  tPTV: {}  reward: {}".format(
                                                episode + LoadEpoch + 1, step_count, iter, Score_fine, Score,scoreall[0], scorebladder, scorerectum, action, tPTV, reward))

                                # ----------------------- training NN 3 ----------------------------------------
                                if len(replay_buffer3) > BATCH_SIZE:
                                    if len(replay_buffer3) > int(REPLAY_MEMORY/10):
                                        train_num_episode = 5
                                    elif len(replay_buffer3) > int(REPLAY_MEMORY/5):
                                        train_num_episode = 20
                                    else:
                                        train_num_episode = 1
                                    for train_in_episode in range(train_num_episode):
                                        minibatch = rnd.sample(replay_buffer3, BATCH_SIZE)
                                        minibatch_states = np.reshape(np.vstack([x[0] for x in minibatch]),
                                                                      (BATCH_SIZE, INPUT_SIZE, 3))
                                        minibatch_actions = np.vstack([x[1] for x in minibatch])
                                        minibatch_rewards = np.vstack([x[2] for x in minibatch])
                                        minibatch_next_states = np.reshape(np.vstack([x[3] for x in minibatch]),
                                                                           (BATCH_SIZE, INPUT_SIZE, 3))
                                        minibatch_done = np.vstack([x[4] for x in minibatch])
                                        X = minibatch_states
                                        Yout = zeros([BATCH_SIZE, OUTPUT_SIZE * 9])
                                        Yout[:, 0:3] = targetDQN1.model.predict(minibatch_next_states)
                                        Yout[:, 3:6] = targetDQN2.model.predict(minibatch_next_states)
                                        Yout[:, 6:9] = targetDQN3.model.predict(minibatch_next_states)
                                        Yout[:, 9:12] = targetDQN4.model.predict(minibatch_next_states)
                                        Yout[:, 12:15] = targetDQN5.model.predict(minibatch_next_states)
                                        Yout[:, 15:18] = targetDQN6.model.predict(minibatch_next_states)
                                        Yout[:, 18:21] = targetDQN7.model.predict(minibatch_next_states)
                                        Yout[:, 21:24] = targetDQN8.model.predict(minibatch_next_states)
                                        Yout[:, 24:27] = targetDQN9.model.predict(minibatch_next_states)
                                        Q_target = minibatch_rewards[:,0]  + DISCOUNT_RATE * np.max(Yout,axis=1) * ~minibatch_done[:,0]
                                        y = mainDQN3.model.predict(X)
                                        y[np.arange(len(X)), minibatch_actions[:,0].astype('int64')] = Q_target[np.arange(len(X))]
                                        #y[np.arange(len(X)), 1] = 0
                                        history3=mainDQN3.model.fit(X, y)
                                    step_count3 += 1
                                    if load_session==0:
                                        logging.info(
                                        "Episode: {}  steps: {} iteration_num: {} \nPlanScoreFine: {} PlanScore: {} Score_ptv: {} Score_bla:{} Score_rec: {} \naction: {}  tPTV: {}  reward: {}".format(
                                                episode + 1, step_count, iter, Score_fine, Score, scoreall[0], scorebladder, scorerectum, action, tPTV, reward))
                                    else:
                                        logging.info(
                                            "Episode: {}  steps: {} iteration_num: {} \nPlanScoreFine: {} PlanScore: {}  Score_ptv: {} Score_bla:{} Score_rec: {} \naction: {}  tPTV: {}  reward: {}".format(
                                                episode + LoadEpoch + 1, step_count, iter, Score_fine, Score,scoreall[0], scorebladder, scorerectum, action, tPTV, reward))

                                # ----------------------- training NN 4 ----------------------------------------
                                if len(replay_buffer4) > BATCH_SIZE:
                                    if len(replay_buffer4) > int(REPLAY_MEMORY/10):
                                        train_num_episode = 5
                                    elif len(replay_buffer4) > int(REPLAY_MEMORY/5):
                                        train_num_episode = 20
                                    else:
                                        train_num_episode = 1
                                    for train_in_episode in range(train_num_episode):
                                        minibatch = rnd.sample(replay_buffer4, BATCH_SIZE)
                                        minibatch_states = np.reshape(np.vstack([x[0] for x in minibatch]),
                                                                      (BATCH_SIZE, INPUT_SIZE,3))
                                        minibatch_actions = np.vstack([x[1] for x in minibatch])
                                        minibatch_rewards = np.vstack([x[2] for x in minibatch])
                                        minibatch_next_states = np.reshape(np.vstack([x[3] for x in minibatch]),
                                                                           (BATCH_SIZE, INPUT_SIZE, 3))
                                        minibatch_done = np.vstack([x[4] for x in minibatch])
                                        X = minibatch_states
                                        Yout = zeros([BATCH_SIZE, OUTPUT_SIZE *9])
                                        Yout[:, 0:3] = targetDQN1.model.predict(minibatch_next_states)
                                        Yout[:, 3:6] = targetDQN2.model.predict(minibatch_next_states)
                                        Yout[:, 6:9] = targetDQN3.model.predict(minibatch_next_states)
                                        Yout[:, 9:12] = targetDQN4.model.predict(minibatch_next_states)
                                        Yout[:, 12:15] = targetDQN5.model.predict(minibatch_next_states)
                                        Yout[:, 15:18] = targetDQN6.model.predict(minibatch_next_states)
                                        Yout[:, 18:21] = targetDQN7.model.predict(minibatch_next_states)
                                        Yout[:, 21:24] = targetDQN8.model.predict(minibatch_next_states)
                                        Yout[:, 24:27] = targetDQN9.model.predict(minibatch_next_states)
                                        Q_target = minibatch_rewards[:,0]  + DISCOUNT_RATE * np.max(Yout,axis=1) * ~minibatch_done[:,0]
                                        y = mainDQN4.model.predict(X)
                                        y[np.arange(len(X)), minibatch_actions[:,0].astype('int64')] = Q_target[np.arange(len(X))]
                                        #y[np.arange(len(X)), 1] = 0
                                        history4=mainDQN4.model.fit(X, y)
                                    step_count4 += 1
                                    if load_session==0:
                                        logging.info(
                                        "Episode: {}  steps: {} iteration_num: {} \nPlanScoreFine: {} PlanScore: {} Score_ptv: {} Score_bla:{} Score_rec: {} \naction: {}  tPTV: {}  reward: {}".format(
                                                episode + 1, step_count, iter, Score_fine, Score, scoreall[0], scorebladder, scorerectum, action, tPTV, reward))
                                    else:
                                        logging.info(
                                            "Episode: {}  steps: {} iteration_num: {} \nPlanScoreFine: {} PlanScore: {}  Score_ptv: {} Score_bla:{} Score_rec: {} \naction: {}  tPTV: {}  reward: {}".format(
                                                episode + LoadEpoch + 1, step_count, iter, Score_fine, Score,scoreall[0], scorebladder, scorerectum, action, tPTV, reward))

                                # ----------------------- training NN 5 ----------------------------------------
                                if len(replay_buffer5) > BATCH_SIZE:
                                    if len(replay_buffer5) > int(REPLAY_MEMORY/10):
                                        train_num_episode = 5
                                    elif len(replay_buffer5) > int(REPLAY_MEMORY/5):
                                        train_num_episode = 20
                                    else:
                                        train_num_episode = 1
                                    for train_in_episode in range(train_num_episode):
                                        minibatch = rnd.sample(replay_buffer5, BATCH_SIZE)
                                        minibatch_states = np.reshape(np.vstack([x[0] for x in minibatch]),
                                                                      (BATCH_SIZE, INPUT_SIZE, 3))
                                        minibatch_actions = np.vstack([x[1] for x in minibatch])
                                        minibatch_rewards = np.vstack([x[2] for x in minibatch])
                                        minibatch_next_states = np.reshape(np.vstack([x[3] for x in minibatch]),
                                                                           (BATCH_SIZE, INPUT_SIZE, 3))
                                        minibatch_done = np.vstack([x[4] for x in minibatch])
                                        X = minibatch_states
                                        Yout = zeros([BATCH_SIZE, OUTPUT_SIZE * 9])
                                        Yout[:, 0:3] = targetDQN1.model.predict(minibatch_next_states)
                                        Yout[:, 3:6] = targetDQN2.model.predict(minibatch_next_states)
                                        Yout[:, 6:9] = targetDQN3.model.predict(minibatch_next_states)
                                        Yout[:, 9:12] = targetDQN4.model.predict(minibatch_next_states)
                                        Yout[:, 12:15] = targetDQN5.model.predict(minibatch_next_states)
                                        Yout[:, 15:18] = targetDQN6.model.predict(minibatch_next_states)
                                        Yout[:, 18:21] = targetDQN7.model.predict(minibatch_next_states)
                                        Yout[:, 21:24] = targetDQN8.model.predict(minibatch_next_states)
                                        Yout[:, 24:27] = targetDQN9.model.predict(minibatch_next_states)
                                        Q_target = minibatch_rewards[:,0]  + DISCOUNT_RATE * np.max(Yout,axis=1) * ~minibatch_done[:,0]
                                        y = mainDQN5.model.predict(X)
                                        y[np.arange(len(X)), minibatch_actions[:,0].astype('int64')] = Q_target[np.arange(len(X))]
                                        #y[np.arange(len(X)), 1] = 0
                                        history5=mainDQN5.model.fit(X, y)
                                    step_count5 += 1
                                    if load_session==0:
                                        logging.info(
                                        "Episode: {}  steps: {} iteration_num: {} \nPlanScoreFine: {} PlanScore: {} Score_ptv: {} Score_bla:{} Score_rec: {} \naction: {}  tPTV: {}  reward: {}".format(
                                                episode + 1, step_count, iter, Score_fine, Score, scoreall[0], scorebladder, scorerectum, action, tPTV, reward))
                                    else:
                                        logging.info(
                                            "Episode: {}  steps: {} iteration_num: {} \nPlanScoreFine: {} PlanScore: {}  Score_ptv: {} Score_bla:{} Score_rec: {} \naction: {}  tPTV: {}  reward: {}".format(
                                                episode + LoadEpoch + 1, step_count, iter, Score_fine, Score,scoreall[0], scorebladder, scorerectum, action, tPTV, reward))

                                # ----------------------- training NN 6 ----------------------------------------
                                if len(replay_buffer6) > BATCH_SIZE:
                                    if len(replay_buffer6) > int(REPLAY_MEMORY/10):
                                        train_num_episode = 5
                                    elif len(replay_buffer6) > int(REPLAY_MEMORY/5):
                                        train_num_episode = 20
                                    else:
                                        train_num_episode = 1
                                    for train_in_episode in range(train_num_episode):
                                        minibatch = rnd.sample(replay_buffer6, BATCH_SIZE)
                                        minibatch_states = np.reshape(np.vstack([x[0] for x in minibatch]),
                                                                      (BATCH_SIZE, INPUT_SIZE, 3))
                                        minibatch_actions = np.vstack([x[1] for x in minibatch])
                                        minibatch_rewards = np.vstack([x[2] for x in minibatch])
                                        minibatch_next_states = np.reshape(np.vstack([x[3] for x in minibatch]),
                                                                           (BATCH_SIZE,  INPUT_SIZE, 3))
                                        minibatch_done = np.vstack([x[4] for x in minibatch])
                                        X = minibatch_states
                                        Yout = zeros([BATCH_SIZE, OUTPUT_SIZE * 9])
                                        Yout[:, 0:3] = targetDQN1.model.predict(minibatch_next_states)
                                        Yout[:, 3:6] = targetDQN2.model.predict(minibatch_next_states)
                                        Yout[:, 6:9] = targetDQN3.model.predict(minibatch_next_states)
                                        Yout[:, 9:12] = targetDQN4.model.predict(minibatch_next_states)
                                        Yout[:, 12:15] = targetDQN5.model.predict(minibatch_next_states)
                                        Yout[:, 15:18] = targetDQN6.model.predict(minibatch_next_states)
                                        Yout[:, 18:21] = targetDQN7.model.predict(minibatch_next_states)
                                        Yout[:, 21:24] = targetDQN8.model.predict(minibatch_next_states)
                                        Yout[:, 24:27] = targetDQN9.model.predict(minibatch_next_states)
                                        Q_target = minibatch_rewards[:,0]  + DISCOUNT_RATE * np.max(Yout,axis=1) * ~minibatch_done[:,0]
                                        y = mainDQN6.model.predict(X)
                                        y[np.arange(len(X)), minibatch_actions[:,0].astype('int64')] = Q_target[np.arange(len(X))]
                                        #y[np.arange(len(X)), 1] = 0
                                        history6=mainDQN6.model.fit(X, y)
                                    step_count6 += 1
                                    if load_session==0:
                                        logging.info(
                                        "Episode: {}  steps: {} iteration_num: {} \nPlanScoreFine: {} PlanScore: {} Score_ptv: {} Score_bla:{} Score_rec: {} \naction: {}  tPTV: {}  reward: {}".format(
                                                episode + 1, step_count, iter, Score_fine, Score, scoreall[0], scorebladder, scorerectum, action, tPTV, reward))
                                    else:
                                        logging.info(
                                            "Episode: {}  steps: {} iteration_num: {} \nPlanScoreFine: {} PlanScore: {}  Score_ptv: {} Score_bla:{} Score_rec: {} \naction: {}  tPTV: {}  reward: {}".format(
                                                episode + LoadEpoch + 1, step_count, iter, Score_fine, Score,scoreall[0], scorebladder, scorerectum, action, tPTV, reward))

                                # ----------------------- training NN 7 ----------------------------------------
                                if len(replay_buffer7) > BATCH_SIZE:
                                    if len(replay_buffer7) > int(REPLAY_MEMORY/10):
                                        train_num_episode = 5
                                    elif len(replay_buffer7) > int(REPLAY_MEMORY/5):
                                        train_num_episode = 20
                                    else:
                                        train_num_episode = 1
                                    for train_in_episode in range(train_num_episode):
                                        minibatch = rnd.sample(replay_buffer7, BATCH_SIZE)
                                        minibatch_states = np.reshape(np.vstack([x[0] for x in minibatch]),
                                                                      (BATCH_SIZE, INPUT_SIZE, 3))
                                        minibatch_actions = np.vstack([x[1] for x in minibatch])
                                        minibatch_rewards = np.vstack([x[2] for x in minibatch])
                                        minibatch_next_states = np.reshape(np.vstack([x[3] for x in minibatch]),
                                                                           (BATCH_SIZE, INPUT_SIZE, 3))
                                        minibatch_done = np.vstack([x[4] for x in minibatch])
                                        X = minibatch_states
                                        Yout = zeros([BATCH_SIZE, OUTPUT_SIZE * 9])
                                        Yout[:, 0:3] = targetDQN1.model.predict(minibatch_next_states)
                                        Yout[:, 3:6] = targetDQN2.model.predict(minibatch_next_states)
                                        Yout[:, 6:9] = targetDQN3.model.predict(minibatch_next_states)
                                        Yout[:, 9:12] = targetDQN4.model.predict(minibatch_next_states)
                                        Yout[:, 12:15] = targetDQN5.model.predict(minibatch_next_states)
                                        Yout[:, 15:18] = targetDQN6.model.predict(minibatch_next_states)
                                        Yout[:, 18:21] = targetDQN7.model.predict(minibatch_next_states)
                                        Yout[:, 21:24] = targetDQN8.model.predict(minibatch_next_states)
                                        Yout[:, 24:27] = targetDQN9.model.predict(minibatch_next_states)
                                        Q_target = minibatch_rewards[:,0]  + DISCOUNT_RATE * np.max(Yout,axis=1) * ~minibatch_done[:,0]
                                        y = mainDQN7.model.predict(X)
                                        y[np.arange(len(X)), minibatch_actions[:,0].astype('int64')] = Q_target[np.arange(len(X))]
                                        #y[np.arange(len(X)), 1] = 0
                                        history7=mainDQN7.model.fit(X, y)
                                    step_count7 += 1
                                    if load_session==0:
                                        logging.info(
                                        "Episode: {}  steps: {} iteration_num: {} \nPlanScoreFine: {} PlanScore: {} Score_ptv: {} Score_bla:{} Score_rec: {} \naction: {}  tPTV: {}  reward: {}".format(
                                                episode + 1, step_count, iter, Score_fine, Score, scoreall[0], scorebladder, scorerectum, action, tPTV, reward))
                                    else:
                                        logging.info(
                                            "Episode: {}  steps: {} iteration_num: {} \nPlanScoreFine: {} PlanScore: {}  Score_ptv: {} Score_bla:{} Score_rec: {} \naction: {}  tPTV: {}  reward: {}".format(
                                                episode + LoadEpoch + 1, step_count, iter, Score_fine, Score,scoreall[0], scorebladder, scorerectum, action, tPTV, reward))

                                # ----------------------- training NN 8 ----------------------------------------
                                if len(replay_buffer8) > BATCH_SIZE:
                                    if len(replay_buffer8) > int(REPLAY_MEMORY/10):
                                        train_num_episode = 5
                                    elif len(replay_buffer8) > int(REPLAY_MEMORY/5):
                                        train_num_episode = 20
                                    else:
                                        train_num_episode = 1
                                    for train_in_episode in range(train_num_episode):
                                        minibatch = rnd.sample(replay_buffer8, BATCH_SIZE)
                                        minibatch_states = np.reshape(np.vstack([x[0] for x in minibatch]),
                                                                      (BATCH_SIZE, INPUT_SIZE, 3))
                                        minibatch_actions = np.vstack([x[1] for x in minibatch])
                                        minibatch_rewards = np.vstack([x[2] for x in minibatch])
                                        minibatch_next_states = np.reshape(np.vstack([x[3] for x in minibatch]),
                                                                           (BATCH_SIZE, INPUT_SIZE, 3))
                                        minibatch_done = np.vstack([x[4] for x in minibatch])
                                        X = minibatch_states
                                        Yout = zeros([BATCH_SIZE, OUTPUT_SIZE * 9])
                                        Yout[:, 0:3] = targetDQN1.model.predict(minibatch_next_states)
                                        Yout[:, 3:6] = targetDQN2.model.predict(minibatch_next_states)
                                        Yout[:, 6:9] = targetDQN3.model.predict(minibatch_next_states)
                                        Yout[:, 9:12] = targetDQN4.model.predict(minibatch_next_states)
                                        Yout[:, 12:15] = targetDQN5.model.predict(minibatch_next_states)
                                        Yout[:, 15:18] = targetDQN6.model.predict(minibatch_next_states)
                                        Yout[:, 18:21] = targetDQN7.model.predict(minibatch_next_states)
                                        Yout[:, 21:24] = targetDQN8.model.predict(minibatch_next_states)
                                        Yout[:, 24:27] = targetDQN9.model.predict(minibatch_next_states)
                                        Q_target = minibatch_rewards[:,0]  + DISCOUNT_RATE * np.max(Yout,axis=1) * ~minibatch_done[:,0]
                                        y = mainDQN8.model.predict(X)
                                        y[np.arange(len(X)), minibatch_actions[:,0].astype('int64')] = Q_target[np.arange(len(X))]
                                        #y[np.arange(len(X)), 1] = 0
                                        history8=mainDQN8.model.fit(X, y)
                                    step_count8 += 1
                                    if load_session==0:
                                        logging.info(
                                        "Episode: {}  steps: {} iteration_num: {} \nPlanScoreFine: {} PlanScore: {} Score_ptv: {} Score_bla:{} Score_rec: {} \naction: {}  tPTV: {}  reward: {}".format(
                                                episode + 1, step_count, iter, Score_fine, Score, scoreall[0], scorebladder, scorerectum, action, tPTV, reward))
                                    else:
                                        logging.info(
                                            "Episode: {}  steps: {} iteration_num: {} \nPlanScoreFine: {} PlanScore: {}  Score_ptv: {} Score_bla:{} Score_rec: {} \naction: {}  tPTV: {}  reward: {}".format(
                                                episode + LoadEpoch + 1, step_count, iter, Score_fine, Score,scoreall[0], scorebladder, scorerectum, action, tPTV, reward))

                                # ----------------------- training NN 9 ----------------------------------------
                                if len(replay_buffer9) > BATCH_SIZE:
                                    if len(replay_buffer9) > int(REPLAY_MEMORY/10):
                                        train_num_episode = 5
                                    elif len(replay_buffer9) > int(REPLAY_MEMORY/5):
                                        train_num_episode = 20
                                    else:
                                        train_num_episode = 1
                                    for train_in_episode in range(train_num_episode):
                                        minibatch = rnd.sample(replay_buffer9, BATCH_SIZE)
                                        minibatch_states = np.reshape(np.vstack([x[0] for x in minibatch]),
                                                                      (BATCH_SIZE, INPUT_SIZE, 3))
                                        minibatch_actions = np.vstack([x[1] for x in minibatch])
                                        minibatch_rewards = np.vstack([x[2] for x in minibatch])
                                        minibatch_next_states = np.reshape(np.vstack([x[3] for x in minibatch]),
                                                                           (BATCH_SIZE, INPUT_SIZE, 3))
                                        minibatch_done = np.vstack([x[4] for x in minibatch])

                                        X = minibatch_states
                                        Yout = zeros([BATCH_SIZE, OUTPUT_SIZE * 9])
                                        Yout[:, 0:3] = targetDQN1.model.predict(minibatch_next_states)
                                        Yout[:, 3:6] = targetDQN2.model.predict(minibatch_next_states)
                                        Yout[:, 6:9] = targetDQN3.model.predict(minibatch_next_states)
                                        Yout[:, 9:12] = targetDQN4.model.predict(minibatch_next_states)
                                        Yout[:, 12:15] = targetDQN5.model.predict(minibatch_next_states)
                                        Yout[:, 15:18] = targetDQN6.model.predict(minibatch_next_states)
                                        Yout[:, 18:21] = targetDQN7.model.predict(minibatch_next_states)
                                        Yout[:, 21:24] = targetDQN8.model.predict(minibatch_next_states)
                                        Yout[:, 24:27] = targetDQN9.model.predict(minibatch_next_states)
                                        Q_target = minibatch_rewards[:,0]  + DISCOUNT_RATE * np.max(Yout,axis=1) * ~minibatch_done[:,0]
                                        y = mainDQN9.model.predict(X)
                                        y[np.arange(len(X)), minibatch_actions[:,0].astype('int64')] = Q_target[np.arange(len(X))]
                                        #y[np.arange(len(X)), 1] = 0
                                        history9=mainDQN9.model.fit(X, y)
                                    step_count9 += 1
                                    if load_session==0:
                                        logging.info(
                                        "Episode: {}  steps: {} iteration_num: {} \nPlanScoreFine: {} PlanScore: {} Score_ptv: {} Score_bla:{} Score_rec: {} \naction: {}  tPTV: {}  reward: {}".format(
                                                episode + 1, step_count, iter, Score_fine, Score, scoreall[0], scorebladder, scorerectum, action, tPTV, reward))
                                    else:
                                        logging.info(
                                            "Episode: {}  steps: {} iteration_num: {} \nPlanScoreFine: {} PlanScore: {}  Score_ptv: {} Score_bla:{} Score_rec: {} \naction: {}  tPTV: {}  reward: {}".format(
                                                episode + LoadEpoch + 1, step_count, iter, Score_fine, Score,scoreall[0], scorebladder, scorerectum, action, tPTV, reward))

                                steps = [step_count1,step_count2,step_count3,step_count4,step_count5,step_count6,step_count7,step_count8,step_count9]
                                if np.min(steps) % TARGET_UPDATE_FREQUENCY == 0:
                                    targetDQN1.model.set_weights(mainDQN1.model.get_weights())
                                    targetDQN2.model.set_weights(mainDQN2.model.get_weights())
                                    targetDQN3.model.set_weights(mainDQN3.model.get_weights())
                                    targetDQN4.model.set_weights(mainDQN4.model.get_weights())
                                    targetDQN5.model.set_weights(mainDQN5.model.get_weights())
                                    targetDQN6.model.set_weights(mainDQN6.model.get_weights())
                                    targetDQN7.model.set_weights(mainDQN7.model.get_weights())
                                    targetDQN8.model.set_weights(mainDQN8.model.get_weights())
                                    targetDQN9.model.set_weights(mainDQN9.model.get_weights())

                                continue


                            if action != 1:
                                # adjust the parameter
                                if parasel == 0:
                                    if action == 0:
                                        action_factor = 1.01
                                        tPTV = tPTV * action_factor
                                        if tPTV >= paraMax_tPTV:
                                            tPTV = paraMax_tPTV
                                    elif action == 1:
                                        action_factor = 1
                                        tPTV = tPTV *  action_factor
                                        if tPTV >= paraMax_tPTV:
                                            tPTV = paraMax_tPTV
                                    else:
                                        action_factor = 0.09
                                        tPTV = tPTV * action_factor
                                        if tPTV <= paraMin_tPTV:
                                            tPTV = paraMin_tPTV

                                if parasel == 1:
                                    if action == 0:
                                        action_factor = 1.25
                                        tBLA = tBLA * action_factor
                                        if tBLA >= paraMax_tOAR:
                                            tBLA = paraMax_tOAR
                                    elif action == 1:
                                        action_factor = 1
                                        tBLA = tBLA * action_factor
                                        if tBLA >= paraMax_tOAR:
                                            tBLA = paraMax_tOAR
                                    else:
                                        action_factor = 0.8
                                        tBLA = tBLA * action_factor
                                        if tBLA <= paraMin:
                                            tBLA = paraMin

                                if parasel == 2:
                                    if action == 0:
                                        action_factor = 1.25
                                        tREC = tREC * action_factor
                                        if tREC >= paraMax_tOAR:
                                            tREC = paraMax_tOAR
                                    elif action == 1:
                                        action_factor = 1
                                        tREC = tREC * action_factor
                                        if tREC >= paraMax_tOAR:
                                            tREC = paraMax_tOAR
                                    else:
                                        action_factor = 0.8
                                        tREC = tREC * action_factor
                                        if tREC <= paraMin:
                                            tREC = paraMin

                                if parasel == 3:
                                    if action == 0:
                                        action_factor = m.exp(0.5)
                                        lambdaPTV = lambdaPTV * action_factor
                                        if lambdaPTV >= paraMax:
                                            lambdaPTV = paraMax
                                    elif action == 1:
                                        action_factor = 1
                                        lambdaPTV = lambdaPTV * action_factor
                                        if lambdaPTV >= paraMax:
                                            lambdaPTV = paraMax
                                    else:
                                        action_factor = m.exp(-0.5)
                                        lambdaPTV = lambdaPTV * action_factor
                                        if lambdaPTV <= paraMin:
                                            lambdaPTV = paraMin

                                if parasel == 4:
                                    if action == 0:
                                        action_factor = m.exp(0.5)
                                        lambdaBLA = lambdaBLA * action_factor
                                        if lambdaBLA >= paraMax:
                                            lambdaBLA = paraMax
                                    elif action == 1:
                                        action_factor = 1
                                        lambdaBLA = lambdaBLA * action_factor
                                        if lambdaBLA >= paraMax:
                                            lambdaBLA = paraMax
                                    else:
                                        action_factor = m.exp(-0.5)
                                        lambdaBLA = lambdaBLA * action_factor
                                        if lambdaBLA <= paraMin:
                                            lambdaBLA = paraMin
                                if parasel == 5:
                                    if action == 0:
                                        action_factor = m.exp(0.5)
                                        lambdaREC = lambdaREC * action_factor
                                        if lambdaREC >= paraMax:
                                            lambdaREC = paraMax
                                    elif action == 1:
                                        action_factor = 1
                                        lambdaREC = lambdaREC * action_factor
                                        if lambdaREC >= paraMax:
                                            lambdaREC = paraMax
                                    else:
                                        action_factor = m.exp(-0.5)
                                        lambdaREC = lambdaREC* action_factor
                                        if lambdaREC <= paraMin:
                                            lambdaREC = paraMin
                                if parasel == 6:
                                    if action == 0:
                                        action_factor = 1.4
                                        VPTV = VPTV * action_factor
                                        if VPTV >= paraMax_VPTV:
                                            VPTV = paraMax_VPTV
                                    elif action == 1:
                                        action_factor = 1
                                        VPTV = VPTV * action_factor
                                        if VPTV >= paraMax:
                                            VPTV = paraMax
                                    else:
                                        action_factor = 0.6
                                        VPTV = VPTV * action_factor
                                        if VPTV <= paraMin:
                                            VPTV = paraMin

                                if parasel == 7:
                                    if action == 0:
                                        action_factor = 1.25
                                        VBLA = VBLA * action_factor
                                        if VBLA >= paraMax_VOAR:
                                            VBLA = paraMax_VOAR
                                    elif action == 1:
                                        action_factor = 1
                                        VBLA = VBLA * action_factor
                                        if VBLA >= paraMax_VOAR:
                                            VBLA = paraMax_VOAR
                                    else:
                                        action_factor = 0.8
                                        VBLA = VBLA * action_factor
                                        if VBLA <= paraMin:
                                            VBLA = paraMin
                                if parasel == 8:
                                    if action == 0:
                                        action_factor = 1.25
                                        VREC = VREC * action_factor
                                        if VREC >= paraMax_VOAR:
                                            VREC = paraMax_VOAR
                                    elif action == 1:
                                        action_factor = 1
                                        VREC = VREC * action_factor
                                        if VREC >= paraMax_VOAR:
                                            VREC = paraMax_VOAR
                                    else:
                                        action_factor = 0.8
                                        VREC = VREC * action_factor
                                        if VREC <= paraMin:
                                            VREC = paraMin


                                # ------------------- treatmentplanning optimization -----------------------------------
                                xVec = np.ones((MPTV.shape[1],))
                                gamma = np.zeros((MPTV.shape[0],))
                                next_state, iter, xVec, gamma = \
                                    runOpt_dvh(MPTV, MBLA, MREC,tPTV,tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, xVec,gamma,pdose,maxiter)
                                Score_fine1, Score1,scoreall = planIQ(MPTV, MBLA1, MREC1, xVec,pdose)

                                reward = Score_fine1-Score_fine+(Score1-Score)*4
                                Score_fine = Score_fine1
                                Score = Score1
                                scorebladder = scoreall[1] + scoreall[2] + scoreall[3] + scoreall[4]
                                scorerectum = scoreall[5] + scoreall[6] + scoreall[7] + scoreall[8]
                                logging.info('------------------- update parameter in step {}'.format(step_count+1)+'-----------------------------')
                                logging.info(
                                    "tPTV: {}  tBLA: {}  tREC: {}  lambdaPTV: {} lambdaBLA: {} lambdaREC: {} vPTV: {}  vBLA: {}  vREC: {} \nplanScore: {} planScore_fine: {} Score_ptv: {} Score_bla:{} Score_rec: {}".format(
                                        round(tPTV,2),round(tBLA,2), round(tREC,2),
                                        round(lambdaPTV,2), round(lambdaBLA,2), round(lambdaREC,2),
                                        round(VPTV,4), round(VBLA,4), round(VREC,4),
                                        round(Score,3), round(Score_fine,3),round(scoreall[0],3), round(scorebladder,3), round(scorerectum,3)))

                                # ---------- stop criterion ----------------------------
                                if step_count >= MAX_STEP - 1:
                                    done = True
                                if Score_fine > 12.5 and Score == 9:
                                    done = True
                                step_count += 1
                                # ------------------------------------------------------

                                if flag >= e:
                                    futureReward = zeros((9))
                                    futureReward[0] = np.max(targetDQN1.model.predict(np.reshape(state,(1,INPUT_SIZE,3))))
                                    futureReward[1] = np.max(targetDQN2.model.predict(np.reshape(state,(1,INPUT_SIZE,3))))
                                    futureReward[2] = np.max(targetDQN3.model.predict(np.reshape(state,(1,INPUT_SIZE,3))))
                                    futureReward[3] = np.max(targetDQN4.model.predict(np.reshape(state,(1,INPUT_SIZE,3))))
                                    futureReward[4] = np.max(targetDQN5.model.predict(np.reshape(state,(1,INPUT_SIZE,3))))
                                    futureReward[5] = np.max(targetDQN6.model.predict(np.reshape(state,(1,INPUT_SIZE,3))))
                                    futureReward[6] = np.max(targetDQN7.model.predict(np.reshape(state,(1,INPUT_SIZE,3))))
                                    futureReward[7] = np.max(targetDQN8.model.predict(np.reshape(state,(1,INPUT_SIZE,3))))
                                    futureReward[8] = np.max(targetDQN9.model.predict(np.reshape(state,(1,INPUT_SIZE,3))))
                                    # futureReward[5] = np.max(targetDQN6.predict(next_state))
                                    if done != True:
                                        reward_sum_total = reward_sum_total + reward + DISCOUNT_RATE * np.max(futureReward)
                                    else:
                                        reward_sum_total = reward_sum_total + reward


                                if parasel == 0:
                                    replay_buffer1.append((state, action, reward, next_state, done))
                                # ----------------------- training NN 1 ----------------------------------------
                                if len(replay_buffer1) > BATCH_SIZE:
                                    if len(replay_buffer1) > int(REPLAY_MEMORY/10):
                                        train_num_episode = 5
                                    elif len(replay_buffer1) > int(REPLAY_MEMORY/5):
                                        train_num_episode = 20
                                    else:
                                        train_num_episode = 1
                                    for train_in_episode in range(train_num_episode):
                                        minibatch = rnd.sample(replay_buffer1, BATCH_SIZE)
                                        minibatch_states = np.reshape(np.vstack([x[0] for x in minibatch]),
                                                                      (BATCH_SIZE, INPUT_SIZE, 3))
                                        minibatch_actions = np.vstack([x[1] for x in minibatch])
                                        minibatch_rewards = np.vstack([x[2] for x in minibatch])
                                        minibatch_next_states = np.reshape(np.vstack([x[3] for x in minibatch]),
                                                                           (BATCH_SIZE, INPUT_SIZE, 3))
                                        minibatch_done = np.vstack([x[4] for x in minibatch])
                                        X = minibatch_states
                                        Yout = zeros([BATCH_SIZE, OUTPUT_SIZE * 9])
                                        Yout[:, 0:3] = targetDQN1.model.predict(minibatch_next_states)
                                        Yout[:, 3:6] = targetDQN2.model.predict(minibatch_next_states)
                                        Yout[:, 6:9] = targetDQN3.model.predict(minibatch_next_states)
                                        Yout[:, 9:12] = targetDQN4.model.predict(minibatch_next_states)
                                        Yout[:, 12:15] = targetDQN5.model.predict(minibatch_next_states)
                                        Yout[:, 15:18] = targetDQN6.model.predict(minibatch_next_states)
                                        Yout[:, 18:21] = targetDQN7.model.predict(minibatch_next_states)
                                        Yout[:, 21:24] = targetDQN8.model.predict(minibatch_next_states)
                                        Yout[:, 24:27] = targetDQN9.model.predict(minibatch_next_states)
                                        Q_target = minibatch_rewards[:,0]  + DISCOUNT_RATE * np.max(Yout,axis=1) * ~minibatch_done[:,0]
                                        y = mainDQN1.model.predict(X)
                                        y[np.arange(len(X)), minibatch_actions.astype('int64')] = Q_target[np.arange(len(X))]
                                        #y[np.arange(len(X)), 1] = 0
                                        history1=mainDQN1.model.fit(X, y)
                                    step_count1 += 1
                                    if load_session == 0:
                                        logging.info(
                                        "------------------ \ntraining: Episode: {}  steps: {}  iteration_num: {} action: {}  tPTV: {}  reward: {} \n PlanScoreFine: {} PlanScore: {} Score_ptv: {} Score_bla:{} Score_rec: {} ".format
                                        (episode + 1, step_count, iter,  action, tPTV, round(reward,3), round(Score_fine,3),
                                           round(Score,3),round(scoreall[0],3), round(scorebladder,3), round(scorerectum,3)))
                                    else:
                                        logging.info(
                                            "------------------ \ntraining: Episode: {}  steps: {}  iteration_num: {} action: {}  tPTV: {}  reward: {} \n PlanScoreFine: {} PlanScore: {} Score_ptv: {} Score_bla:{} Score_rec: {} ".format
                                            (episode + LoadEpoch + 1, step_count, iter, action, tPTV, round(reward,3),
                                             round(Score_fine,3), round(Score,3),round(scoreall[0],3), round(scorebladder,3), round(scorerectum,3)))

                                # ----------------------- training NN 2 ----------------------------------------
                                if parasel == 1:
                                    replay_buffer2.append((state, action, reward, next_state, done))
                                if len(replay_buffer2) > BATCH_SIZE:
                                    if len(replay_buffer2) > int(REPLAY_MEMORY/10):
                                        train_num_episode = 5
                                    elif len(replay_buffer2) > int(REPLAY_MEMORY/5):
                                        train_num_episode = 20
                                    else:
                                        train_num_episode = 1
                                    for train_in_episode in range(train_num_episode):
                                        minibatch = rnd.sample(replay_buffer2, BATCH_SIZE)
                                        minibatch_states = np.reshape(np.vstack([x[0] for x in minibatch]),
                                                                      (BATCH_SIZE, INPUT_SIZE, 3))
                                        minibatch_actions = np.vstack([x[1] for x in minibatch])
                                        minibatch_rewards = np.vstack([x[2] for x in minibatch])
                                        minibatch_next_states = np.reshape(np.vstack([x[3] for x in minibatch]),
                                                                           (BATCH_SIZE, INPUT_SIZE, 3))
                                        minibatch_done = np.vstack([x[4] for x in minibatch])
                                        X = minibatch_states
                                        Yout = zeros([BATCH_SIZE, OUTPUT_SIZE *9])
                                        Yout[:, 0:3] = targetDQN1.model.predict(minibatch_next_states)
                                        Yout[:, 3:6] = targetDQN2.model.predict(minibatch_next_states)
                                        Yout[:, 6:9] = targetDQN3.model.predict(minibatch_next_states)
                                        Yout[:, 9:12] = targetDQN4.model.predict(minibatch_next_states)
                                        Yout[:, 12:15] = targetDQN5.model.predict(minibatch_next_states)
                                        Yout[:, 15:18] = targetDQN6.model.predict(minibatch_next_states)
                                        Yout[:, 18:21] = targetDQN7.model.predict(minibatch_next_states)
                                        Yout[:, 21:24] = targetDQN8.model.predict(minibatch_next_states)
                                        Yout[:, 24:27] = targetDQN9.model.predict(minibatch_next_states)
                                        Q_target = minibatch_rewards[:,0]  + DISCOUNT_RATE * np.max(Yout,axis=1) * ~minibatch_done[:,0]
                                        y = mainDQN2.model.predict(X)
                                        y[np.arange(len(X)), minibatch_actions.astype('int64')] = Q_target[np.arange(len(X))]
                                        #y[np.arange(len(X)), 1] = 0
                                        history2=mainDQN2.model.fit(X, y)
                                    step_count2 += 1
                                    if load_session == 0:
                                        logging.info(
                                        "------------------ \ntraining: Episode: {}  steps: {}  iteration_num: {} action: {}  tPTV: {}  reward: {} \n PlanScoreFine: {} PlanScore: {} Score_ptv: {} Score_bla:{} Score_rec: {} ".format
                                        (episode + 1, step_count, iter,  action, tPTV, round(reward,3), round(Score_fine,3),
                                           round(Score,3),round(scoreall[0],3), round(scorebladder,3), round(scorerectum,3)))
                                    else:
                                        logging.info(
                                            "------------------ \ntraining: Episode: {}  steps: {}  iteration_num: {} action: {}  tPTV: {}  reward: {} \n PlanScoreFine: {} PlanScore: {} Score_ptv: {} Score_bla:{} Score_rec: {} ".format
                                            (episode + LoadEpoch + 1, step_count, iter, action, tPTV, round(reward,3),
                                             round(Score_fine,3), round(Score,3),round(scoreall[0],3), round(scorebladder,3), round(scorerectum,3)))
                                # ----------------------- training NN 3 ----------------------------------------
                                if parasel == 2:
                                    replay_buffer3.append((state, action, reward, next_state, done))
                                if len(replay_buffer3) > BATCH_SIZE:
                                    if len(replay_buffer3) > int(REPLAY_MEMORY/10):
                                        train_num_episode = 5
                                    elif len(replay_buffer3) > int(REPLAY_MEMORY/5):
                                        train_num_episode = 20
                                    else:
                                        train_num_episode = 1
                                    for train_in_episode in range(train_num_episode):
                                        minibatch = rnd.sample(replay_buffer3, BATCH_SIZE)
                                        minibatch_states = np.reshape(np.vstack([x[0] for x in minibatch]),
                                                                      (BATCH_SIZE, INPUT_SIZE, 3))
                                        minibatch_actions = np.vstack([x[1] for x in minibatch])
                                        minibatch_rewards = np.vstack([x[2] for x in minibatch])
                                        minibatch_next_states = np.reshape(np.vstack([x[3] for x in minibatch]),
                                                                           (BATCH_SIZE, INPUT_SIZE, 3))
                                        minibatch_done = np.vstack([x[4] for x in minibatch])
                                        X = minibatch_states
                                        Yout = zeros([BATCH_SIZE, OUTPUT_SIZE * 9])
                                        Yout[:, 0:3] = targetDQN1.model.predict(minibatch_next_states)
                                        Yout[:, 3:6] = targetDQN2.model.predict(minibatch_next_states)
                                        Yout[:, 6:9] = targetDQN3.model.predict(minibatch_next_states)
                                        Yout[:, 9:12] = targetDQN4.model.predict(minibatch_next_states)
                                        Yout[:, 12:15] = targetDQN5.model.predict(minibatch_next_states)
                                        Yout[:, 15:18] = targetDQN6.model.predict(minibatch_next_states)
                                        Yout[:, 18:21] = targetDQN7.model.predict(minibatch_next_states)
                                        Yout[:, 21:24] = targetDQN8.model.predict(minibatch_next_states)
                                        Yout[:, 24:27] = targetDQN9.model.predict(minibatch_next_states)
                                        Q_target = minibatch_rewards[:,0]  + DISCOUNT_RATE * np.max(Yout,axis=1) * ~minibatch_done[:,0]
                                        y = mainDQN3.model.predict(X)
                                        y[np.arange(len(X)), minibatch_actions.astype('int64')] = Q_target[np.arange(len(X))]
                                        #y[np.arange(len(X)), 1] = 0
                                        history3=mainDQN3.model.fit(X, y)
                                    step_count3 += 1
                                    if load_session == 0:
                                        logging.info(
                                        "------------------ \ntraining: Episode: {}  steps: {}  iteration_num: {} action: {}  tPTV: {}  reward: {} \n PlanScoreFine: {} PlanScore: {} Score_ptv: {} Score_bla:{} Score_rec: {} ".format
                                        (episode + 1, step_count, iter,  action, tPTV, round(reward,3), round(Score_fine,3),
                                           round(Score,3),round(scoreall[0],3), round(scorebladder,3), round(scorerectum,3)))
                                    else:
                                        logging.info(
                                            "------------------ \ntraining: Episode: {}  steps: {}  iteration_num: {} action: {}  tPTV: {}  reward: {} \n PlanScoreFine: {} PlanScore: {} Score_ptv: {} Score_bla:{} Score_rec: {} ".format
                                            (episode + LoadEpoch + 1, step_count, iter, action, tPTV, round(reward,3),
                                             round(Score_fine,3), round(Score,3),round(scoreall[0],3), round(scorebladder,3), round(scorerectum,3)))
                                # ----------------------- training NN 4 ----------------------------------------
                                if parasel == 3:
                                    replay_buffer4.append((state, action, reward, next_state, done))
                                if len(replay_buffer4) > BATCH_SIZE:
                                    if len(replay_buffer4) > int(REPLAY_MEMORY/10):
                                        train_num_episode = 5
                                    elif len(replay_buffer4) > int(REPLAY_MEMORY/5):
                                        train_num_episode = 20
                                    else:
                                        train_num_episode = 1
                                    for train_in_episode in range(train_num_episode):
                                        minibatch = rnd.sample(replay_buffer4, BATCH_SIZE)
                                        minibatch_states = np.reshape(np.vstack([x[0] for x in minibatch]),
                                                                      (BATCH_SIZE, INPUT_SIZE, 3))
                                        minibatch_actions = np.vstack([x[1] for x in minibatch])
                                        minibatch_rewards = np.vstack([x[2] for x in minibatch])
                                        minibatch_next_states = np.reshape(np.vstack([x[3] for x in minibatch]),
                                                                           (BATCH_SIZE, INPUT_SIZE, 3))
                                        minibatch_done = np.vstack([x[4] for x in minibatch])
                                        X = minibatch_states
                                        Yout = zeros([BATCH_SIZE, OUTPUT_SIZE * 9])
                                        Yout[:, 0:3] = targetDQN1.model.predict(minibatch_next_states)
                                        Yout[:, 3:6] = targetDQN2.model.predict(minibatch_next_states)
                                        Yout[:, 6:9] = targetDQN3.model.predict(minibatch_next_states)
                                        Yout[:, 9:12] = targetDQN4.model.predict(minibatch_next_states)
                                        Yout[:, 12:15] = targetDQN5.model.predict(minibatch_next_states)
                                        Yout[:, 15:18] = targetDQN6.model.predict(minibatch_next_states)
                                        Yout[:, 18:21] = targetDQN7.model.predict(minibatch_next_states)
                                        Yout[:, 21:24] = targetDQN8.model.predict(minibatch_next_states)
                                        Yout[:, 24:27] = targetDQN9.model.predict(minibatch_next_states)
                                        Q_target = minibatch_rewards[:,0]  + DISCOUNT_RATE * np.max(Yout,axis=1) * ~minibatch_done[:,0]
                                        y = mainDQN4.model.predict(X)
                                        y[np.arange(len(X)), minibatch_actions.astype('int64')] = Q_target
                                        y[np.arange(len(X)), 1] = 0
                                        history4=mainDQN4.model.fit(X, y)
                                    step_count4 += 1
                                    if load_session == 0:
                                        logging.info(
                                        "------------------ \ntraining: Episode: {}  steps: {}  iteration_num: {} action: {}  tPTV: {}  reward: {} \n PlanScoreFine: {} PlanScore: {} Score_ptv: {} Score_bla:{} Score_rec: {} ".format
                                        (episode + 1, step_count, iter,  action, tPTV, round(reward,3), round(Score_fine,3),
                                           round(Score,3),round(scoreall[0],3), round(scorebladder,3), round(scorerectum,3)))
                                    else:
                                        logging.info(
                                            "------------------ \ntraining: Episode: {}  steps: {}  iteration_num: {} action: {}  tPTV: {}  reward: {} \n PlanScoreFine: {} PlanScore: {} Score_ptv: {} Score_bla:{} Score_rec: {} ".format
                                            (episode + LoadEpoch + 1, step_count, iter, action, tPTV, round(reward,3),
                                             round(Score_fine,3), round(Score,3),round(scoreall[0],3), round(scorebladder,3), round(scorerectum,3)))

                                # ----------------------- training NN 5 ----------------------------------------
                                if parasel == 4:
                                    replay_buffer5.append((state, action, reward, next_state, done))
                                if len(replay_buffer5) > BATCH_SIZE:
                                    if len(replay_buffer5) > int(REPLAY_MEMORY/10):
                                        train_num_episode = 5
                                    elif len(replay_buffer5) > int(REPLAY_MEMORY/5):
                                        train_num_episode = 20
                                    else:
                                        train_num_episode = 1
                                    for train_in_episode in range(train_num_episode):
                                        minibatch = rnd.sample(replay_buffer5, BATCH_SIZE)
                                        minibatch_states = np.reshape(np.vstack([x[0] for x in minibatch]),
                                                                      (BATCH_SIZE, INPUT_SIZE, 3))
                                        minibatch_actions = np.vstack([x[1] for x in minibatch])
                                        minibatch_rewards = np.vstack([x[2] for x in minibatch])
                                        minibatch_next_states = np.reshape(np.vstack([x[3] for x in minibatch]),
                                                                           (BATCH_SIZE, INPUT_SIZE, 3))
                                        minibatch_done = np.vstack([x[4] for x in minibatch])
                                        X = minibatch_states
                                        Yout = zeros([BATCH_SIZE, OUTPUT_SIZE * 9])
                                        Yout[:, 0:3] = targetDQN1.model.predict(minibatch_next_states)
                                        Yout[:, 3:6] = targetDQN2.model.predict(minibatch_next_states)
                                        Yout[:, 6:9] = targetDQN3.model.predict(minibatch_next_states)
                                        Yout[:, 9:12] = targetDQN4.model.predict(minibatch_next_states)
                                        Yout[:, 12:15] = targetDQN5.model.predict(minibatch_next_states)
                                        Yout[:, 15:18] = targetDQN6.model.predict(minibatch_next_states)
                                        Yout[:, 18:21] = targetDQN7.model.predict(minibatch_next_states)
                                        Yout[:, 21:24] = targetDQN8.model.predict(minibatch_next_states)
                                        Yout[:, 24:27] = targetDQN9.model.predict(minibatch_next_states)
                                        Q_target = minibatch_rewards[:,0]  + DISCOUNT_RATE * np.max(Yout,axis=1) * ~minibatch_done[:,0]
                                        y = mainDQN5.model.predict(X)
                                        y[np.arange(len(X)), minibatch_actions.astype('int64')] = Q_target[np.arange(len(X))]
                                        #y[np.arange(len(X)), 1] = 0
                                        history5=mainDQN5.model.fit(X, y)
                                    step_count5 += 1
                                    if load_session == 0:
                                        logging.info(
                                        "------------------ \ntraining: Episode: {}  steps: {}  iteration_num: {} action: {}  tPTV: {}  reward: {} \n PlanScoreFine: {} PlanScore: {} Score_ptv: {} Score_bla:{} Score_rec: {} ".format
                                        (episode + 1, step_count, iter,  action, tPTV, round(reward,3), round(Score_fine,3),
                                           round(Score,3),round(scoreall[0],3), round(scorebladder,3), round(scorerectum,3)))
                                    else:
                                        logging.info(
                                            "------------------ \ntraining: Episode: {}  steps: {}  iteration_num: {} action: {}  tPTV: {}  reward: {} \n PlanScoreFine: {} PlanScore: {} Score_ptv: {} Score_bla:{} Score_rec: {} ".format
                                            (episode + LoadEpoch + 1, step_count, iter, action, tPTV, round(reward,3),
                                             round(Score_fine,3), round(Score,3),round(scoreall[0],3), round(scorebladder,3), round(scorerectum,3)))

                                # ----------------------- training NN 6 ----------------------------------------
                                if parasel == 5:
                                    replay_buffer6.append((state, action, reward, next_state, done))
                                if len(replay_buffer6) > BATCH_SIZE:
                                    if len(replay_buffer6) > int(REPLAY_MEMORY/10):
                                        train_num_episode = 5
                                    elif len(replay_buffer6) > int(REPLAY_MEMORY/5):
                                        train_num_episode = 20
                                    else:
                                        train_num_episode = 1
                                    for train_in_episode in range(train_num_episode):
                                        minibatch = rnd.sample(replay_buffer6, BATCH_SIZE)
                                        minibatch_states = np.reshape(np.vstack([x[0] for x in minibatch]),
                                                                      (BATCH_SIZE, INPUT_SIZE, 3))
                                        minibatch_actions = np.vstack([x[1] for x in minibatch])
                                        minibatch_rewards = np.vstack([x[2] for x in minibatch])
                                        minibatch_next_states = np.reshape(np.vstack([x[3] for x in minibatch]),
                                                                           (BATCH_SIZE, INPUT_SIZE, 3))
                                        minibatch_done = np.vstack([x[4] for x in minibatch])
                                        X = minibatch_states
                                        Yout = zeros([BATCH_SIZE, OUTPUT_SIZE * 9])
                                        Yout[:, 0:3] = targetDQN1.model.predict(minibatch_next_states)
                                        Yout[:, 3:6] = targetDQN2.model.predict(minibatch_next_states)
                                        Yout[:, 6:9] = targetDQN3.model.predict(minibatch_next_states)
                                        Yout[:, 9:12] = targetDQN4.model.predict(minibatch_next_states)
                                        Yout[:, 12:15] = targetDQN5.model.predict(minibatch_next_states)
                                        Yout[:, 15:18] = targetDQN6.model.predict(minibatch_next_states)
                                        Yout[:, 18:21] = targetDQN7.model.predict(minibatch_next_states)
                                        Yout[:, 21:24] = targetDQN8.model.predict(minibatch_next_states)
                                        Yout[:, 24:27] = targetDQN9.model.predict(minibatch_next_states)
                                        Q_target = minibatch_rewards[:,0]  + DISCOUNT_RATE * np.max(Yout,axis=1) * ~minibatch_done[:,0]
                                        y = mainDQN6.model.predict(X)
                                        y[np.arange(len(X)), minibatch_actions.astype('int64')] = Q_target[np.arange(len(X))]
                                        #y[np.arange(len(X)), 1] = 0
                                        history6=mainDQN6.model.fit(X, y)
                                    step_count6 += 1
                                    if load_session == 0:
                                        logging.info(
                                        "------------------ \ntraining: Episode: {}  steps: {}  iteration_num: {} action: {}  tPTV: {}  reward: {} \n PlanScoreFine: {} PlanScore: {} Score_ptv: {} Score_bla:{} Score_rec: {} ".format
                                        (episode + 1, step_count, iter,  action, tPTV, round(reward,3), round(Score_fine,3),
                                           round(Score,3),round(scoreall[0],3), round(scorebladder,3), round(scorerectum,3)))
                                    else:
                                        logging.info(
                                            "------------------ \ntraining: Episode: {}  steps: {}  iteration_num: {} action: {}  tPTV: {}  reward: {} \n PlanScoreFine: {} PlanScore: {} Score_ptv: {} Score_bla:{} Score_rec: {} ".format
                                            (episode + LoadEpoch + 1, step_count, iter, action, tPTV, round(reward,3),
                                             round(Score_fine,3), round(Score,3),round(scoreall[0],3), round(scorebladder,3), round(scorerectum,3)))

                                # ----------------------- training NN 7 ----------------------------------------
                                if parasel == 6:
                                    replay_buffer7.append((state, action, reward, next_state, done))
                                if len(replay_buffer7) > BATCH_SIZE:
                                    if len(replay_buffer7) > int(REPLAY_MEMORY/10):
                                        train_num_episode = 5
                                    elif len(replay_buffer7) > int(REPLAY_MEMORY/5):
                                        train_num_episode = 20
                                    else:
                                        train_num_episode = 1
                                    for train_in_episode in range(train_num_episode):
                                        minibatch = rnd.sample(replay_buffer7, BATCH_SIZE)
                                        minibatch_states = np.reshape(np.vstack([x[0] for x in minibatch]),
                                                                      (BATCH_SIZE, INPUT_SIZE, 3))
                                        minibatch_actions = np.vstack([x[1] for x in minibatch])
                                        minibatch_rewards = np.vstack([x[2] for x in minibatch])
                                        minibatch_next_states = np.reshape(np.vstack([x[3] for x in minibatch]),
                                                                           (BATCH_SIZE, INPUT_SIZE, 3))
                                        minibatch_done = np.vstack([x[4] for x in minibatch])
                                        X = minibatch_states
                                        Yout = zeros([BATCH_SIZE, OUTPUT_SIZE * 9])
                                        Yout[:, 0:3] = targetDQN1.model.predict(minibatch_next_states)
                                        Yout[:, 3:6] = targetDQN2.model.predict(minibatch_next_states)
                                        Yout[:, 6:9] = targetDQN3.model.predict(minibatch_next_states)
                                        Yout[:, 9:12] = targetDQN4.model.predict(minibatch_next_states)
                                        Yout[:, 12:15] = targetDQN5.model.predict(minibatch_next_states)
                                        Yout[:, 15:18] = targetDQN6.model.predict(minibatch_next_states)
                                        Yout[:, 18:21] = targetDQN7.model.predict(minibatch_next_states)
                                        Yout[:, 21:24] = targetDQN8.model.predict(minibatch_next_states)
                                        Yout[:, 24:27] = targetDQN9.model.predict(minibatch_next_states)
                                        Q_target = minibatch_rewards[:,0]  + DISCOUNT_RATE * np.max(Yout,axis=1) * ~minibatch_done[:,0]
                                        y = mainDQN7.model.predict(X)
                                        y[np.arange(len(X)), minibatch_actions.astype('int64')] = Q_target[np.arange(len(X))]
                                        #y[np.arange(len(X)), 1] = 0
                                        history7=mainDQN7.model.fit(X, y)
                                    step_count7 += 1
                                    if load_session == 0:
                                        logging.info(
                                        "------------------ \ntraining: Episode: {}  steps: {}  iteration_num: {} action: {}  tPTV: {}  reward: {} \n PlanScoreFine: {} PlanScore: {} Score_ptv: {} Score_bla:{} Score_rec: {} ".format
                                        (episode + 1, step_count, iter,  action, tPTV, round(reward,3), round(Score_fine,3),
                                           round(Score,3),round(scoreall[0],3), round(scorebladder,3), round(scorerectum,3)))
                                    else:
                                        logging.info(
                                            "------------------ \ntraining: Episode: {}  steps: {}  iteration_num: {} action: {}  tPTV: {}  reward: {} \n PlanScoreFine: {} PlanScore: {} Score_ptv: {} Score_bla:{} Score_rec: {} ".format
                                            (episode + LoadEpoch + 1, step_count, iter, action, tPTV, round(reward,3),
                                             round(Score_fine,3), round(Score,3),round(scoreall[0],3), round(scorebladder,3), round(scorerectum,3)))

                                # ----------------------- training NN 8 ----------------------------------------
                                if parasel == 7:
                                    replay_buffer8.append((state, action, reward, next_state, done))
                                if len(replay_buffer8) > BATCH_SIZE:
                                    if len(replay_buffer8) > int(REPLAY_MEMORY/10):
                                        train_num_episode = 5
                                    elif len(replay_buffer8) > int(REPLAY_MEMORY/5):
                                        train_num_episode = 20
                                    else:
                                        train_num_episode = 1
                                    for train_in_episode in range(train_num_episode):
                                        minibatch = rnd.sample(replay_buffer8, BATCH_SIZE)
                                        minibatch_states = np.reshape(np.vstack([x[0] for x in minibatch]),
                                                                      (BATCH_SIZE, INPUT_SIZE, 3))
                                        minibatch_actions = np.vstack([x[1] for x in minibatch])
                                        minibatch_rewards = np.vstack([x[2] for x in minibatch])
                                        minibatch_next_states = np.reshape(np.vstack([x[3] for x in minibatch]),
                                                                           (BATCH_SIZE, INPUT_SIZE, 3))
                                        minibatch_done = np.vstack([x[4] for x in minibatch])
                                        X = minibatch_states
                                        Yout = zeros([BATCH_SIZE, OUTPUT_SIZE * 9])
                                        Yout[:, 0:3] = targetDQN1.model.predict(minibatch_next_states)
                                        Yout[:, 3:6] = targetDQN2.model.predict(minibatch_next_states)
                                        Yout[:, 6:9] = targetDQN3.model.predict(minibatch_next_states)
                                        Yout[:, 9:12] = targetDQN4.model.predict(minibatch_next_states)
                                        Yout[:, 12:15] = targetDQN5.model.predict(minibatch_next_states)
                                        Yout[:, 15:18] = targetDQN6.model.predict(minibatch_next_states)
                                        Yout[:, 18:21] = targetDQN7.model.predict(minibatch_next_states)
                                        Yout[:, 21:24] = targetDQN8.model.predict(minibatch_next_states)
                                        Yout[:, 24:27] = targetDQN9.model.predict(minibatch_next_states)
                                        Q_target = minibatch_rewards[:,0]  + DISCOUNT_RATE * np.max(Yout,axis=1) * ~minibatch_done[:,0]
                                        y = mainDQN8.model.predict(X)
                                        y[np.arange(len(X)), minibatch_actions.astype('int64')] = Q_target[np.arange(len(X))]
                                        #y[np.arange(len(X)), 1] = 0
                                        history8=mainDQN8.model.fit(X, y)
                                    step_count8 += 1
                                    if load_session == 0:
                                        logging.info(
                                        "------------------ \ntraining: Episode: {}  ste ps: {}  iteration_num: {} action: {}  tPTV: {}  reward: {} \n PlanScoreFine: {} PlanScore: {} Score_ptv: {} Score_bla:{} Score_rec: {} ".format
                                        (episode + 1, step_count, iter,  action, tPTV, round(reward,3), round(Score_fine,3),
                                           round(Score,3),round(scoreall[0],3), round(scorebladder,3), round(scorerectum,3)))
                                    else:
                                        logging.info(
                                            "------------------ \ntraining: Episode: {}  steps: {}  iteration_num: {} action: {}  tPTV: {}  reward: {} \n PlanScoreFine: {} PlanScore: {} Score_ptv: {} Score_bla:{} Score_rec: {} ".format
                                            (episode + LoadEpoch + 1, step_count, iter, action, tPTV, round(reward,3),
                                             round(Score_fine,3), round(Score,3),round(scoreall[0],3), round(scorebladder,3), round(scorerectum,3)))

                                # ----------------------- training NN 9 ----------------------------------------
                                if parasel == 8:
                                    replay_buffer9.append((state, action, reward, next_state, done))
                                if len(replay_buffer9) > BATCH_SIZE:
                                    if len(replay_buffer9) > int(REPLAY_MEMORY/10):
                                        train_num_episode = 5
                                    elif len(replay_buffer9) > int(REPLAY_MEMORY/5):
                                        train_num_episode = 20
                                    else:
                                        train_num_episode = 1
                                    for train_in_episode in range(train_num_episode):
                                        minibatch = rnd.sample(replay_buffer9, BATCH_SIZE)
                                        minibatch_states = np.reshape(np.vstack([x[0] for x in minibatch]),
                                                                      (BATCH_SIZE, INPUT_SIZE, 3))
                                        minibatch_actions = np.vstack([x[1] for x in minibatch])
                                        minibatch_rewards = np.vstack([x[2] for x in minibatch])
                                        minibatch_next_states = np.reshape(np.vstack([x[3] for x in minibatch]),
                                                                           (BATCH_SIZE, INPUT_SIZE, 3))
                                        minibatch_done = np.vstack([x[4] for x in minibatch])
                                        X = minibatch_states
                                        Yout = zeros([BATCH_SIZE, OUTPUT_SIZE * 9])
                                        Yout[:, 0:3] = targetDQN1.model.predict(minibatch_next_states)
                                        Yout[:, 3:6] = targetDQN2.model.predict(minibatch_next_states)
                                        Yout[:, 6:9] = targetDQN3.model.predict(minibatch_next_states)
                                        Yout[:, 9:12] = targetDQN4.model.predict(minibatch_next_states)
                                        Yout[:, 12:15] = targetDQN5.model.predict(minibatch_next_states)
                                        Yout[:, 15:18] = targetDQN6.model.predict(minibatch_next_states)
                                        Yout[:, 18:21] = targetDQN7.model.predict(minibatch_next_states)
                                        Yout[:, 21:24] = targetDQN8.model.predict(minibatch_next_states)
                                        Yout[:, 24:27] = targetDQN9.model.predict(minibatch_next_states)
                                        Q_target = minibatch_rewards[:,0]  + DISCOUNT_RATE * np.max(Yout,
                                                                                              axis=1) * ~minibatch_done[:,0]
                                        y = mainDQN9.model.predict(X)
                                        y[np.arange(len(X)), minibatch_actions.astype('int64')] = Q_target[np.arange(len(X))]
                                        #y[np.arange(len(X)), 1] = 0
                                        history9=mainDQN9.model.fit(X, y)
                                    step_count9 += 1
                                    if load_session == 0:
                                        logging.info(
                                        "------------------ \ntraining: Episode: {}  steps: {}  iteration_num: {} action: {}  tPTV: {}  reward: {} \n PlanScoreFine: {} PlanScore: {} Score_ptv: {} Score_bla:{} Score_rec: {} ".format
                                        (episode + 1, step_count, iter,  action, tPTV, round(reward,3), round(Score_fine,3),
                                           round(Score,3),round(scoreall[0],3), round(scorebladder,3), round(scorerectum,3)))
                                    else:
                                        logging.info(
                                            "------------------ \ntraining: Episode: {}  steps: {}  iteration_num: {} action: {}  tPTV: {}  reward: {} \n PlanScoreFine: {} PlanScore: {} Score_ptv: {} Score_bla:{} Score_rec: {} ".format
                                            (episode + LoadEpoch + 1, step_count, iter, action, tPTV, round(reward,3),
                                             round(Score_fine,3), round(Score,3),round(scoreall[0],3), round(scorebladder,3), round(scorerectum,3)))


                                steps = [step_count1, step_count2, step_count3, step_count4, step_count5,step_count6, step_count7, step_count8, step_count9]
                                if np.min(steps) % TARGET_UPDATE_FREQUENCY == 0:
                                    targetDQN1.model.set_weights(mainDQN1.model.get_weights())
                                    targetDQN2.model.set_weights(mainDQN2.model.get_weights())
                                    targetDQN3.model.set_weights(mainDQN3.model.get_weights())
                                    targetDQN4.model.set_weights(mainDQN4.model.get_weights())
                                    targetDQN5.model.set_weights(mainDQN5.model.get_weights())
                                    targetDQN6.model.set_weights(mainDQN6.model.get_weights())
                                    targetDQN7.model.set_weights(mainDQN7.model.get_weights())
                                    targetDQN8.model.set_weights(mainDQN8.model.get_weights())
                                    targetDQN9.model.set_weights(mainDQN9.model.get_weights())


                                state = next_state

                    if num_q !=0:
                        reward_check[episode]=reward_sum_total/num_q
                    else:
                        reward_check[episode]=0

                    if num_q!=0:
                        q_check[episode] = qvalue_sum/num_q
                    else:
                        q_check[episode]=0
                    if num_q !=0:
                       loss_check[episode] = loss_sum/num_q
                    else:
                       loss_check[episode]=0

                    print("Episode: {}  Reward: {}  Q-value: {} loss: {} q_num: {}".format
                                          (episode + 1, reward_check[episode], q_check[episode], loss_check[episode],num_q))



                    if save_session == 1 and (episode+1) % 5 == 0:
                        mainDQN1.model.save(save_session_name +'mainDQN1_episode_'+str(episode+1+LoadEpoch)+'.h5')
                        mainDQN2.model.save(save_session_name + 'mainDQN2_episode_' + str(episode + 1 + LoadEpoch)+'.h5')
                        mainDQN3.model.save(save_session_name + 'mainDQN3_episode_' + str(episode + 1 + LoadEpoch)+'.h5')
                        mainDQN4.model.save(save_session_name + 'mainDQN4_episode_' + str(episode + 1 + LoadEpoch)+'.h5')
                        mainDQN5.model.save(save_session_name + 'mainDQN5_episode_' + str(episode + 1 + LoadEpoch)+'.h5')
                        mainDQN6.model.save(save_session_name + 'mainDQN6_episode_' + str(episode + 1 + LoadEpoch)+'.h5')
                        mainDQN7.model.save(save_session_name + 'mainDQN7_episode_' + str(episode + 1 + LoadEpoch)+'.h5')
                        mainDQN8.model.save(save_session_name + 'mainDQN8_episode_' + str(episode + 1 + LoadEpoch)+'.h5')
                        mainDQN9.model.save(save_session_name + 'mainDQN9_episode_' + str(episode + 1 + LoadEpoch)+'.h5')

                        targetDQN1.model.save(save_session_name + 'targetDQN1_episode_' + str(episode + 1 + LoadEpoch)+'.h5')
                        targetDQN2.model.save(save_session_name + 'targetDQN2_episode_' + str(episode + 1 + LoadEpoch)+'.h5')
                        targetDQN3.model.save(save_session_name + 'targetDQN3_episode_' + str(episode + 1 + LoadEpoch)+'.h5')
                        targetDQN4.model.save(save_session_name + 'targetDQN4_episode_' + str(episode + 1 + LoadEpoch)+'.h5')
                        targetDQN5.model.save(save_session_name + 'targetDQN5_episode_' + str(episode + 1 + LoadEpoch)+'.h5')
                        targetDQN6.model.save(save_session_name + 'targetDQN6_episode_' + str(episode + 1 + LoadEpoch)+'.h5')
                        targetDQN7.model.save(save_session_name + 'targetDQN7_episode_' + str(episode + 1 + LoadEpoch)+'.h5')
                        targetDQN8.model.save(save_session_name + 'targetDQN8_episode_' + str(episode + 1 + LoadEpoch)+'.h5')
                        targetDQN9.model.save(save_session_name + 'targetDQN9_episode_' + str(episode + 1 + LoadEpoch)+'.h5')
                        data_out = '/data/data/Results/GPU2/checksPYdrm15/'
                        data_out2 ='/data/data/Results/GPU2/replayPYdrm15/'
                        np.save(data_out+'reward_check.npy'+str(episode+1),reward_check)
                        np.save(data_out+'q_check.npy' + str(episode + 1), q_check)
                        np.save(data_out2+'replay_buffer1_episode.npy'+str(episode+1),replay_buffer1)
                        np.save(data_out2+'replay_buffer2_episode.npy'+str(episode+1),replay_buffer2)
                        np.save(data_out2+'replay_buffer3_episode.npy'+str(episode+1),replay_buffer3)
                        np.save(data_out2+'replay_buffer4_episode.npy'+str(episode+1),replay_buffer4)
                        np.save(data_out2+'replay_buffer5_episode.npy'+str(episode+1),replay_buffer5)
                        np.save(data_out2+'replay_buffer6_episode.npy'+str(episode+1),replay_buffer6)
                        np.save(data_out2+'replay_buffer7_episode.npy'+str(episode+1),replay_buffer7)
                        np.save(data_out2+'replay_buffer8_episode.npy'+str(episode+1),replay_buffer8)
                        np.save(data_out2+'replay_buffer9_episode.npy'+str(episode+1),replay_buffer9)
                    if (episode+1) % 5 == 0:
                        flagg=0
                        vali_num = vali_num + 1
                        plt.figure(vali_num)
                        evalu_training(mainDQN1, mainDQN2, mainDQN3, mainDQN4, mainDQN5, mainDQN6, mainDQN7, mainDQN8, mainDQN9,runOpt_dvh,episode+1,flagg,pdose,maxiter)

            print("Training done!, Test start!")
            flagg=0
            plt.figure(100)
            #evalu_training(mainDQN1, mainDQN2, mainDQN3, mainDQN4, mainDQN5, mainDQN6, mainDQN7, mainDQN8, mainDQN9, runOpt_dvh, LoadEpoch,flagg,pdose,maxiter)
            bot_play(mainDQN1, mainDQN2, mainDQN3, mainDQN4, mainDQN5, mainDQN6, mainDQN7, mainDQN8, mainDQN9, runOpt_dvh, LoadEpoch,flagg,pdose,maxiter)
            

#         bot_play(mainDQN1,mainDQN2,mainDQN3,mainDQN4,mainDQN5,runOpt1,episode+1)



if __name__ == "__main__":
	main()
