
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:55:19 2019

aim: validate the training set during the DQN. 

@author: writed by Chenyang Shen, modified by Chao Wang
"""
import numpy as np
from numpy import zeros
from .dqn_rule_network import DQN
import matplotlib.pyplot as plt
from .score_calcu import planIQ
import math as m
import pandas as pd
import csv



INPUT_SIZE = 100  # DVH interval number
MAX_STEP = 1 # maximum No. of tuning parameter
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

def evalu_training(mainDQN1: DQN, mainDQN2: DQN, mainDQN3: DQN, 
             mainDQN4: DQN,  mainDQN5: DQN, mainDQN6: DQN,
             mainDQN7: DQN,  mainDQN8: DQN, mainDQN9: DQN,
             runOpt_dvh, episode,flagg,pdose,maxiter,DPTVex,DBLAex,DRECex) -> None:
    #test_set =['01','12','17','24','29','35','40','50','59','62']#,'18','20','22','23','25','26','27','28','30','31','36','37','42','43','45','46','54','57','61','65','66','68','70','73','74','77','80','81','83','84','85','87','88','91','92','93','95','97','98'] #['12','17'] # training set
        
    #----------------two ways of NN schemes (with/without rules) --------------------------
    DPTV=np.array(DPTVex)
    DBLA=np.array(DBLAex)
    DREC=np.array(DRECex)
    score_fine,score,scoreall=planIQ(MPTV, MBLA, MREC, xVec,pdose)
    tPTV_all = np.zeros((MAX_STEP + 1))
    tBLA_all = np.zeros((MAX_STEP + 1))
    tREC_all = np.zeros((MAX_STEP + 1))
    lambdaPTV_all = np.zeros((MAX_STEP + 1))
    lambdaBLA_all = np.zeros((MAX_STEP + 1))
    lambdaREC_all = np.zeros((MAX_STEP + 1))
    VPTV_all = np.zeros((MAX_STEP + 1))
    VBLA_all = np.zeros((MAX_STEP + 1))
    VREC_all = np.zeros((MAX_STEP + 1))

    #DPTV=[x/4500 for x in DPTV]
    #DBLA=[x/4500 for x in DBLA]
    #DREC=[x/4500 for x in DREC]
    DPTV=np.sort(DPTV)
    DBLA=np.sort(DBLA)
    DREC=np.sort(DREC)
    DPTV=np.flipud(DPTV)
    DBLA=np.flipud(DBLA)
    DREC=np.flipud(DREC)
    #print(DPTVex)
    #print(DPTV)
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
    state_test0=Y
    # path='/data/data/testdata/'
    # np.save(path+'PAT2DVHinput',Y)

    

    for case in range(1): # here we only consdier ruled based. Change range(1) into range(2) will test both cases
        state_test = state_test0
        # ------------------------ initial paramaters & input --------------------
        tPTV = 1.1
        tBLA = 1
        tREC = 1
        lambdaPTV = 1
        lambdaBLA = 1
        lambdaREC = 1
        VPTV = 1
        VBLA = 1
        VREC = 1
        # -------------- NN ----------------------------------
        for i in range(MAX_STEP):
            X = np.reshape(state_test,(1,INPUT_SIZE,3))
            tempoutput = mainDQN1.model.predict(X)
            tempoutput1 = tempoutput[0,:]
            tempoutput = mainDQN2.model.predict(X)
            tempoutput2 = tempoutput[0, :]
            tempoutput = mainDQN3.model.predict(X)
            tempoutput3 = tempoutput[0, :]
            tempoutput = mainDQN4.model.predict(X)
            tempoutput4 = tempoutput[0, :]
            tempoutput = mainDQN5.model.predict(X)
            tempoutput5 = tempoutput[0, :]
            tempoutput = mainDQN6.model.predict(X)
            tempoutput6 = tempoutput[0,:]
            tempoutput = mainDQN7.model.predict(X)
            tempoutput7 = tempoutput[0, :]
            tempoutput = mainDQN8.model.predict(X)
            tempoutput8 = tempoutput[0, :]
            tempoutput = mainDQN9.model.predict(X)
            tempoutput9 = tempoutput[0, :]
            
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
            value = np.zeros(9)
            value[0] = np.max(tempoutput1)
            value[1] = np.max(tempoutput2)
            value[2] = np.max(tempoutput3)
            value[3] = np.max(tempoutput4)
            value[4] = np.max(tempoutput5)
            value[5] = np.max(tempoutput6)
            value[6] = np.max(tempoutput7)
            value[7] = np.max(tempoutput8)
            value[8] = np.max(tempoutput9)


            ######################################## tune parameter according to NN  ############################################
            paraidx = np.argmax(value)
            if paraidx == 0:
                action = np.argmax(tempoutput1)
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

            if paraidx == 1:
                action = np.argmax(tempoutput2)
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

            if paraidx == 2:
                action = np.argmax(tempoutput3)
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

            if paraidx == 3:
                action = np.argmax(tempoutput4)
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

            if paraidx == 4:
                action = np.argmax(tempoutput5)
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
            if paraidx == 5:
                action = np.argmax(tempoutput6)
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
            if paraidx == 6:
                action = np.argmax(tempoutput7)
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

            if paraidx == 7:
                action = np.argmax(tempoutput8)
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
            if paraidx == 8:
                action = np.argmax(tempoutput9)
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

            # --------------------- solve treatment planning optmization -----------------------------
            print("planScore:{} planScore_fine:{} TPPs:{} action {}".format(score,score_fine,paraidx,action))
            #print("tPTV:{} tBLA:{} tREC:{} lambdaPTV:{} lambdaBLA:{} lambdaREC:{} VPTV:{} VBLA:{} VREC:{}".format(tPTV,tBLA,tREC,lambdaPTV,lambdaBLA,lambdaREC,VPTV,VBLA,VREC))
            
            # collect the result in each iteration 
            tPTV_all[i + 1] = tPTV
            tBLA_all[i + 1] = tBLA
            tREC_all[i + 1] = tREC
            lambdaPTV_all[i + 1] = lambdaPTV
            lambdaBLA_all[i + 1] = lambdaBLA
            lambdaREC_all[i + 1] = lambdaREC
            VPTV_all[i + 1] = VPTV
            VBLA_all[i + 1] = VBLA
            VREC_all[i + 1] = VREC


            #data_result_path2='/data/data/Results/GPU2/figuresrm75/'
            #plt.plot(x_ptv, y_ptv)
            #plt.plot(x_bladder, y_bladder)
            #plt.plot(x_rectum, y_rectum)
            #plt.legend(('ptv','bladder','rectum'))
            #plt.show(block=False)
            #plt.title('DVH'+str(episode)+'step'+str(i+1))
            #plt.savefig(data_result_path2+id+'DVH'+str(episode)+'step'+str(i+1)+'.png')
            #plt.close()



  



