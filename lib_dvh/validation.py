# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:55:19 2019

aim: validate & test DQN 

@author: writed by Chenyang Shen, modified by Chao Wang
"""
import numpy as np
import logging 
from .dqn_rule_network import DQN
import matplotlib.pyplot as plt
from .score_calcu import planIQ
import math as m
import os

INPUT_SIZE = 100  # DVH interval number
MAX_STEP = 45 # maximum No. of tuning parameter
# ------------- range of parmaeter -----------------
paraMax = 100000 # change in validation as well
paraMin = 0
paraMax_tPTV = 1.2
paraMin_tPTV = 1
paraMax_tOAR = 1
paraMax_VOAR = 1
paraMax_VPTV = 0.3
# ---------------------------------------------------

logging.basicConfig(filename = 'result_training_withoutrule.log',level = logging.INFO,
                    format = '%(message)s')

from lib_dvh.data_prep import loadDoseMatrix,loadMask,ProcessDmat

def bot_play(mainDQN1: DQN, mainDQN2: DQN, mainDQN3: DQN, 
             mainDQN4: DQN,  mainDQN5: DQN, mainDQN6: DQN,
             mainDQN7: DQN,  mainDQN8: DQN, mainDQN9: DQN,
             runOpt_dvh, episode,flagg,pdose,maxiter) -> None:
    #path = "/data/data/dose_depositon/prostate_dijs/f_dijs/"# directory of treatment planning system
    #dirs = os.listdir(path)
    #plt.clf()
    #if flagg == 0:
        #test_num = 3 # validate only the first three cases
    #else:
        #test_num = len(dirs)-15 # avoid the 21st case which is lack of data
    test_set= ['01','12','17','24','29','35','40','50','59','62']
    logging.info('------------------------------------------ validation ----------------------------------------------------')
    for sampleid in range(10):   
        #id = os.path.splitext(dirs[sampleid+15])[0]
        id = test_set[sampleid]
        logging.info('############################# Testing start for patient '+ id +' #################################################')
        data_path= '/data/data/dose_deposition3/f_dijs/0'
        data_path2='/data/data/dose_deposition3/plostate_dijs/f_masks/0'
        doseMatrix_test = loadDoseMatrix(data_path + id + '.hdf5')
        targetLabels_test, bladderLabeltest, rectumLabeltest, PTVLabeltest = loadMask(data_path2 + id + '.h5')
       
        # ---------------- generate the linear systems -----------------------------
        MPTV, MBLA, MREC, MBLA1, MREC1 = ProcessDmat(doseMatrix_test, 
                                                     targetLabels_test, 
                                                     bladderLabeltest, 
                                                     rectumLabeltest) 
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
        xVec = np.ones((MPTV.shape[1],))
        gamma = np.zeros((MPTV.shape[0],))
        # --------------------- solve treatment planning optmization -----------------------------
        state_test0, iter, xVec, gamma = \
            runOpt_dvh(MPTV, MBLA, MREC,tPTV,tBLA, tREC, 
                       lambdaPTV, lambdaBLA, lambdaREC,
                       VPTV, VBLA, VREC, xVec,gamma,pdose,maxiter)  
        # --------------------- generate input for NN -----------------------------    
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

        Y = np.zeros((1000, 3))
        Y[:, 0] = y_ptv
        Y[:, 1] = y_bladder
        Y[:, 2] = y_rectum

        X = np.zeros((1000, 3))
        X[:, 0] = x_ptv
        X[:, 1] = x_bladder
        X[:, 2] = x_rectum
        
        data_result_path='/data/data/Results/GPU2/general180/'
        np.save(data_result_path+id+'xDVHYInitial',
                Y)
        np.save(data_result_path+id+'xDVHXInitial',
                X)
        np.save(data_result_path+id+'xVecInitial', xVec)
        np.save(data_result_path+id+'DoseInitial', Dose)
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
        logging.info('---------------------- initialization ------------------------------')
        logging.info("Iteration_num: {}  PlanScore: {}  PlanScore_fine: {}".format(iter, planScore, planScore_fine))

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
        #----------------two ways of NN schemes (with/without rules) -------------------------- 
        for case in range(2):
            state_test = state_test0
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
                
                if case==0: # Rule-based 
                    if scoreall[0]<0.01:
                        tempoutput4[1] = -100 # only allow lambda_ptv increases 
                        tempoutput4[2] = -100

                        tempoutput1[0] = -100 # only allow t_ptv decreases
                        tempoutput1[1] = -100

                        tempoutput7[0] = -100 # only allow v_ptv decreases
                        tempoutput7[1] = -100

                        tempoutput5[0] = -100 # only allow lambda_bla decreases
                        tempoutput5[1] = -100

                        tempoutput6[0] = -100 # only allow lambda_rec decreases
                        tempoutput6[1] = -100

                    else:
                        scorebladder = scoreall[1]+scoreall[2]+scoreall[3]+scoreall[4]
                        scorerectum = scoreall[5] + scoreall[6] + scoreall[7] + scoreall[8]
                        if scorebladder<2 and scorebladder<scorerectum: # need to improve bla
                            tempoutput5[1] = -100 # only allow lambda_bla increases 
                            tempoutput5[2] = -100
    
                            tempoutput2[0] = -100 # only allow t_bla decreases
                            tempoutput2[1] = -100
    
                            tempoutput8[0] = -100 # only allow v_bla decreases
                            tempoutput8[1] = -100
    
                            tempoutput4[0] = -100 # only allow lambda_ptv decreases
                            tempoutput4[1] = -100
    
                            tempoutput6[0] = -100 # only allow lambda_rec decreases
                            tempoutput6[1] = -100
    
                        if scorerectum<2 and scorerectum<scorebladder:
                            tempoutput6[1] = -100 # only allow lambda_rec increases 
                            tempoutput6[2] = -100
    
                            tempoutput3[0] = -100 # only allow t_rec decreases
                            tempoutput3[1] = -100
    
                            tempoutput9[0] = -100 # only allow v_rec decreases
                            tempoutput9[1] = -100
    
                            tempoutput4[0] = -100 # only allow lambda_ptv decreases
                            tempoutput4[1] = -100
    
                            tempoutput5[0] = -100 # only allow lambda_bla decreases
                            tempoutput5[1] = -100
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


                ##################################### tune parameter according to NN ####################################################
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
                if action != 1:
                    xVec = np.ones((MPTV.shape[1],))
                    gamma = np.zeros((MPTV.shape[0],))
                    state_test, iter, xVec , gamma = \
                            runOpt_dvh(MPTV, MBLA, MREC,tPTV,tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, xVec,gamma,pdose,maxiter)
                planScore_fine,  planScore, scoreall = planIQ(MPTV, MBLA1, MREC1, xVec,pdose)
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
                planScore_all[i + 1] = planScore
                planScore_fine_all[i + 1] = planScore_fine

                if paraidx == 0:
                    logging.info("Step: {}  Iteration: {}  Action: {}  tPTV: {} case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter, action,
                                                                                                    round(tPTV,2), case, round(planScore_fine,3), round(planScore,3)))
                if paraidx == 1:
                    logging.info("Step: {}  Iteration: {}  Action: {}  tBLA: {}  case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter, action,
                                                                                                   round(tBLA,2), case, round(planScore_fine,3), round(planScore,3)))
                if paraidx == 2:
                    logging.info("Step: {}  Iteration: {}  Action: {}  tREC: {}  case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter, action,
                                                                                                    round(tREC,2), case, round(planScore_fine,3), round(planScore,3)))
                if paraidx == 3:
                    logging.info(
                        "Step: {}  Iteration: {}  Action: {}  lambdaPTV: {}  case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter, 
                                 action, round(lambdaPTV,2),
                                                                                                                               case,
                                                                                                                               round(planScore_fine,3),
                                                                                                                      round(planScore,3)))

                if paraidx == 4:
                    logging.info("Step: {}  Iteration: {}  Action: {}  lambdaBLA: {}  case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1, 
                                 iter, action,
                                                                                                    round(lambdaBLA,2), case, round(planScore_fine,3), 
                                                                                                    round(planScore,3)))
                if paraidx == 5:
                    logging.info("Step: {}  Iteration: {}  Action: {}  lambdaREC: {}  case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter, 
                                 action,
                                                                                                    round(lambdaREC,2), case, round(planScore_fine,3), 
                                                                                                    round(planScore,3)))
                if paraidx == 6:
                    logging.info("Step: {}  Iteration: {}  Action: {}  VPTV: {}  case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter, action,
                                                                                                    round(VPTV,4), case, round(planScore_fine,3), round(planScore,3)))
                if paraidx == 7:
                    logging.info("Step: {}  Iteration: {}  Action: {}  VBLA: {}  case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter, action,
                                                                                                   round(VBLA,4), case, round(planScore_fine,3), round(planScore,3)))
                if paraidx == 8:
                    logging.info("Step: {}  Iteration: {}  Action: {}  VREC: {}  case: {} \nPlanScore_fine: {}  PlanScore: {} ".format(i + 1, iter, action,
                                                                                                    round(VREC,4), case, round(planScore_fine,3), round(planScore,3)))
                

                Dose = doseMatrix_test.dot(xVec)
                np.save(data_result_path+id+'xVec' + str(episode) + 'case' +str(case) + 'step' + str(i + 1), xVec)
                np.save(data_result_path+id+'xDose' + str(episode) + 'case' +str(case) + 'step' + str(i + 1), Dose)

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

                Y = np.zeros((1000, 3))
                Y[:, 0] = y_ptv
                Y[:, 1] = y_bladder
                Y[:, 2] = y_rectum

                X = np.zeros((1000, 3))
                X[:, 0] = x_ptv
                X[:, 1] = x_bladder
                X[:, 2] = x_rectum

                np.save(data_result_path+id+'xDVHY' + str(episode) + 'case' +str(case) + 'step' + str(i + 1),
                        Y)
                np.save(data_result_path+id+'xDVHX' + str(episode) + 'case' +str(case) + 'step' + str(i + 1),
                        X)


                if planScore==9:
                    print("parameter tuning is done at step: {} ".format(i+1))
                    break
            if case == 0:
                plt.plot(planScore_all)
                plt.show(block=False)
                plt.pause(5)

            np.save(
                data_result_path+id+'tPTV' + str(episode) + 'case' +str(case),
                tPTV_all)
            np.save(
                data_result_path +id+'tBLA' + str(episode) + 'case' +str(case),
                tBLA_all)
            np.save(
                data_result_path+id+'tREC' + str(episode) + 'case' +str(case),
                tREC_all)
            np.save(
                data_result_path+id+'lambdaPTV' + str(episode) + 'case' +str(case),
                lambdaPTV_all)
            np.save(
                data_result_path+id+'lambdaBLA' + str(episode) + 'case' +str(case),
                lambdaBLA_all)
            np.save(
                data_result_path+id+'lambdaREC' + str(episode) + 'case' +str(case),
                lambdaREC_all)
            np.save(
                data_result_path+id+'VPTV' + str(episode) + 'case' +str(case),
                VPTV_all)
            np.save(
                data_result_path +id+'VBLA' + str(episode) + 'case' +str(case),
                VBLA_all)
            np.save(
                data_result_path+id+'VREC' + str(episode) + 'case' +str(case),
                VREC_all)
            np.save(
                data_result_path+id+'planScore' + str(episode) + 'case' +str(case),
                planScore_all)
            np.save(
                data_result_path+id+'planScore_fine' + str(episode) + 'case' +str(case),
                planScore_fine_all)




