


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
logging.basicConfig(filename = '/data/data/Results/GPU1/result_training_withoutrule.log',level = logging.INFO,
                    format = '%(asctime)s:%(message)s')

# ------------------------- setting for training and testing  ------------------------------
INPUT_SIZE = 100  # DVH interval number
OUTPUT_SIZE = 3  # number of actions, each lambda has three actions(+,=,-)
TRAIN_NUM = 10 # number of training set
DISCOUNT_RATE = 0.70
REPLAY_MEMORY = 125000
BATCH_SIZE = 16
TARGET_UPDATE_FREQUENCY = 500
MAX_EPISODES = 15
MAX_STEP = 30
load_session = 0 # 1 for loading weight from #LoadEpoch; 0 for starting NN from randomn weight
save_session = 1 # 1 for saving the output
Start = 1 # 1 for training and 0 for testing
LoadEpoch = 0 # if load_session is 1, then loading the weight from LoadEpoch
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


    save_session_name = '/data/data/Results/GPU2/session/'
   # ---------------store the previous observations in replay memory ------------
    ## ------------------------------- initial 9 NNs for all parameters ----------------------------------------------------
    for i in range(1,10):
        locals ()["replay_buffer"+str(i)] = deque(maxlen=REPLAY_MEMORY)
   # ---------------------------------------------------------------------------

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True ,gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
         ## ------------------------------- initial 9 NNs for all parameters ----------------------------------------------------
        for i in range(1,10):
            globals ()["mainDQN"+str(i)] = DQN()
            globals()["mainDQN"+str(i)].build_and_compile_DQN(img_x = INPUT_SIZE, channels_in = 3, channels_out=1, dropout_rate=0, learningRate = 1e-5)
            globals()["targetDQN"+str(i)] = DQN()
            globals()["targetDQN"+str(i)].build_and_compile_DQN(img_x = INPUT_SIZE, channels_in = 3, channels_out=1, dropout_rate=0, learningRate = 1e-5)
            globals()["targetDQN"+str(i)].model.set_weights(globals()["mainDQN"+str(i)].model.get_weights())


        if load_session == 1:
            # loading the weight
            for i in range (1,10):
                globals()["mainDQN"+str(i)].model.load_weights(save_session_name + 'mainDQN'+str(i)+'_episode_' + str(LoadEpoch)+'.h5')
                globals()["targetDQN"+str(i)].model.load_weights(save_session_name + 'targetDQN'+str(i)+'_episode_' + str(LoadEpoch)+'.h5')

        if Start == 1:
              # --------------------------------------load matrix and mask -------------------------------------------------------------------
            pid=('07','08','09','10','11','12','13','14','15','16')
            for i in range (len(pid)):
                locals()['doseMatrix_'+str(i)] = loadDoseMatrix(r'/home/exx/dose_deposition/prostate_dijs/f_dijs/0'+str(pid[i])+'.hdf5')
                locals()['targetLabels_'+str(i)], locals()['bladderLabel'+str(i)], locals()['rectumLabel'+str(i)],locals()['PTVLabel'+str(i)]= loadMask(r'/data/data/dose_deposition3/plostate_dijs/f_masks/0'+str(pid[i])+'.h5')
                print(locals()['doseMatrix_'+str(i)].shape)


            reward_check = zeros((MAX_EPISODES))
            q_check = zeros((MAX_EPISODES))
            loss_check = zeros((MAX_EPISODES))
            for i in range(1,10):
                globals()['step_count'+str(i)] = 0
            vali_num = 0
            ## -------------------- loop for each episode -------------------------------------------
            for episode in range(MAX_EPISODES):
                reward_sum_total = 0
                reward_f_all = np.zeros((MAX_STEP+1))
                reward_f = 0
                reward_t = 0
                reward_t_all = np.zeros((MAX_STEP+1))
                qvalue_sum = 0
                loss_sum = 0
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


                    doseMatrix = locals()['doseMatrix_'+str(testcase)]
                    targetLabels= locals()['targetLabels_'+str(testcase)]
                    bladderLabel = locals()['bladderLabel'+str(testcase)]
                    rectumLabel = locals()['rectumLabel'+str(testcase)]
                    PTVLabel = locals()['PTVLabel'+str(testcase)]


                    done = False
                    step_count=0
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
                        for i in range(1,10):
                            tempoutput = globals ()['mainDQN'+str(i)].model.predict(np.reshape(state,(1,INPUT_SIZE,3)))
                            locals ()['tempoutput'+str(i)] = tempoutput[0, :]

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
                            for i in range(9):
                                value[i] = np.max(locals()['tempoutput'+str(i+1)])
                            parasel = np.argmax(value)
                            action = np.argmax(locals()['tempoutput'+str(parasel+1)])
                            qvalue_sum = qvalue_sum + value[parasel]
                            num_q += 1
                        ## tune parameter
                        if action == 1:
                            # unchange parameter
                            reward = 0
                            next_state = state
                            # ---------- stop criterion ----------------------------
                            if step_count >= MAX_STEP-1:  # 29
                                done = True
                            step_count += 1



                        elif action != 1:
                            # adjust the parameter
                            if parasel == 0:
                                if action == 0:
                                    action_factor = 1.01
                                    tPTV = min(tPTV * action_factor,paraMax_tPTV)
                                else:
                                    action_factor = 0.09
                                    tPTV = max(tPTV * action_factor,paraMin_tPTV)


                            if parasel == 1:
                                if action == 0:
                                    action_factor = 1.25
                                    tBLA = min(tBLA * action_factor,paraMax_tOAR)
                                else:
                                    action_factor = 0.8
                                    tBLA = max(tBLA * action_factor,paraMin)


                            if parasel == 2:
                                if action == 0:
                                    action_factor = 1.25
                                    tREC = min(tREC * action_factor,paraMax_tOAR)
                                else:
                                    action_factor = 0.8
                                    tREC = max(tREC * action_factor,paraMin)


                            if parasel == 3:
                                if action == 0:
                                    action_factor = m.exp(0.5)
                                    lambdaPTV = min(lambdaPTV * action_factor,paraMax)
                                else:
                                    action_factor = m.exp(-0.5)
                                    lambdaPTV = max(lambdaPTV * action_factor,paraMin)


                            if parasel == 4:
                                if action == 0:
                                    action_factor = m.exp(0.5)
                                    lambdaBLA = min(lambdaBLA * action_factor,paraMax)
                                else:
                                    action_factor = m.exp(-0.5)
                                    lambdaBLA = max(lambdaBLA * action_factor,paraMin)


                            if parasel == 5:
                                if action == 0:
                                    action_factor = m.exp(0.5)
                                    lambdaREC = min(lambdaREC * action_factor,paraMax)
                                else:
                                    action_factor = m.exp(-0.5)
                                    lambdaREC = max(lambdaREC* action_factor,paraMin)


                            if parasel == 6:
                                if action == 0:
                                    action_factor = 1.4
                                    VPTV = min(VPTV * action_factor,paraMax_VPTV)
                                else:
                                    action_factor = 0.6
                                    VPTV = max(VPTV * action_factor,paraMin)


                            if parasel == 7:
                                if action == 0:
                                    action_factor = 1.25
                                    VBLA = min(VBLA * action_factor,paraMax_VOAR)
                                else:
                                    action_factor = 0.8
                                    VBLA = max(VBLA * action_factor,paraMin)


                            if parasel == 8:
                                if action == 0:
                                    action_factor = 1.25
                                    VREC = min(VREC * action_factor, paraMax_VOAR)
                                else:
                                    action_factor = 0.8
                                    VREC = max(VREC * action_factor,paraMin)


                            # ------------------- treatmentplanning optimization -----------------------------------
                            #xVec = np.ones((MPTV.shape[1],))
                            #gamma = np.zeros((MPTV.shape[0],))
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

                                if done==True:
                                    reward_sum_total = reward_sum_total + reward
                                else:
                                    futureReward = zeros((9))
                                    for i in range(9):
                                        futureReward[i] = np.max(globals()['targetDQN'+str(i+1)].model.predict(np.reshape(state,(1,INPUT_SIZE,3))))

                                # futureReward[5] = np.max(targetDQN6.predict(next_state))
                                        reward_sum_total = reward_sum_total + reward + DISCOUNT_RATE * np.max(futureReward)
                                        reward_f=np.max(futureReward)



                                # ------------------------------------------------------
                        locals ()['replay_buffer'+str(int(parasel+1))].append((state, action, reward, next_state, done))

                        # ----------------------- training NN 1-9 ----------------------------------------
                        for i in range (1,10):
                            replaytemp=locals()['replay_buffer'+str(i)]
                            if len(replaytemp) > BATCH_SIZE:
                                if len(replaytemp) > int(REPLAY_MEMORY/10):
                                    train_num_episode = 5
                                elif len(replaytemp) > int(REPLAY_MEMORY/5):
                                    train_num_episode = 20
                                else:
                                    train_num_episode = 1
                                for train_in_episode in range(train_num_episode):
                                   minibatch = rnd.sample(replaytemp, BATCH_SIZE)
                                   minibatch_states = np.reshape(np.vstack([x[0] for x in minibatch]),(BATCH_SIZE,INPUT_SIZE,3))
                                   minibatch_actions = np.vstack([x[1] for x in minibatch])
                                   minibatch_rewards = np.vstack([x[2] for x in minibatch])
                                   minibatch_next_states = np.reshape(np.vstack([x[3] for x in minibatch]),(BATCH_SIZE, INPUT_SIZE, 3))
                                   minibatch_done = np.vstack([x[4] for x in minibatch])

                                   X = minibatch_states
                                   Yout = zeros([BATCH_SIZE, OUTPUT_SIZE * 9])
                                   for j in range(1,10):
                                       Yout[:, 3*(j-1):3*j] =globals()['targetDQN'+str(i)].model.predict(minibatch_next_states)

                                   Q_target = minibatch_rewards[:,0]  + DISCOUNT_RATE * np.max(Yout,axis=1) * ~minibatch_done[:,0]
                                   y = globals ()['mainDQN'+str(i)].model.predict(X)
                                   y[np.arange(len(X)), minibatch_actions[:,0].astype('int64')] = Q_target[np.arange(len(X))]
                                    # y[np.arange(len(X)), 1] = 0
                                   globals ()['mainDQN'+str(i)].model.fit(X, y)
                                globals ()['step_count'+str(i)] += 1
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
                                for i in range(1,10):
                                    globals ()["targetDQN"+str(i)].model.set_weights(globals()["mainDQN"+str(i)].model.get_weights())

                        
                        state = next_state
                        if reward_f > 0 and flag >=e:
                        	reward_f_all[step_count]=reward_f
                        else:
                        	reward_f_all[step_count]=0
                        reward_t_all[step_count]=reward
                        data_out3='/data/data/Results/GPU2/RFsingle/'
                        np.save(data_out3+str(testcase)+'reward_f.npy',reward_f_all)
                        np.save(data_out3+str(testcase)+'reward_total',reward_t_all)
                        if done == True:
                        	np.save(data_out3+'final step'+str(testcase),step_count)
                if num_q !=0:
                    reward_check[episode]=reward_sum_total/num_q
                    q_check[episode] = qvalue_sum/num_q
                    loss_check[episode] = loss_sum/num_q
                else:
                    reward_check[episode]=0
                    q_check[episode]=0
                    loss_check[episode]=0


                print("Episode: {}  Reward: {}  Q-value: {} ".format
                                      (episode + 1, reward_check[episode], q_check[episode]))



                if save_session == 1 and (episode+1)% 5 == 0:
                    for i in range(1,10):
                        data_out = '/data/data/Results/GPU2/checksshort/'
                        data_out2 ='/data/data/Results/GPU2/replayshort/'
                        globals ()["mainDQN"+str(i)].model.save(save_session_name+'mainDQN'+str(i)+'_episode_'+str(episode+1+LoadEpoch)+'.h5')
                        globals ()["targetDQN"+str(i)].model.save(save_session_name + 'targetDQN'+str(i)+'_episode_' + str(episode + 1 + LoadEpoch)+'.h5')
                        np.save(data_out2 +str(i)+'_episode'+str(episode+1),locals()["replay_buffer"+str(i)])
                    np.save(data_out+'reward_check.npy'+str(episode+1),reward_check)
                    np.save(data_out+'q_check.npy' + str(episode + 1), q_check)


                if (episode+1)% 5 == 0:
                    flagg=0
                    vali_num = vali_num + 1
                    plt.figure(vali_num)
                    evalu_training(mainDQN1, mainDQN2, mainDQN3, mainDQN4, mainDQN5, mainDQN6, mainDQN7, mainDQN8, mainDQN9,runOpt_dvh,episode+1,flagg,pdose,maxiter)

        print("Training done!, Test start!")
        flagg=0
        plt.figure(100)
        evalu_training(mainDQN1, mainDQN2, mainDQN3, mainDQN4, mainDQN5, mainDQN6, mainDQN7, mainDQN8, mainDQN9, runOpt_dvh, LoadEpoch,flagg,pdose,maxiter)

#         bot_play(mainDQN1,mainDQN2,mainDQN3,mainDQN4,mainDQN5,runOpt1,episode+1)



if __name__ == "__main__":
	main()
