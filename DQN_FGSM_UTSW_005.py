
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
import pandas as pd
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
from lib_dvh.FGSM_attack_UTSW_005_50 import evalu_training
#from lib_dvh.StoreModelHistory import StoreModelHistory
logging.basicConfig(filename = '/data/GPU6results/logstuff/',level = logging.INFO,
                    format = '%(asctime)s:%(message)s')

# ------------------------- setting for training and testing  ------------------------------
INPUT_SIZE = 100  # DVH interval number
OUTPUT_SIZE = 3  # number of actions, each lambda has three actions(+,=,-)
TRAIN_NUM = 10 # number of training set
DISCOUNT_RATE = 0.70
REPLAY_MEMORY = 125000
BATCH_SIZE = 16
TARGET_UPDATE_FREQUENCY = 500
MAX_EPISODES = 200
MAX_STEP = 30
load_session = 1# 1 for loading weight from #LoadEpoch; 0 for starting NN from randomn weight
save_session = 1 # 1 for saving the output
Start = 0# 1 for training and 0 for testing
LoadEpoch =65# if load_session is 1, then loading the weight from LoadEpoch
pdose = 4500 # target dose for PTV
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
     #save_session_name ='/data/data/Results/GPU5/session20test/'
   # ---------------store the previous observations in replay memory ------------
     replay_buffer1 = deque(maxlen=REPLAY_MEMORY)
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
            for i in range(1,10):
                globals()["mainDQN"+str(i)] = DQN()
                globals()["mainDQN"+str(i)].build_and_compile_DQN(img_x = INPUT_SIZE, channels_in = 3, channels_out=1, dropout_rate=0, learningRate = 1e-5)
                globals()["targetDQN"+str(i)] = DQN()
                globals()["targetDQN"+str(i)].build_and_compile_DQN(img_x = INPUT_SIZE, channels_in = 3, channels_out=1, dropout_rate=0, learningRate = 1e-5)
                globals()["targetDQN"+str(i)].model.set_weights(globals()["mainDQN"+str(i)].model.get_weights())


        if load_session == 1:
            # loading the weight
            for i in range (1,10):
                globals()["mainDQN"+str(i)].model.load_weights(save_session_name + 'mainDQN'+str(i)+'_episode_' + str(LoadEpoch)+'.h5')
                globals()["targetDQN"+str(i)].model.load_weights(save_session_name + 'targetDQN'+str(i)+'_episode_' + str(LoadEpoch)+'.h5')
            #DPTVex=pd.read_csv('/data/data/testdata/DVH_AI_IMRT_1_PTV.csv')
            # DPTVex.head()
            # DPTVex[DPTVex["Dose"]].head()
            #DBLAex=pd.read_csv('/data/data/testdata/DVH_AI_IMRT_1_Bladder.csv')
            #DBLAex.head(0)
            #DRECex=pd.read_csv('/data/data/testdata/DVH_AI_IMRT_1_Rectum.csv')
            #DRECex.head(0)
            flagg=0
            evalu_training(mainDQN1, mainDQN2, mainDQN3, mainDQN4, mainDQN5, mainDQN6, mainDQN7, mainDQN8, mainDQN9, runOpt_dvh, LoadEpoch,flagg,pdose,maxiter)#,DPTVex,DBLAex,DRECex)

            #bot_play(mainDQN1,mainDQN2,mainDQN3,mainDQN4,mainDQN5,mainDQN6,mainDQN7,mainDQN8,mainDQN9,runOpt_dvh,LoadEpoch,flagg,pdose,maxiter)#episode+1)



if __name__ == "__main__":
	main()
