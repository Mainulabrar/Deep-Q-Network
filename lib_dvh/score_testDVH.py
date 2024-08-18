import numpy as np
#import time
#import math as m
from math import pi
def planIQ(DPTV,DBLA,DREC,pdose):
    # score of treatment plan, two kinds of scores: 
    # 1: score from standard criterion, 2: score_fined for self-defined in order to emphasize ptv
    #DPTV = MPTV.dot(xVec)
    #DBLA = MBLA.dot(xVec)
    #DREC = MREC.dot(xVec)
    
    DPTV = np.sort(DPTV)
    DPTV = np.flipud(DPTV)
    DBLA = np.sort(DBLA)
    DBLA = np.flipud(DBLA)
    DREC = np.sort(DREC)
    DREC = np.flipud(DREC)

    scoreall = np.zeros((9,))
#    tt = time.time()
    ind = round(0.03/0.015)-1
    a = 1/(1.07*pdose-1.1*pdose)
    b = 1-a*1.07*pdose
    score2 = a*(DPTV[ind]+DPTV[ind+1]+DPTV[ind-1])/3+b
    if score2>1:
        score2=1
    if score2<0:
        score2=0
    delta2 = 0.08
    if (DPTV[ind]+DPTV[ind+1]+DPTV[ind-1])/3>1.07:
        score2_fine = (1 / pi * np.arctan(-((DPTV[ind] + DPTV[ind + 1] + DPTV[ind - 1]) / 3 - (1.07 * pdose + 1.1 * pdose) / 2) / delta2) + 0.5)*8
    else:
        score2_fine=6
    # score2_fine = score2
    scoreall[0]=score2


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
    scoreall[1] = score3

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
    scoreall[2] = score4

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
    scoreall[3] = score5

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
    scoreall[4] = score6


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
    scoreall[5] = score7


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

    scoreall[6] = score8

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

    scoreall[7] = score9

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

    scoreall[8] = score10
#    elapsedTime = time.time()-tt
#    print('time:{}',format(elapsedTime))


    score = score2+score3+score4+score5+score6+score7+score8+score9+score10
    if score2_fine>0.5:
        score_fine = score2_fine + score3_fine + score4_fine + score5_fine + score6_fine + score7_fine + score8_fine + score9_fine + score10_fine
    else:
        score_fine = (score2_fine + score3_fine + score4_fine + score5_fine + score6_fine + score7_fine + score8_fine + score9_fine + score10_fine)

    return score_fine, score, scoreall