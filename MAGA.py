# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 09:30:19 2018

@author: Vittorio Fortino
"""

## MAGA - Cineca
"""
A Python implementation of the MAGA algorithm.
Basic usage of the module is very simple:
    >  ()  main function
    
"""

## Needed python modules
from sklearn import cross_validation #, grid_search
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import univariate_selection
from sklearn.feature_selection import f_classif
from collections import deque
from collections import Counter
from multiprocessing import Event, Pipe, Process
import matplotlib.pyplot as plt
from time import time
from operator import itemgetter
from bisect import bisect
from deap import base
from deap import creator
from deap import tools
import numpy as np
import random
import skfuzzy as fuzz
import cPickle as pickle
import pandas as panda
import scipy.cluster.hierarchy as sch
import scipy.sparse as sp
import scipy.spatial as spt
import scipy.stats as sps
import sys, getopt

################################################
#########      FUZZY LOGIC SYSTEM      #########
################################################

######################################################## Generate membership functions
#   <inputs>
fv = np.arange(-0.01, 0.7, 0.001)
ft = np.arange(-0.01, 0.7, 0.001)
mlc = np.arange(0, 200, 1)
ssc = np.arange(-0.01, 1.01, 0.01)
eva = np.arange(0, 1.1, 0.01)

#  <outputs>
cr = np.arange(0, 0.8, 0.01)
mr = np.arange(0, 0.3, 0.01)
pi = np.arange(0.1, 0.4, 0.01)
pd = np.arange(0.1, 0.4, 0.01)
pl = np.arange(0, 0.6, 0.01)
est = np.arange(-0.12, 0.12, 0.01)

# Membership functions for FV
fv_low = fuzz.trapmf(fv,[-0.01, -0.01, 0.0, 0.03])
fv_med = fuzz.trimf(fv, [0.02, 0.05, 0.08])
fv_hig = fuzz.trapmf(fv,[0.07, 0.1, 0.4, 0.7])

# Membership functions for FT
ft_low = fuzz.trapmf(ft,[-0.01, -0.01, 0.005, 0.01])
ft_med = fuzz.trimf(ft, [0.0075, 0.015, 0.03])
ft_hig = fuzz.trapmf(ft,[0.02, 0.04, 0.4, 0.7])

# Membership functions for MLC
mlc_low = fuzz.trapmf(mlc, [0, 0, 5, 15])
mlc_med = fuzz.trimf(mlc, [10, 20, 30])
mlc_hig = fuzz.trapmf(mlc, [25, 35, 200, 200])

# Membership functions for EVA
eva_low = fuzz.trapmf(eva,[0, 0, 0.1, 0.4])
eva_med = fuzz.trimf(eva,[0.2, 0.5, 0.8])
eva_hig = fuzz.trapmf(eva,[0.6, .9, 1.1, 1.1])

# Membership functions for CR
cr_low      = fuzz.trapmf(cr, [0, 0, 0.1, 0.2])
cr_med_low  = fuzz.trimf(cr,  [0.15, 0.25, 0.35])
cr_med      = fuzz.trimf(cr,  [0.3, 0.4, 0.5])
cr_med_high = fuzz.trimf(cr,  [0.45, 0.55, 0.65])
cr_high     = fuzz.trapmf(cr, [0.6, 0.7, 0.8, 0.8])

# Membership functions for MR
mr_low      = fuzz.trapmf(mr, [0, 0, 0.05, 0.08])
mr_med_low  = fuzz.trimf(mr,  [0.07, 0.1, 0.13])
mr_med      = fuzz.trimf(mr,  [0.12, 0.15, 0.18])
mr_med_high = fuzz.trimf(mr,  [0.17, 0.2, 0.23])
mr_high     = fuzz.trapmf(mr, [0.22, 0.25, 0.3, 0.3])

# Membership functions for SSC
ssc_low = fuzz.trapmf(ssc,[-0.01, -0.01, 0, 0.2])
ssc_med = fuzz.trimf(ssc, [0.1, 0.3, 0.5])
ssc_hig = fuzz.trapmf(ssc,[0.4, 0.6, 1.01, 1.01])

# Membership functions for pINSERTION
pi_low      = fuzz.trapmf(pi, [0.1, 0.1, 0.1, 0.15])
pi_med_low  = fuzz.trimf(pi,  [0.125, 0.175, 0.225])
pi_med      = fuzz.trimf(pi,  [0.2, 0.25, 0.3])
pi_med_high = fuzz.trimf(pi,  [0.275, 0.325, 0.375])
pi_high     = fuzz.trapmf(pi, [0.35, 0.4, 0.4, 0.4])

# Membership functions for pDELETION
pd_low      = fuzz.trapmf(pd, [0.1, 0.1, 0.1, 0.15])
pd_med_low  = fuzz.trimf(pd,  [0.125, 0.175, 0.225])
pd_med      = fuzz.trimf(pd,  [0.2, 0.25, 0.3])
pd_med_high = fuzz.trimf(pd,  [0.275, 0.325, 0.375])
pd_high     = fuzz.trapmf(pd, [0.35, 0.4, 0.4, 0.4])

# Membership functions for EST
est_low      = fuzz.trapmf(est, [-0.12, -0.12, -0.1, -0.07])
est_med_low  = fuzz.trimf(est,  [-0.08, -0.05, -0.02])
est_med      = fuzz.trimf(est,  [-0.03, 0, 0.03])
est_med_high = fuzz.trimf(est,  [0.02, 0.05, 0.08])
est_high     = fuzz.trapmf(est, [0.07, 0.1, 0.12, 0.12])

######################################################## Rule application
def intFV(val_FV):
    return dict(low = fuzz.interp_membership(fv, fv_low, val_FV), 
                med = fuzz.interp_membership(fv, fv_med, val_FV), 
                high = fuzz.interp_membership(fv, fv_hig, val_FV))
def intFT(val_FT):
    return dict(low = fuzz.interp_membership(ft, ft_low, val_FT), 
                med = fuzz.interp_membership(ft, ft_med, val_FT), 
                high = fuzz.interp_membership(ft, ft_hig, val_FT))
def intMLC(val_MLC):
    return dict(low = fuzz.interp_membership(mlc, mlc_low, val_MLC), 
                med = fuzz.interp_membership(mlc, mlc_med, val_MLC), 
                high = fuzz.interp_membership(mlc, mlc_hig, val_MLC))
def intSSC(val_SSC):
    return dict(low = fuzz.interp_membership(ssc, ssc_low, val_SSC), 
                med = fuzz.interp_membership(ssc, ssc_med, val_SSC), 
                high = fuzz.interp_membership(ssc, ssc_hig, val_SSC))
def intEVA(val):
    return dict(low = fuzz.interp_membership(eva, eva_low, val), 
                med = fuzz.interp_membership(eva, eva_med, val), 
                high = fuzz.interp_membership(eva, eva_hig, val))
                
def crossoverFLRules (fv_input, mlc_input):  
    FV_class = intFV(fv_input)
    MLC_class = intMLC(mlc_input)
    rule1=np.fmin(FV_class['high'], MLC_class['high'])
    rule2=np.fmin(FV_class['high'], MLC_class['med'])
    rule3=np.fmin(FV_class['high'], MLC_class['low'])
    rule4=np.fmin(FV_class['med'], MLC_class['high'])
    rule5=np.fmin(FV_class['med'], MLC_class['med'])
    rule6=np.fmin(FV_class['med'], MLC_class['low'])
    rule7=np.fmin(FV_class['low'], MLC_class['high'])
    rule8=np.fmin(FV_class['low'], MLC_class['med'])
    rule9=np.fmin(FV_class['low'], MLC_class['low'])
    ################################## Rules activation
    imp1=np.fmin(rule1,cr_high)
    imp2=np.fmin(rule2,cr_med_high)
    imp3=np.fmin(rule3,cr_med_high)
    imp4=np.fmin(rule4,cr_med)
    imp5=np.fmin(rule5,cr_med)
    imp6=np.fmin(rule6,cr_med_low)
    imp7=np.fmin(rule7,cr_med_low)
    imp8=np.fmin(rule8,cr_low)
    imp9=np.fmin(rule9,cr_low)
    ################################## Rule aggregation
    aggregated=np.fmax(imp1, np.fmax(imp2, np.fmax(imp3, np.fmax(imp4, 
                             np.fmax(imp5, np.fmax(imp6, np.fmax(imp7,
                             np.fmax(imp8, imp9))))))))
    ################################## Defuzzification:
    return fuzz.defuzz(cr, aggregated, 'centroid')     

def mutationFLRules (ft_input, mlc_input): 
    FT_class = intFV(ft_input)
    MLC_class = intMLC(mlc_input)
    rule1=np.fmin(MLC_class['high'], FT_class['high'])
    rule2=np.fmin(MLC_class['high'], FT_class['med'])
    rule3=np.fmin(MLC_class['high'], FT_class['low'])
    rule4=np.fmin(MLC_class['med'], FT_class['high'])
    rule5=np.fmin(MLC_class['med'], FT_class['med'])
    rule6=np.fmin(MLC_class['med'], FT_class['low'])
    rule7=np.fmin(MLC_class['low'], FT_class['high'])
    rule8=np.fmin(MLC_class['low'], FT_class['med'])
    rule9=np.fmin(MLC_class['low'], FT_class['low'])
    ################################## Rules activation
    imp1=np.fmin(rule1,mr_med)
    imp2=np.fmin(rule2,mr_med_high)
    imp3=np.fmin(rule3,mr_high)
    imp4=np.fmin(rule4,mr_med_low)
    imp5=np.fmin(rule5,mr_med)
    imp6=np.fmin(rule6,mr_med_high)
    imp7=np.fmin(rule7,mr_low)
    imp8=np.fmin(rule8,mr_med_low)
    imp9=np.fmin(rule9,mr_med)
    ################################## Rule aggregation
    aggregated=np.fmax(imp1, np.fmax(imp2, np.fmax(imp3, np.fmax(imp4, 
                             np.fmax(imp5, np.fmax(imp6, np.fmax(imp7,
                             np.fmax(imp8, imp9))))))))
    ################################## Defuzzification:
    return fuzz.defuzz(mr, aggregated, 'centroid')    

def insertionFLRules (ssc_input, mlc_input): 
    SSC_class = intSSC(ssc_input)
    MLC_class = intMLC(mlc_input)
    ################################## Rules activation (Insertion)
    rule1=np.fmin(SSC_class['high'], MLC_class['low'])
    rule2=np.fmin(SSC_class['high'], MLC_class['med'])
    rule3=np.fmin(SSC_class['high'], MLC_class['high'])
    rule4=np.fmin(SSC_class['med'], MLC_class['low'])
    rule5=np.fmin(SSC_class['med'], MLC_class['med'])
    rule6=np.fmin(SSC_class['med'], MLC_class['high'])
    rule7=np.fmin(SSC_class['low'], MLC_class['low'])
    rule8=np.fmin(SSC_class['low'], MLC_class['med'])
    rule9=np.fmin(SSC_class['low'], MLC_class['high'])
    imp1=np.fmin(rule1,pi_high)
    imp2=np.fmin(rule2,pi_med)
    imp3=np.fmin(rule3,pi_med_low)
    imp4=np.fmin(rule4,pi_med_high)
    imp5=np.fmin(rule5,pi_med_low)
    imp6=np.fmin(rule6,pi_low)
    imp7=np.fmin(rule7,pi_med_low)
    imp8=np.fmin(rule8,pi_low)
    imp9=np.fmin(rule9,pi_low)
    ################################## Rule aggregation
    aggregated=np.fmax(imp1, np.fmax(imp2, np.fmax(imp3, np.fmax(imp4, 
                             np.fmax(imp5, np.fmax(imp6, np.fmax(imp7,
                             np.fmax(imp8, imp9))))))))
    ################################## Defuzzification:
    return fuzz.defuzz(pi, aggregated, 'centroid')

def deletionFLRules (ssc_input, mlc_input): 
    SSC_class = intSSC(ssc_input)
    MLC_class = intMLC(mlc_input)
    rule1=np.fmin(SSC_class['high'], MLC_class['low'])
    rule2=np.fmin(SSC_class['high'], MLC_class['med'])
    rule3=np.fmin(SSC_class['high'], MLC_class['high'])
    rule4=np.fmin(SSC_class['med'], MLC_class['low'])
    rule5=np.fmin(SSC_class['med'], MLC_class['med'])
    rule6=np.fmin(SSC_class['med'], MLC_class['high'])
    rule7=np.fmin(SSC_class['low'], MLC_class['low'])
    rule8=np.fmin(SSC_class['low'], MLC_class['med'])
    rule9=np.fmin(SSC_class['low'], MLC_class['high'])
    ################################## Rules activation
    imp1=np.fmin(rule1,pd_med_low)
    imp2=np.fmin(rule2,pd_med)
    imp3=np.fmin(rule3,pd_high)
    imp4=np.fmin(rule4,pd_low)
    imp5=np.fmin(rule5,pd_med_low)
    imp6=np.fmin(rule6,pd_med_high)
    imp7=np.fmin(rule7,pd_low)
    imp8=np.fmin(rule8,pd_low)
    imp9=np.fmin(rule9,pd_med_low)
    ################################## Rule aggregation
    aggregated=np.fmax(imp1, np.fmax(imp2, np.fmax(imp3, np.fmax(imp4, 
                             np.fmax(imp5, np.fmax(imp6, np.fmax(imp7,
                             np.fmax(imp8, imp9))))))))
    ################################## Defuzzification:
    return fuzz.defuzz(pd, aggregated, 'centroid')    

def mutationOpFLRules (ssc_input, mlc_input):
    pD = deletionFLRules(ssc_input, mlc_input)   
    pI = insertionFLRules(ssc_input, mlc_input) 
    pS = 1 - (pD + pI)
    return [pI, pD, pS]
    
def updateURankFLRules (usa_input, ben_input): 
    USA_class = intEVA(usa_input)
    BENEFIT_class = intEVA(ben_input)
    rule1=np.fmin(USA_class['high'], BENEFIT_class['low'])
    rule2=np.fmin(USA_class['high'], BENEFIT_class['med'])
    rule3=np.fmin(USA_class['high'], BENEFIT_class['high'])
    rule4=np.fmin(USA_class['med'], BENEFIT_class['low'])
    rule5=np.fmin(USA_class['med'], BENEFIT_class['med'])
    rule6=np.fmin(USA_class['med'], BENEFIT_class['high'])
    rule7=np.fmin(USA_class['low'], BENEFIT_class['low'])
    rule8=np.fmin(USA_class['low'], BENEFIT_class['med'])
    rule9=np.fmin(USA_class['low'], BENEFIT_class['high'])
    ################################## Rules activation
    imp1=np.fmin(rule1, est_low)
    imp2=np.fmin(rule2, est_med)
    imp3=np.fmin(rule3, est_med_high)
    imp4=np.fmin(rule4, est_med_low)
    imp5=np.fmin(rule5, est_med_high)
    imp6=np.fmin(rule6, est_high)
    imp7=np.fmin(rule7, est_med_low)
    imp8=np.fmin(rule8, est_med)
    imp9=np.fmin(rule9, est_med_high)
    ################################## Rule aggregation
    aggregated=np.fmax(imp1, np.fmax(imp2, np.fmax(imp3, np.fmax(imp4, 
                             np.fmax(imp5, np.fmax(imp6, np.fmax(imp7,
                             np.fmax(imp8, imp9))))))))
    ################################## Defuzzification:
    defuzz_est = fuzz.defuzz(est, aggregated, 'centroid')  
    return defuzz_est
    
######################################################## Ranking Functions
def compileProbs(input_array):
    if np.unique(input_array).shape[0]==1:
        pass #do thing if the input_array is constant
    else:
        rr = (input_array-np.min(input_array))/np.ptp(input_array)
        rr *= (0.8-0.1)
        rr += .1
    return rr

######################################################## Fitness Function
def evalKFRF(individual, data, nf = 3, nb = 5):
    X = data[0][:,list(individual)]
    y = data[1]
    mean_scores = []
    for param in range(nb):
        scores = []
        # normal cross-validation
        kfolds = cross_validation.StratifiedKFold(y=data[1],
                                                  n_folds=nf,
                                                  shuffle=True,
                                                  random_state=None)
        for train_index, test_index in kfolds:
            # split the training data
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # set and fit random forest model
            cl_rf = RandomForestClassifier(n_estimators=100,oob_score = True,
                                           n_jobs=1, random_state=1,
                                           class_weight = "balanced")
                                           
            #params = {'n_estimators': 100, 'max_depth': 3, 'subsample': 1,
            #          'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 1}
            #cl_rf = GradientBoostingClassifier(**params)
        
            cl_rf.fit(X_train, y_train)
            scores.append(cl_rf.score(X_test, y_test))
        # calculate mean score for folds
        mean_scores.append(np.mean(scores))
    # get average
    value = np.mean(mean_scores)
    return float(value),

######################################################## Delta Penality
def deltaPenality(individual, d):                                             
    #t1 = time()
    return (individual.fitness.values[0]-d),

######################################################## Feasibility Evaluation
def feasible(individual, clusters):
    """Feasability function for the individual. Returns True if feasible False
    otherwise."""
    cls_info = clusters[list(individual)]
    if any([True for k,v in Counter(cls_info).items() if v>1]):
        return False
    return True

def distance(individual, cls):
    """A distance function to the feasability region."""
    cls_info = cls[list(individual)]
    dups = [1*v for k,v in Counter(cls_info).items() if v>1]
    if len(dups) > 0:
        fact = len(individual) - 1
        per_dups = (float(fact-sum(dups)) / float(fact))
        return 1 - per_dups
    else:
        return 0

######################################################## Crossover operator
def cxRankBased(ind1, ind2, w):
    ind1 = list(ind1)                   
    ind2 = list(ind2)  
    w1 = [w[i] for i in ind1]
    w2 = [w[i] for i in ind2]
    s1 = sorted(zip(ind1, w1), key=itemgetter(1), reverse = True)
    s2 = sorted(zip(ind2, w2), key=itemgetter(1), reverse = True)
    ind1 = [x[0] for x in s1]
    w1 = [x[1] for x in s1]
    ind2 = [x[0] for x in s2]
    w2 = [x[1] for x in s2]
    bin1 = np.random.binomial(1, w1, len(w1))
    bin2 = np.random.binomial(1, w2, len(w2))
    if len(ind1) == len(ind2):
        for i in range(len(bin1)):
            if bin1[i] == 1 and bin2[i] == 0:
                ind2[i] = ind1[i]
            if bin1[i] == 0 and bin2[i] == 1:
                ind1[i] = ind2[i]
            if bin1[i] == 0 and bin2[i] == 0:
                if random.random() < 0.5:
                    temp = ind1[i]
                    ind1[i] = ind2[i]
                    ind2[i] = temp
    elif len(ind1) > len(ind2):
        psc1 = 0
        psc2 = 0
        while bin1[psc1] == 1 and psc1 < (len(ind1)-len(ind2)):
            psc1 = psc1 + 1
            #print(psc1)
        if psc1 > (len(ind1)-len(ind2)): psc1 = psc1 - 1
        #print(psc1)
        while psc2 < len(ind2):
            if bin1[psc1] == 0 and bin2[psc2] == 1:
                ind1[psc1] = ind2[psc2]
            if bin1[psc1] == 1 and bin2[psc2] == 0:
                ind2[psc2] = ind1[psc1] 
            if bin1[psc1] == 0 and bin2[psc2] == 0:
                if random.random() < 0.5:
                    temp = ind1[psc1]
                    ind1[psc1] = ind2[psc2]
                    ind2[psc2] = temp
            psc1 = psc1 + 1
            psc2 = psc2 + 1
    else:  
        psc1 = 0
        psc2 = 0
        while bin2[psc2] == 1 and psc2 < (len(ind2)-len(ind1)):
            psc2 = psc2 + 1
            #print(psc2)
        if psc2 > (len(ind2)-len(ind1)): psc2 = psc2 - 1
        #print(psc2)
        while psc1 < len(ind1):
            if bin1[psc1] == 0 and bin2[psc2] == 1:
                ind1[psc1] = ind2[psc2]
            if bin1[psc1] == 1 and bin2[psc2] == 0:
                ind2[psc2] = ind1[psc1] 
            if bin1[psc1] == 0 and bin2[psc2] == 0:
                if random.random() < 0.5:
                    temp = ind1[psc1]
                    ind1[psc1] = ind2[psc2]
                    ind2[psc2] = temp
            psc1 = psc1 + 1
            psc2 = psc2 + 1
    return set(ind1), set(ind2)

######################################################## Mutation operator
def mutRankBased(individual, bm, add_0, add_a, mutp):
    new_items = set() 
    rem_items = set()
    plus = set()
    #chain_ops = ">"
    if sum(bm[list(individual)]) == len(list(individual)): # No features to mutate (add new ones)
        for item in individual:
            if random.random() < mutp[0]:
                if len(add_0) > 0:
                    # select a non active feature
                    plus.add(np.random.choice(add_0, 1)[0])
                elif len(add_a) > 0:
                    # select an active feature
                    plus.add(np.random.choice(add_a, 1)[0])
    else:
        for item in individual:
            if bm[item] == 0:  # Feature to mutate
                op = np.random.choice(['I','D','S'], 1, p = mutp) 
                #chain_ops = chain_ops + str(op)
                #print op
                if op == 'I' or op == 'S':
                    # insert a new feature
                    if len(add_0) > 0:
                        # select a non active feature
                        new_items.add(np.random.choice(add_0, 1)[0])
                    elif len(add_a) > 0:
                        # select an active feature
                        new_items.add(np.random.choice(add_a, 1)[0])
                    if op == 'S':
                        rem_items.add(item)
                else:
                    # delete a feature
                    rem_items.add(item)
    if len(new_items) > 0: [individual.add(x) for x in new_items]
    if len(rem_items) > 0 and len(individual) > len(rem_items): individual.difference_update(rem_items)
    #print chain_ops     
    return individual

######################################################## TO generate random integer vectors based on a given vector of weigths
def weightedChoice(values, weights):
    total = 0
    cum_weights = []
    for w in weights:
        total += w
        cum_weights.append(total)
    x = random.random() * total
    i = bisect(cum_weights, x)
    return values[i]

######################################################## TO calculate the similarity between two chromosomes
def jaccard(st1, st2):
    union = st1.union(st2)
    inter = st1.intersection(st2)
    return (float(len(inter))/float(len(union)))
    
######################################################## TO measure the population similarity
def getSimScore(population):
    sum_sim_score = 0
    ncomp = 0
    for i in range(len(population)):
        for j in range(len(population)):
            if i > j:
                sum_sim_score = sum_sim_score + jaccard(population[i],
                                                        population[j])
                ncomp += 1
    return float(sum_sim_score/ncomp)
    
######################################################## TO update the feature space mask
def updateFSM(fsm, pop, fn):
    big_set = list()
    [big_set.extend(list(s)) for s in pop]
    fr = [0] * fn
    for i in range(fn):
        fr[i] = big_set.count(i)
    fsm = np.vstack([fsm, np.asarray(fr)])
    return fsm
    
######################################################## TO compile the usage-based metric
def findItem(theList, item):
   return [ind for ind in xrange(len(theList)) if item in theList[ind]]
    
def partition(list_, indexes):
    if indexes[0] != 0:
    	indexes = [0] + indexes
    if indexes[-1] != len(list_):
    	indexes = indexes + [len(list_)]
    return [ list_[a:b] for (a,b) in zip(indexes[:-1], indexes[1:])]

def getUsage(feat_history, npop = 10):
    sum_u = np.zeros(feat_history.shape[1])
    for c in range(feat_history.shape[1]):
        perc = feat_history[:,c] / float(npop)
        sum_u[c] = np.mean(perc)
    #print sum[c]
    return sum_u

######################################################## TO compile benefit-based metric
def getBenefit(ff_inv, ff_rest):
    nf = ff_inv.shape[1]
    benefit = np.zeros(nf)
    for i in range(nf):
        x = ff_inv[:,i]
        y = ff_rest[:,i]
        ## by considering only the generations where this feature was involved
        cc = [(x[j],y[j]) for j in range(len(x)) if x[j] != 0]
        if len(cc) > 0:
            benefit[i] = float(sum([xj > yj for xj,yj in cc]))/ float(len(cc)) 
    return benefit
    
######################################################## TO adjust the feature importance
def updateFeatRank(w, fsm_his, ff_inv, ff_rest):
    usage = getUsage(fsm_his)
    benefit = getBenefit(ff_inv, ff_rest)
    #u = np.zeros(len(w))
    for i in range(len(w)):
        w[i] = w[i] + updateURankFLRules(usage[i], benefit[i])
        #u[i] = updateURankFLRules(usage[i], benefit[i])
        if w[i] < 0.1: w[i] = 0.1
        if w[i] > 0.8: w[i] = 0.8
    return w
   
######################################################## Migration Operator
def migPipe(deme, k, pipein, pipeout, selection, replacement=None):
    emigrants = selection(deme, k)
    if replacement is None:
        # If no replacement strategy is selected, replace those who migrate
        immigrants = emigrants
    else:
        # Else select those who will be replaced
        immigrants = replacement(deme, k)
    
    pipeout.send(emigrants)
    buf = pipein.recv()
    
    for place, immigrant in zip(immigrants, buf):
        indx = deme.index(place)
        deme[indx] = immigrant
    
######################################################## IMPORT DATASET FROM R
def percentage(percent, whole):
    return (percent * whole) / 100.0

def load_data_layer(file, rank, cut_cl = 0.25):
    # Load data
    dat = panda.read_csv(file)
    dat.head()
    dat_class = dat['class']
    del dat['class']
    datac = (dat.iloc[:,:].values, dat_class[:].values, (dat.columns).values)
    tdat = np.transpose(datac[0])
    # Compile univariate feature selection
    if np.count_nonzero((tdat!=0) & (tdat!=1)) == 0:
        pvals = [sps.spearmanr(datac[0][:,i],datac[1])[1] for i in range(datac[0].shape[1])]
        pvals = np.where(np.isnan(pvals), 1, pvals)
    else:
        fit_res = univariate_selection.SelectKBest(f_classif).fit(datac[0], datac[1])
        pvals = fit_res.pvalues_
    # Compile clustering
    if np.count_nonzero((tdat!=0) & (tdat!=1)) == 0:
        Z = sch.linkage(sch.distance.pdist(tdat, metric='hamming'), 'average')
        cls = sch.fcluster(Z, .25, criterion = "distance")
    else:
        Z = sch.linkage(tdat, 'average', 'correlation')
        cls = sch.fcluster(Z, cut_cl, criterion = "distance")
    if rank == True:
        datai = (-np.log10(pvals), cls)
    else:
        datai = (np.repeat(0,len(cls)), cls)
    return {'dat': datac, 'info': datai}

####################################### Setting the toolbox for the "sender"           
creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", set, fitness=creator.Fitness)                    
toolbox = base.Toolbox()
toolbox.register("evaluate", evalKFRF, nf = 3, nb = 1)
toolbox.register("feasible", feasible)
toolbox.register("distance", distance)
toolbox.register("penality", deltaPenality)
toolbox.register("mate", cxRankBased)
toolbox.register("mutate", mutRankBased)
toolbox.register("get_add_pb", compileProbs)
toolbox.register("get_ssc", getSimScore)
toolbox.register("get_cxpb", crossoverFLRules)
toolbox.register("get_mupb", mutationFLRules)
toolbox.register("get_mutop", mutationOpFLRules)
toolbox.register("select", tools.selDoubleTournament, fitness_size = 3,
                 parsimony_size = 1.6, fitness_first = False)
                 
def niche(procid, dat, scores, cls, ng, nn, min_len, max_len,
          pipein, pipeout, sync, out_file, seed=None):
    #def main(dat, scores, cls, min_len, max_len):
    random.seed(seed)  
    ngen=ng
    npop=nn
    ## static parameters
    upr_rate=20
    mig_rate=50
    save_rate=20
    pen_factor=0.1
    ## dynamic parameters
    cxpb=0.6
    mutpb=0.1
    mutop=[0.2, 0.2, 0.6]
    best_k=int(percentage(25, npop))
    weights = toolbox.get_add_pb(scores)
    www = toolbox.get_add_pb(scores)
    # Structure initializers
    toolbox.register("attr_feat", weightedChoice, 
                     range(dat[0].shape[1]), weights)
    toolbox.register("individual", 
                 tools.initRepeat, 
                 creator.Individual, 
                 toolbox.attr_feat, 
                 random.randrange(min_len, max_len))
    toolbox.register("population", 
                 tools.initRepeat, 
                 list, 
                 toolbox.individual)
    toolbox.register("migrate", migPipe, k=best_k, pipein=pipein, pipeout=pipeout,
                     selection=tools.selBest, replacement=random.sample)
   
    # Init the population of indiduals
    population = toolbox.population(n=npop)
    
    # Init list of saved populations
    saved_populations = list()

    # Define data structures necessary 
    # to store information about the usage of the features
    fsm = np.zeros(len(weights)) 
    ff_inv_ind = np.zeros(len(weights))
    ff_rest_ind = np.zeros(len(weights))
    
    # Init general statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    
    # Init the logbook
    logbook = tools.Logbook()
    logbook.header = ['gen', 'procid', 'evals', 'vfeats', 'feats0', 'featsA', 
                      'fv', 'ft', 'mlc', 'ssc', 'fes', 'mdist', 'time', 'avg', 'max'] 

    # Evaluate the individuals with an invalid fitness      
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind, data = dat)
        
    # Gather all the fitnesses in one list and print the stats
    fits = [ind.fitness.values[0] for ind in population]
    fes = [toolbox.feasible(ind, cls) for ind in population]
    mdist = np.mean([toolbox.distance(ind, cls) for ind in population])
    crt_mean_fit = sum(fits)/len(population)
    
    # Update the logbook
    # procid = 0
    record = stats.compile(population)
    logbook.record(gen=0, procid=procid, evals=len(population), 
                   vfeats=0, feats0=0, featsA=0, fv=0, ft=0, mlc=0, ssc=0,
                   pc=0, pm=0, pi=0, pd=0, fes=sum(fes), mdist=mdist, 
                   time=0, **record)
   
    if procid == 0:
        # Synchronization needed to log header on top and only once
        print(logbook.stream)
        sync.set()
    else:
        logbook.log_header = False  # Never output the header
        sync.wait()
        print(logbook.stream)
 
    # Update feature space masks
    fsm = updateFSM(fsm, population, len(weights))
    # Update  the information about the usage/utility of the selected features
    eval_ff = np.zeros(fsm.shape[1])
    eval_rr = np.zeros(fsm.shape[1])
    for i, f in enumerate(list(fsm[(fsm.shape[0]-1),:])):
        if f == 0:
            eval_rr[i] = np.average(fits)  
        if f > 0:
            indices = findItem([list(p) for p in population], i)
            eval_ff[i] = np.average(np.asarray(fits)[indices])
            # if this feature is not always used 
            if len(indices) < (len(population) - 1):
                eval_rr[i] = np.average(np.delete(np.asarray(fits), indices))               
    ff_inv_ind  = np.vstack([ff_inv_ind, eval_ff]) 
    ff_rest_ind = np.vstack([ff_rest_ind, eval_rr]) 

    # Begin the generational process
    for gen in range(1, ngen + 1):
        t1 = time()
        # Update the univariate rank
        #if gen % upr_rate == 0 and gen > 0:
            #weights = updateFeatRank(weights, fsm, ff_inv_ind, ff_rest_ind)
            #print('--------')
            #print(len([i for i, val in enumerate(weights) if val > .7]))
            #print(len([i for i, val in enumerate(weights) if val > .5]))
            #print(len([i for i, val in enumerate(weights) if val < .25]))
            #print('--------')
            #totp = sum([True for i,j in zip(weights, www) if i > j])
            #print totp
            #print sum([(i-j) for i,j in zip(weights, www) if i > j])/totp
            #totn = sum([True for i,j in zip(weights, www) if i < j])
            #print totn
            #print sum([(j-i) for i,j in zip(weights, www) if i < j])/totn
 
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        # Clone the selected individuals
        offspring = [toolbox.clone(ind) for ind in offspring]
        # Build the bin mask for the relevant features
        bin_mask = np.random.binomial(1, weights, len(weights))
        # Determine features that have been activated 
        eval_ff = np.zeros(fsm.shape[1])
        eval_rr = np.zeros(fsm.shape[1])
        feats_crt_vis = 0
        feats_to_add_0 = []
        feats_to_add_a = []
        for i, (f, bit) in enumerate(zip(list(fsm[(fsm.shape[0]-1),:]), bin_mask)):
            if f == 0:
                eval_rr[i] = np.average(fits)  
                if bit == 1: feats_to_add_0.append(i)
            if f > 0:
                feats_crt_vis += 1
                indices = findItem([list(p) for p in population], i)
                if len(np.asarray(fits)[indices]) == 0:
                    eval_ff[i] = 0
                else:
                    eval_ff[i] = np.average(np.asarray(fits)[indices])
                if len(np.delete(np.asarray(fits), indices)) == 0:
                    eval_rr[i] = 0
                else:
                    eval_rr[i] = np.average(np.delete(np.asarray(fits), indices))
                if bit == 1: feats_to_add_a.append(i)
        ff_inv_ind  = np.vstack([ff_inv_ind, eval_ff]) 
        ff_rest_ind = np.vstack([ff_rest_ind, eval_rr]) 
        
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # cross two individuals with probability CXPB
            if random.random() < cxpb:
                toolbox.mate(child1, child2, weights)
                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            # mutate an individual with probability MUTPB
            if random.random() < mutpb:
                toolbox.mutate(mutant, bin_mask, 
                               feats_to_add_0, 
                               feats_to_add_a, 
                               mutop)
                del mutant.fitness.values
            
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind, dat)
           
        # The population is entirely replaced by the offspring
        population[:] = offspring
        
        # Update feature space mask and rank values
        fsm = updateFSM(fsm, population, len(weights))
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in population]
        
        # Count how many inidivudals are feasible
        mdist = np.mean([toolbox.distance(ind, cls) for ind in population])       
        not_feasible_ind = [ind for ind in population if not toolbox.feasible(ind, cls)]
        for ind in not_feasible_ind:
            p = toolbox.distance(ind, cls) * pen_factor
            ind.fitness.values = toolbox.penality(ind, p)
            
        # Compile metrics
        fv = (max(fits) - (sum(fits) / len(population))) / max(fits)
        ft = abs((sum(fits) / len(population)) - crt_mean_fit)
        mlc = np.mean([len(i) for i in population])
        ssc = toolbox.get_ssc(population)

        if ssc > 0.75:
            # Re-calculate the CXPB, MUTPB, MUTOP
            cxpb  = 0
            mutpb = 1
            mutop = [0.9,0.1,0]
        else:
            # Re-calculate the CXPB, MUTPB, MUTOP
            cxpb  = toolbox.get_cxpb(fv, mlc)
            mutpb = toolbox.get_mupb(ft, mlc) 
            mutop = toolbox.get_mutop(ssc, mlc)
        
        # Update current mean fitness
        crt_mean_fit = sum(fits)/len(population)

        # Compile time
        t2 = time()
        dt = t2-t1

        # Append the current generation statistics to the logbook
        record = stats.compile(population) 
        logbook.record(gen = gen, 
                       procid = procid,
                       evals = len(invalid_ind), 
                       vfeats = feats_crt_vis, 
                       feats0 = len(feats_to_add_0), 
                       featsA = len(feats_to_add_a),
                       fv = fv, ft = ft, mlc = mlc, ssc = ssc, 
                       fes = len(not_feasible_ind), mdist = mdist,
                       time = dt,
                       **record)           
        
        # send/receive migrants 
        if gen % mig_rate == 0 and gen > 0:
            toolbox.migrate(population)
            
        if gen % save_rate == 0 and gen > 0:
            # append current population
            saved_populations.append(list(population))
            
    # save the outputs 
    f = open(out_file, 'wb')
    for obj in [population, saved_populations, logbook, weights]:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

def readDataResult(pathname, nn = 10):
    ga_out_all = []
    list_all_chr = []
    for i in range(nn):
        f = open(pathname + str(i) + '.pkl', 'rb')
        ga_out = []
        for i in range(4):
            ga_out.append(pickle.load(f))
        ga_out_all.append(ga_out[:4])
        list_all_chr = list_all_chr + ga_out[1]
        f.close()
    return ga_out_all

# import MAGA as ma
# maga_result = ma.readDataResult('data_ccle_erl_ge', nn=10)

    
