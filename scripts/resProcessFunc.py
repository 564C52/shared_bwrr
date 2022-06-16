#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 20:21:32 2021

@author: VictorRosi
"""

import pandas as pd 

from os import listdir
from os.path import isfile, join, isdir
#from spreadsheet import Spreadsheet

def formatData(resFile, seqPath, formatPath, save=True):
    """
    resFile : path to the raw results
    seqFile : path to the sequences folder
    formatPath : path to the formatted data folder (for scoring)
    retest : number of retest 
    
    """
    import pandas as pd
        
    ## takes name of the sequence, and put results in dataframe
    res_df = pd.read_json(resFile)
    seqName = res_df["sequence"][0] # name of the sequence
    res_df = res_df.drop(['sequence'], axis=1)
    res_df = res_df.transpose()
    
    # put sequence in dataframe
    seq_df = pd.read_json(seqPath+seqName).transpose()
    # generate column names, whatever the number of items
    col_names = ["option"+str(i+1) for i in range(seq_df.shape[1])]
    seq_df.columns=col_names 
    
    # few numbers
    nbTrial = seq_df.shape[0] # number of trials
    nbRetest = len(seq_df[seq_df.duplicated()]) # number of retest
    nbTrialEff = nbTrial - nbRetest # number of trial we use for scoring
    
    ## form a large dataframe with all datas 
    # put general data in dataframe
    name = resFile.split('/')[-1].split('_')[0]
    prof = name[0:2]
    term = resFile.split('/')[-1].split('_')[1].split('.')[0]
    cond_df = pd.DataFrame({'ID':name,'prof':prof, 'term':term}, index = range(1))
    cond_df = pd.concat([cond_df]*nbTrial, ignore_index=True)
    
    
    # create common index for the 3 dataframes IMPROVE THAT
    index = list(range(1,seq_df.shape[0]+1))    
    res_df['trialNo'] = index
    seq_df['trialNo'] = index
    cond_df['trialNo'] = index
    # merge dataframes
    res_df = pd.merge(seq_df, res_df, on='trialNo')
    res_df = pd.merge(cond_df, res_df, on='trialNo')
    
    # mean_time data
    mean_time, med_time, total_time = timeElapsed(res_df, False)
    mean_list = [mean_time]*res_df.shape[0]
    res_df["mean_time"] = mean_list
    med_list = [med_time]*res_df.shape[0]
    res_df["med_time"] = med_list

    
    # retest computation
    index_retest = retestEval(res_df)
    index_retest2 = retestEval2(res_df)
    res_df["index_retest"] = [index_retest]*res_df.shape[0]
    res_df["index_retest2"] = [index_retest2]*res_df.shape[0]
    
    if save == True:
        #print("... saving scoring ready data in folder ...")
        # number of trials, retest trials, trials used for scoring

        # create dataframe for scoring dropping the retests
        scoring_data_df = res_df[['best', 'worst', "option1", "option2", "option3", "option4"]].copy()
        scoring_data_df = scoring_data_df[:-nbRetest]
        # writing the data into the file 
        formatFile = formatPath+"data_"+name+"_"+term+".csv"
        scoring_data_df.to_csv(formatFile, index=False)
    
    return res_df


def timeElapsed(res_df, plotting=False):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    ms2s = 1000
    res_df['time'] = res_df['time']/ms2s
    mean_time = int(res_df["time"].mean()) # average time
    med_time = int(res_df["time"].median()) # median time
    total_time = sum(res_df["time"])
    
    if plotting == True: # to be redone
        sns.regplot(x=res_df.index, y="time", data=res_df,
                         scatter_kws={"s": 50},
                         order=3, ci=None, line_kws={"color": "darkblue"})
        plt.title("median time per trial : "+ str(mean_time) +" sec")
        plt.show()
        
    return mean_time, med_time, total_time


def retestEval(res_df):
    import pandas
    from itertools import combinations
    
    # create a dataframe with both choices for test and retest at each row
    duplicate_df1 = res_df[res_df.duplicated(subset=['option1'],keep='first')]
    duplicate_df2 = res_df[res_df.duplicated(subset=['option1'],keep='last')]
    duplicate_df = pd.merge(duplicate_df1, duplicate_df2, on=['option1','option2','option3','option4'])   
    # apply the pair duels winners for computing intra-consistency index
    duplicate_df["nbSimRetest"] = duplicate_df.apply(pairMan, axis=1)  
    # compute index for the 13 retests
    index_retest = duplicate_df["nbSimRetest"].sum()/duplicate_df.shape[0]
       
    return index_retest

def pairMan(group):
    """
    Function that computes indices of consistency for each pairs

    """
    from itertools import combinations
    
    # create all pairs
    options = [group["option1"], group["option2"], group["option3"], group["option4"]]
    pairEff = int(len(options)*(len(options)-1)/2-2) # sur 4 (à vérifier)
    # create dataframe with pairs, bes/worst for test and retest
    pairs_df = pd.DataFrame(list(combinations(options,2)), columns=['pair1', 'pair2'])
    pairs_df["best1"] = group["best_x"]
    pairs_df["worst1"] = group["worst_x"]
    pairs_df["best2"] = group["best_y"]
    pairs_df["worst2"] = group["worst_y"]
    # identify which would win at each pairs
    pairs_df.loc[pairs_df['pair1'] == pairs_df['best1'], 'outcome1'] = pairs_df['pair1']
    pairs_df.loc[pairs_df['pair1'] == pairs_df['worst1'], 'outcome1'] = pairs_df['pair2']
    pairs_df.loc[pairs_df['pair2'] == pairs_df['best1'], 'outcome1'] = pairs_df['pair2']
    pairs_df.loc[pairs_df['pair2'] == pairs_df['worst1'], 'outcome1'] = pairs_df['pair1']
    pairs_df.loc[pairs_df['pair1'] == pairs_df['best2'], 'outcome2'] = pairs_df['pair1']
    pairs_df.loc[pairs_df['pair1'] == pairs_df['worst2'], 'outcome2'] = pairs_df['pair2']
    pairs_df.loc[pairs_df['pair2'] == pairs_df['best2'], 'outcome2'] = pairs_df['pair2']
    pairs_df.loc[pairs_df['pair2'] == pairs_df['worst2'], 'outcome2'] = pairs_df['pair1']

    # Compare the outcomes of test and retest, and count number of common among the pairs  
    pairs_df.loc[pairs_df['outcome1'] == pairs_df['outcome2'], 'comparison'] = True
    pairs_df.loc[pairs_df['outcome1'] != pairs_df['outcome2'], 'comparison'] = False
    # number of common pair outcomes
    nbSim = pairs_df[pairs_df.comparison == True].shape[0]
    
    # print(pairs_df["pair1"], pairs_df["pair2"])
    # print(pairs_df["outcome1"])
    # print(pairs_df["outcome2"])
    # print("Number of similarities : "+str(nbSim))
    if nbSim/pairEff < 1:
        return nbSim/pairEff
    else:
        return 1


def retestEval2(res_df):
    """
    Another intra participant consistency, but just counting best and worst

    """


    # create a dataframe with both choices for test and retest at each row
    duplicate_df1 = res_df[res_df.duplicated(subset=['option1'],keep='first')]
    duplicate_df2 = res_df[res_df.duplicated(subset=['option1'],keep='last')]
    duplicate_df = pd.merge(duplicate_df1, duplicate_df2, on=['option1','option2','option3','option4'])   
    duplicate_df.loc[duplicate_df['best_x'] == duplicate_df['best_y'], 'compBest'] = True
    duplicate_df.loc[duplicate_df['best_x'] != duplicate_df['best_y'], 'compBest'] = False
    duplicate_df.loc[duplicate_df['worst_x'] == duplicate_df['worst_y'], 'compWorst'] = True
    duplicate_df.loc[duplicate_df['worst_x'] != duplicate_df['worst_y'], 'compWorst'] = False
    nbBestSim = duplicate_df[duplicate_df.compBest == True].shape[0]/duplicate_df.shape[0]
    nbWorstSim = duplicate_df[duplicate_df.compWorst == True].shape[0]/duplicate_df.shape[0]
    index_retest2 = (nbBestSim+nbWorstSim)/2
       
    return index_retest2


def scoringManagement(term,prof,res_df, scorePath, methodChoice):
    import scoring # script containing the scoring algorithm
    import pandas as pd
    
    ## preparing the data
    scoring_data_df = res_df.loc[res_df['term'] == term]
    if prof != "ALL":
        scoring_data_df = scoring_data_df.loc[res_df['prof'] == prof]

    # few number for deleting retests
    nbTrial = scoring_data_df['trialNo'].max()
    nbParticipant= scoring_data_df['ID'].nunique()
    nbRetest = int(len(scoring_data_df[scoring_data_df.duplicated(subset=['option1','option2','option3','option4'])])/nbParticipant)
    scoring_data_df = scoring_data_df.loc[res_df['trialNo'] <= nbTrial-nbRetest]
    scoring_data_df = scoring_data_df[['best', 'worst', "option1", "option2", "option3", "option4"]].copy()
    # transforming a dataframe into a list
    scoring_data_list = scoring_data_df.values.tolist()
    
    
    tmpTrials = []
    trials = []
    for trial in scoring_data_list:
        best = trial[0]
        worst = trial[1]
        opt = trial[2:6]
        opt.remove(best)
        opt.remove(worst)
        tmpTrials.append((best, worst, tuple(opt)))
    trials += tmpTrials
    
    
    methods = ["Value","Elo","RW","Best","Worst","Unchosen","BestWorst","ABW","David","ValueLogit","RWLogit","BestWorstLogit"] # "EloLogit",
    results = scoring.score_trials(trials, methods)
    
    # start building table of scored values for each item
    name = 'sounds'
    # gotta find eloMin and eloMax
    #   elos   = [ item.elo for item in results.itervalues() ]
    #   eloMin = min(elos)
    #   eloMax = max(elos)
    
    # print the header and results
    header = [ name ] + methods
    out = []
    #print(",".join(header))
    for name, data in results.items():
        # skip dummy items
        if type(name) != str:
            continue
        
        scores = [ scoring.scoring_methods[method](data) for method in methods ]
        out.append([ name ] + [ str(score) for score in scores ])
        #print(",".join(out))
    
    res = [header]+ out
    score_df = pd.DataFrame(res)
    new_header = score_df.iloc[0] #grab the first row for the header
    score_df = score_df[1:] #take the data less the header row
    score_df.columns = new_header #set the header row as the df header
    # save in csv
    scoreFile = "scores_"+term+"_"+prof+".csv"
    score_df.to_csv(scorePath+scoreFile, index=False)

    # DOESN'T RETURN DATAFRAME FOR NOW BECAUSE DOESN'T WORK
    # for some reason, scores are note encoded in floats, so it messes with the plot later
    # score_export = score_df[["sounds", methodChoice]]
    # score_export[methodChoice] = pd.to_numeric(score_export[methodChoice])
    
    # return score_export      
    
    
    

def flagNonCompliance(term, prof, resPath, score_df, score_method):
    """
    @brief A measurement of the compliance of each participants with the scores obtained.
    Compliance is a measurement of the congruency of the outcome of each pair match 
    computed with the final scores with the outcomes obtained from one participant.
    @example: trial X = [A: 0.9, B:0.5, C:0.3, D:0.7] (scores)
                       [A: 1, B: 0, C: 0, D: -1] (participant Y)

    
    @param pathResults: (string ) path to the Folder containing the formatted results for scoring.
    @param pathScoreFile: (string) path to the score file.
    @param methodInd: (int)

   
    @return accuracy : (float in [0,1])  

    """
    
    # load 
    mypath = resPath
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    if ".DS_Store" in files:
        files.remove(".DS_Store")
    
    
    # scores into dict
    scores = pd.Series(score_df[term].values,index=score_df.sounds).to_dict()


    # go through each file and calculate participant's compliance and log their
    # trials
    compliance  = { }
    paircount   = { }
    user_trials = { }
    
    for file in files:
        # if a participant ID column is not specified, use the name of the file.
        # Otherwise, use values in the specified column to determine ID.
        id_col = None
        prof_file = file.split('_')[1][0:2]
        term_file = file.split('_')[-1].split('.')[0]
              
        if  (term_file == term and prof_file in prof) :   
            # read in the user's data, calculate compliance for each trial
            trials = []
            fileTrial = pd.read_csv(mypath+file).values.tolist()
            for trial in fileTrial:
                best = trial[0]
                worst = trial[1]
                opt = trial[2:6]
                opt.remove(best)
                opt.remove(worst)
                trials.append((best, worst, tuple(opt)))
                
            # generate a user ID based on the file name
            if id_col == None:
                id_col = [ file ] * len(trials)
    
        
            # check for agreement on each trial
            for i in range(len(trials)):
                ID    = id_col[i]
                trial = trials[i]
                best, worst, others = trial
        
                # set default values for participant if needed
                if ID not in compliance:
                    compliance[ID] = 0
                    paircount[ID]  = 0
                    user_trials[ID]= [ ]
        
                # calculate compliance
                #create all pairs in 1 trials from the results files (except for the equality condition ?)
                pairs = [ (best, other) for other in ((worst,) + others) ] + [ (other, worst) for other in others ]
                # compute consistency (check if each score on the left is bigger than the score on the right)
                consistent = [ scores[pair[0]] > scores[pair[1]] for pair in pairs ]
                compliance[ID] += sum(consistent) # add 1 for each True
                paircount[ID]  += len(pairs) # generally equal to 5 
                user_trials[ID].append(trial)
                
    # calculate overall accuracy for each participant
    accuracy = { }
    for user in compliance.keys():
        accuracy[user] = float(compliance[user]) / paircount[user]
    
    # filter for accuracy, define in order to reject participants
    Filter = None
    # print compliance for each person
    if Filter == None:
        # sort users by their accuracy
        users = accuracy.keys()
        users = sorted(users, reverse=False)
        #print("ID,Compliance")
        #for user in users:
            #print("%s,%0.3f" % (str(user), accuracy[user]))
    
            
    # print trials for users that meet the filter threshold
    else:
        user_count = 0
        for user in accuracy.iterkeys():
            # skip by everyone who doesn't make the cut
            if accuracy[user] < Filter:
                continue
    
            # print out data for everyone else
            best,worst,others = user_trials[user][0]
            options = list(others)
    
            # do we need to print the header?
            if user_count == 0:
                optCols = [ "option%d" % (i+1) for i in range(len(options)) ]
                header  = [ "User", "best", "worst" ] + optCols
                print(",".join(header))
    
            # print out each trial for the user
            for trial in user_trials[user]:
                best,worst, others = trial
                options = list(others)
    
                out = [ user, best, worst ] + options
                out = [ str(v) for v in out ]
                print(",".join(out))
    
            # increment our users
            user_count += 1
    
    accuracy_df = pd.DataFrame(list(accuracy.items()), columns=["ID","compliance"])
    accuracy_df["term"] = accuracy_df["ID"].apply(lambda x: x.split('_')[2].split('.')[0])
    accuracy_df["ID"] = accuracy_df["ID"].apply(lambda x: x.split('_')[1])
    accuracy_df["prof"] = accuracy_df["ID"].apply(lambda x: x[0:2])
    
    return accuracy_df




def checkInteractionWords(res_df, ID, nbTrial = 130):
    import itertools
    
    res_df = res_df.loc[res_df["ID"] == ID]
    terms = list(res_df["term"].unique())
    pairs_object = itertools.combinations(terms, 2)
    pairs = list(pairs_object)
    comp_sum_df = pd.DataFrame()
    for pair in pairs:  
        pair = sorted(pair)
        # does not consider retests
        res_df = res_df[res_df["trialNo"] <= nbTrial]
        res_df = res_df[["ID","prof","term","trialNo","option1","option2","option3","option4","best","worst"]]
        
        # index that helps us to recovàer the similar trials for comparison. 
        res_df["simIndex"] = res_df.apply(lambda row: sorted([row["option1"], row["option2"], row["option3"], row["option4"]])[0], axis=1)   
        term1_df = res_df.loc[res_df["term"] == pair[0]]
        term2_df = res_df.loc[res_df["term"] == pair[1]]
        comp_df = pd.merge(term1_df, term2_df, on=["ID","prof","simIndex"])
        #comp_df.loc[(comp_df["best_x"] == comp_df["best_y"] || comp_df["worst_x"] == comp_df["worst_y"]), "sim"] = 1
        #comp_df.loc[(comp_df["best_x"] == comp_df["worst_y"] || comp_df["worst_x"] == comp_df["best_y"]), "inv"] = 1
        comp_df.loc[comp_df["best_x"] == comp_df["best_y"], "simBest"] = 1
        comp_df.loc[comp_df["worst_x"] == comp_df["worst_y"], "simWorst"] = 1
        comp_df.loc[comp_df["best_x"] == comp_df["worst_y"], "BiW"] = 1
        comp_df.loc[comp_df["worst_x"] == comp_df["best_y"], "WiB"] = 1
        #comp_df.loc[comp_df["BiW"] == comp_df["WiB"], "BiW"] = 1
        
        # now replace par the sumvalue   
        tmp_df = pd.DataFrame(comp_df[['simBest','simWorst',"BiW","WiB"]].sum(), columns=["value"])
        tmp_df["value"] = tmp_df["value"].apply(lambda x: x*100/nbTrial)
        tmp_df['simIndex'] = tmp_df.index
        # tmp_df = tmp_df.T
        tmp_df["ID"] = ID
        tmp_df["terms"] = pair[0]+"-"+pair[1]
        #tmp_df["term2"] = pair[1]
        tmp_df["prof"] = res_df["prof"].loc[res_df["ID"] == ID].unique()[0]
        comp_sum_df = comp_sum_df.append(tmp_df)
    return comp_sum_df

def checkInteractionWords2(res_df, ID, nbTrial = 130):
    import itertools
    
    res_df = res_df.loc[res_df["ID"] == ID]
    terms = list(res_df["term"].unique())
    pairs_object = itertools.combinations(terms, 2)
    pairs = list(pairs_object)
    comp_sum_df = pd.DataFrame()
    for pair in pairs:  
        pair = sorted(pair)
        # does not consider retests
        res_df = res_df[res_df["trialNo"] <= nbTrial]
        res_df = res_df[["ID","prof","term","trialNo","option1","option2","option3","option4","best","worst"]]
        
        # index that helps us to recover the similar trials for comparison. 
        res_df["simIndex"] = res_df.apply(lambda row: sorted([row["option1"], row["option2"], row["option3"], row["option4"]])[0], axis=1)   
        term1_df = res_df.loc[res_df["term"] == pair[0]]
        term2_df = res_df.loc[res_df["term"] == pair[1]]
        comp_df = pd.merge(term1_df, term2_df, on=["ID","prof","simIndex"])
        comp_df.loc[(comp_df["best_x"] == comp_df["best_y"]) & (comp_df["worst_x"] == comp_df["worst_y"]), "sim_2"] = 1
        comp_df.loc[(comp_df["best_x"] == comp_df["best_y"]) | (comp_df["worst_x"] == comp_df["worst_y"]), "sim_all"] = 1
        comp_df.loc[((comp_df["best_x"] == comp_df["best_y"]) & (comp_df["worst_x"] != comp_df["worst_y"])) | ((comp_df["best_x"] != comp_df["best_y"]) & (comp_df["worst_x"] == comp_df["worst_y"])), "sim_1"] = 1
        #comp_df.loc[(comp_df["best_x"] != comp_df["best_y"]) & (comp_df["worst_x"] == comp_df["worst_y"]), "simWorst"] = 1
        comp_df.loc[(comp_df["best_x"] == comp_df["worst_y"]) & (comp_df["worst_x"] == comp_df["best_y"]), "inv_2"] = 1
        comp_df.loc[(comp_df["best_x"] == comp_df["worst_y"]) | (comp_df["worst_x"] == comp_df["best_y"]), "inv_all"] = 1
        comp_df.loc[((comp_df["best_x"] == comp_df["worst_y"]) & (comp_df["worst_x"] != comp_df["best_y"])) | ((comp_df["best_x"] != comp_df["worst_y"]) & (comp_df["worst_x"] == comp_df["best_y"])), "inv_1"] = 1
        comp_df.loc[(comp_df["best_x"] != comp_df["worst_y"]) & (comp_df["worst_x"] == comp_df["best_y"]), "inv2"] = 1
        comp_df.loc[(comp_df["best_x"] != comp_df["worst_y"]) & (comp_df["worst_x"] != comp_df["best_y"]) & (comp_df["worst_x"] != comp_df["worst_y"]) & (comp_df["best_x"] != comp_df["best_y"]), "none"] = 1        
        
        # now replace par the sumvalue   
        tmp_df = pd.DataFrame(comp_df[['sim_2','sim_1','sim_all','inv_2','inv_1','inv_all','none']].sum(), columns=["value"])
        tmp_df["value"] = tmp_df["value"].apply(lambda x: x*100/nbTrial)
        tmp_df['simIndex'] = tmp_df.index
        # tmp_df = tmp_df.T
        tmp_df["ID"] = ID
        tmp_df["terms"] = pair[0]+"-"+pair[1]
        #tmp_df["term2"] = pair[1]
        tmp_df["prof"] = res_df["prof"].loc[res_df["ID"] == ID].unique()[0]
        comp_sum_df = comp_sum_df.append(tmp_df)
    return comp_sum_df


def trialStratFormatting(res_df, prof, term, scorePath, tuple_size = 4, time_cutoff = 30):    
    import numpy as np
    from collections import Counter
    
    res_df = res_df.loc[res_df["term"] == term]
    if prof != "ALL":
       res_df = res_df.loc[res_df["prof"] == prof] 
    # delete duplicate retests
    res_df = res_df.drop_duplicates(subset=['term', 'option1','option2','option3','option4'], keep='first')
    strat_df  = (res_df[["trialNo", "option1", "option2", "option3","option4"]].set_index('trialNo', append=True)
                .stack()
                .reset_index()
                .rename(columns={0:'sounds'})
                .drop('level_2',1)
                .drop('level_0',1))
    strat_df["ID"] = np.repeat(list(res_df["ID"]), tuple_size)
    strat_df["time_response"] = np.repeat(list(res_df["time"]), tuple_size)
    strat_df["term"] = np.repeat(list(res_df["term"]), tuple_size)
    listNo = list(np.arange(tuple_size)+1)
    strat_df["soundNo"] =  np.tile(listNo, len(strat_df)//len(listNo))
    strat_df["prof"] = np.repeat(list(res_df["prof"]), tuple_size)
    strat_df['best'] = np.repeat(list(res_df["best"]), tuple_size)
    strat_df['worst'] = np.repeat(list(res_df["worst"]), tuple_size)
    # set choices made for each sound
    strat_df.loc[strat_df["sounds"] == strat_df["best"], 'choice'] = 'best'
    strat_df.loc[strat_df["sounds"] == strat_df["worst"], 'choice'] = 'worst'
    strat_df.loc[(strat_df["sounds"] != strat_df["worst"]) & (strat_df["sounds"] != strat_df["best"]), 'choice'] = 'nc'
    strat_df = strat_df.drop(['best', 'worst'],1)
    
    # Adding the number of listening per sounds
    listening_list_dict = [dict(Counter(row)) for row in list(res_df["listenSeq"])]
    nbListen_df = pd.DataFrame(listening_list_dict)
    trialNo_list = list(np.arange(max(res_df["trialNo"]))+1)
    nbListen_df["trialNo"] = np.tile(trialNo_list, len(nbListen_df)//len(trialNo_list))
    n_nbListen_df = (nbListen_df[["trialNo"]+listNo].set_index('trialNo', append=True)
                    .stack()
                    .reset_index()
                    .rename(columns={0:'nb_listen'})
                    .drop('level_2',1)
                    .drop('level_0',1))
    n_nbListen_df["ID"] = np.repeat(list(res_df["ID"]), tuple_size)
    n_nbListen_df["soundNo"] =  np.tile(listNo, len(n_nbListen_df)//len(listNo))
    strat_df = strat_df.join(n_nbListen_df.set_index(["ID","trialNo","soundNo"]), on=["ID","trialNo","soundNo"])
    
    # adding the scores
    strat_df = strat_df.loc[strat_df["term"] == term]
    term_df = pd.read_csv(scorePath+"scores_"+term+"_"+prof+".csv")[["sounds", "Value"]]
    term_df.columns = ['sounds', term]
    strat_df = strat_df.join(term_df.set_index('sounds'), on="sounds")
    strat_df = strat_df.rename(columns={term: "scores"})
    # std of scores per trial 
    std_df = strat_df.groupby(["ID","trialNo"])["scores"].std()
    std_df = std_df.reset_index()
    std_df = std_df.rename(columns={"scores":"std_scores"})
    strat_df = strat_df.merge(std_df, on=['ID','trialNo'])

    # min and max time response
    strat_df['min_time'] = np.repeat(list(strat_df.groupby(["ID"])["time_response"].min()), len(strat_df.groupby(["sounds"])))
    strat_df['max_time'] = np.repeat(list(strat_df.groupby(["ID"])["time_response"].max()), len(strat_df.groupby(["sounds"])))
    # cutoff outliers
    strat_df = strat_df.loc[strat_df["time_response"] < time_cutoff]
    strat_df['n_time_response'] = (strat_df["time_response"] - strat_df["min_time"])/(strat_df["max_time"] - strat_df["min_time"])
    strat_df = strat_df.drop(["min_time","max_time"], 1)

    return strat_df


def cleanName(name, dynamic,pitch):
    """
    Create a dataframe from the name of sound files

    """
    
    if len(name.split('-')) <= 2:
        name_split = name.split('_')
        name = name_split[0]+"-"+"_".join(name_split[1:-2])+"-"+name_split[-1]+"-"+name_split[-2]
        #name = name_split[0]+"-"+name_split[1]+"-"+name_split[-1]+"-"+name_split[-2]
        #name = name.replace('_','-')
    # elif len(name.split('_')) == 2:
    #     name = name.replace('_','-')
    if name.split('-')[-1] not in dynamic:
        name = '-'.join(name.split('-')[0:-1])
    if name.split('-')[-1] in pitch: 
        name = '-'.join(name.split('-')[0:-1]+['none'])       
    return name 

def addMetaData(names_l):
    import numpy as np
    """
    Create a dataframe from the name of sound files

    """
    
    instruments ={"Vn" : "violin",
              "Va":"alto",
              "Vc":"cello",
              "Cb":"doublebass",
              "Fl":"flute",
              "AFL":"alto_flute",
              "PFL":"piccolo_flute",
              "Ob":"oboe",
              "EH":"english_horn",
              "ClBb":"clarinet",
              "BKL":"bass_clarinet",
              "Bn":"bassoon",
              "KFA":"contrabassoon",
              "ASax":"alto_saxophone",
              "TpC":"trumpet",
              "TpC+H":"trumpet",
              "TpC+C":"trumpet",
              "Hn":"french_horn",
              "Tbn":"trombone",
              "Tbn+H":"trombone",
              "Tbn+C":"trombone",
              "BTb":"bass_tuba",
              "MA":"marimba",
              "Vib":"vibraphone",
              "Xyl":"xylophone",
              "Gsp":"glokenspiel",
              "Hp":"harpa",
              "Gtr":"guitare",
              "Acc":"accordion",
              "Piano":"piano"}
    
    family ={"Vn" : "strings",
          "Va":"strings",
          "Vc":"strings",
          "Cb":"strings",
          "Fl":"woodwinds",
          "AFL":"woodwinds",
          "PFL":"woodwinds",
          "Ob":"woodwinds",
          "EH":"woodwinds",
          "ClBb":"woodwinds",
          "BKL":"woodwinds",
          "Bn":"woodwinds",
          "KFA":"woodwinds",
          "ASax":"woodwinds",
          "TpC":"brass",
          "TpC+H":"brass",
          "TpC+C":"brass",
          "Hn":"brass",
          "Tbn":"brass",
          "Tbn+H":"brass",
          "Tbn+C":"brass",
          "BTb":"brass",
          "MA":"mallets",
          "Vib":"mallets",
          "Xyl":"mallets",
          "Gsp":"mallets",
          "Hp":"other",
          "Gtr":"other",
          "Acc":"other",
          "Piano":"piano"}

    technique = {"ord":"ordinario",
                 "":"ordinario",
                 "L-LV_nA_sus":"ordinario",
                 "L-pA_sus":"ordinario",
                 "L-nA_sus":"ordinario",
                 "L-oV_pA_sus":"ordinario",
                 "nonvib":"non_vibrato",
                 "pont":"sul_ponticello",
                 "harm_artificial":"artificial_harmonic",
                 "pizz_lv":"pizzicato",
                 "pizz":"pizzicato",
                 "pizz_bartok":"pizzicato_bartok",
                 "flatt":"flatterzunge",
                 "FLatter":"flatterzunge",
                 "flatter":"flatterzunge",
                 "Flatter":"flatterzunge",
                 "mul":"multiphonic",
                 "aeolian+ord":"semi_eolian",
                 "stacc":"staccato",
                 "stac":"staccato",
                 "brassy":"brassy",
                 "pedal_tone":"pedal_tone",
                 "play+sing_5th":"sing_play",
                 "play+sing_aug4th":"sing_play",
                 "play+sing_min2nd":"sing_play",
                 "bow_sp":"vib_arco",
                 "ES_Ha_sp-0": "hard_stick",
                 "ES_So_sp-0": "soft_stick",
                 "ES_A-Ha": "hard_stick",
                 "ES_A-So": "soft_stick",
                 "harm_fingering":"harmonic",
                 "ME":"hard_stick",
                 "HH":"soft_stick",
                 "HO":"hard_stick",
                 "GA":"soft_stick"}
    
    dynamic = {"ff":"fortissimo",
               "fp":"fortepiano",
               "f":"forte",
               "f1":"fortissimo",
               "mf":"mezzoforte",
               "mf1":"mezzoforte",
               "mp":"mezzoforte",
               "p":"piano",
               "p1":"piano",
               "pp":"pianissimo",
               "pp1":"pianissimo",
               "none":"none"}
    
    pitch = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]

    term_df = pd.DataFrame(names_l, columns=["sounds"])
    term_df["soundfile2"] = term_df["sounds"].apply(lambda x : x[2:-4])
    term_df["soundname"] = term_df["soundfile2"].apply(lambda x : cleanName(x, dynamic, pitch))
    term_df = term_df.drop(columns="soundfile2")
    term_df["instrument"] = term_df["soundname"].apply(lambda x : instruments[x.split('-')[0]])
    term_df["family"] = term_df["soundname"].apply(lambda x : family[x.split('-')[0]])
    term_df["technique"] = term_df["soundname"].apply(lambda x : technique['-'.join(x.split('-')[1:-2])])
    term_df["dynamic"] = term_df["soundname"].apply(lambda x : dynamic[x.split('-')[-1]])
    term_df["pitch"] = term_df["soundname"].apply(lambda x : x.split('-')[-2])
    term_df["pitch"]=term_df["pitch"].apply(lambda x: x if len(x)<3 else 'multi')
    term_df = term_df.drop(columns="soundname")
    return term_df