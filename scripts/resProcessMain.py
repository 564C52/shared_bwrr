#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 10:43:54 2021

@author: VictorRosi
"""

import pandas as pd 
from tqdm import tqdm
import seaborn as sns
from os import listdir
from os.path import isfile, join, isdir
from resProcessFunc import *
from SHR_BWS import *
from scipy import stats



resPath = "/Users/VictorRosi/Nextcloud/These_Victor_Rosi/exp2_BWS/MAX/raw_results/"
seqPath = "/Users/VictorRosi/Nextcloud/These_Victor_Rosi/exp2_BWS/MAX/gen_trial/sequences/"
formatPath = "/Users/VictorRosi/Nextcloud/These_Victor_Rosi/exp2_BWS/results_processing/scoring/formatted/"
scorePath = "/Users/VictorRosi/Nextcloud/These_Victor_Rosi/exp2_BWS/results_processing/scoring/new_scores/"

#%% FORMATTING / GATHERING DATA
# remove DS file
files = [f for f in listdir(resPath) if isfile(join(resPath, f))]
if ".DS_Store" in files:
    files.remove(".DS_Store")
    
res_df = pd.DataFrame()
for file in tqdm(files, desc ="Formatting results : "):
    resFile = resPath+file
    tmp_df = formatData(resFile, seqPath, formatPath, save=False)
    res_df = pd.concat([res_df, tmp_df], ignore_index=True)
print(" \n >> All results formatted in 'res_df dataframe' << ")
#res_df = res_df.set_index(['ID'])

#%% TEST SHR
#test_df = res_df.loc[res_df["prof"] == "SE"]
#test_df = test_df.loc[test_df["term"] == "rond"]
test_df = res_df.loc[res_df["term"] == "rugueux"]
dt = getSplitHalf(test_df)

#%% SCORING 
terms = ["brillant","chaud","rond","rugueux"]
prof = ["SE"]
methods = ["Value","Elo","RW","Best","Worst","Unchosen","BestWorst","ABW","David","ValueLogit","RWLogit","BestWorstLogit"] # "EloLogit",
methodChoice = methods[0]
#%%
for i,term in enumerate(terms):
    scoringManagement(term,prof,res_df, scorePath, methodChoice)
print("Done !") 
# in case we export something    
# for i,term in enumerate(terms):
#     print(term)
#     if i == 0:
#         score_data_df = scoringManagement(term,prof,res_df, scorePath, methodChoice)
#         score_data_df = score_data_df.rename(columns = {methodChoice: term})
#     else: 
#         score_data_df[term] = scoringManagement(term,prof,res_df, scorePath, methodChoice)[methodChoice]

#%% Compliance test function
term1_df = pd.read_csv(scorePath+"scores_"+terms[0]+"_"+prof[0]+".csv")[["sounds", "Value"]]
term1_df.columns = ['sounds', terms[0]]
accuracy_df = flagNonCompliance(terms[0], prof, formatPath, term1_df, methodChoice)

#%% Test strategy relations
term ="rugueux"
prof="ALL"

# plot of time response/std_trial 
test_df = trialStratFormatting(res_df, prof, term, scorePath)
#%%
df_analysis = test_df.groupby(["ID","trialNo"]).mean().reset_index()
sns.lmplot(x="n_time_response", y="std_scores", aspect=0.8, scatter_kws={'alpha':0.3}, data=df_analysis)
spear = stats.spearmanr(df_analysis["n_time_response"].values, df_analysis["std_scores"].values)
print(spear)
#%% 
sns.catplot(x="prof", y="nb_listen", hue="choice", hue_order=["best","nc","worst"], data=test_df, kind="bar", aspect=2.5)
#%%
df_analysis = test_df.groupby(["ID","trialNo"]).mean().reset_index()
sns.lmplot(x="nb_listen", y="std_scores", aspect=0.8, scatter_kws={'alpha':0.3}, data=df_analysis)
spear = stats.spearmanr(df_analysis["nb_listen"].values, df_analysis["std_scores"].values)
print(spear)

