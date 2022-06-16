#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 15:49:07 2021

@author: VictorRosi
"""

# def dataset_build(attribute, profil, total_df, target_tot_df, X_meta, metadata=True, scale=True):
#     import pandas  as pd
#     from sklearn.preprocessing import StandardScaler

#     # function for choosing the preset of result scores
#     target_name = profil+"_"+attribute
#     y = total_df[target_name]
#     #print(target_name)
        
#     if metadata == True:
#         total_df = total_df.merge(X_meta, on="sounds")
#     else: # still add pitch, because !
#         total_df = total_df.merge(X_meta[["sounds","pitch"]], on="sounds") 
#     rm_target =  list(target_tot_df.columns)
#     X = total_df.drop(rm_target, axis=1)
#     if scale == True: # useless for RF, but useful for mrmr and lasso
#         scaled_features = StandardScaler().fit_transform(X.values)
#         X = pd.DataFrame(scaled_features, index=X.index, columns=X.columns)
    
#     return X, y

def checkMultColl(X, save=True):
    from scipy.stats import spearmanr
    from scipy.cluster import hierarchy
    import matplotlib.pyplot as plt
    import numpy as np
    from  matplotlib.backends.backend_pdf import PdfPages
    
    with PdfPages('figure/multicol.pdf') as pdf:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        corr = spearmanr(X).correlation
        corr_linkage = hierarchy.ward(corr)
        dendro = hierarchy.dendrogram(
            corr_linkage, labels=X.columns.tolist(), ax=ax1, leaf_rotation=90)
        dendro_idx = np.arange(0, len(dendro['ivl']))
        ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']], cmap="viridis", interpolation='nearest')
        ax2.set_xticks(dendro_idx)
        ax2.set_yticks(dendro_idx)
        ax2.set_xticklabels(dendro['ivl'], rotation='vertical', fontsize = 12)
        ax2.set_yticklabels(dendro['ivl'], fontsize = 12)
        plt.grid(None)
        fig.tight_layout()
        plt.show()

        if save==True:
            pdf.savefig(fig, bbox_inches='tight')


def dataset_build(attribute, profil, feat_df, meta_df, target_df, meta=True, scale=True):
    import pandas  as pd
    from sklearn.preprocessing import StandardScaler
    # function for choosing the preset of result scores
    target_name = profil+"_"+attribute
    
    # making sure indexing is the same for feature and target
    total_df = pd.merge(feat_df, target_df[["sounds",target_name]], on="sounds") 
    # target
    y = total_df[target_name]
    
    if meta == True:
        total_df = total_df.merge(meta_df, on="sounds")
    # feature
    X = total_df.drop(["sounds", target_name], axis=1)
    if scale == True: # useless for RF, but useful for mrmr and lasso
        scaled_features = StandardScaler().fit_transform(X.values)
        X = pd.DataFrame(scaled_features, index=X.index, columns=X.columns)    
    return X, y



def rf_plot(X, shap_values):
    import matplotlib. pyplot as plt 
    import seaborn as sns 
    fig = plt.figure(figsize=(5,15))
    plt.subplot(1,2,1)
    shap.summary_plot(shap_values, X_test, show=False)
    fig1 = plt.gcf()
    ax1 = plt.gca()
    fig1.set_figwidth(20)
    plt.subplot(1,2,2)
    shap.summary_plot(shap_values, X, feature_names=X.columns, show=False, plot_type='bar')
    # Get the current figure and axes objects.
    fig2 = plt.gcf()
    ax2 = plt.gca()
    fig2.set_figwidth(14)
    ax2.get_yaxis().set_visible(False)
    ax2.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelleft=False) # labels along the bottom edge are off
    return fig
