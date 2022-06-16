#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 12:12:33 2021

@author: VictorRosi
"""

import scipy as sp
import librosa 
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import crepe
import soundfile as sf

eps = 2.2204*pow(10, -16.0)

def harmonic_rep(mypath, soundFile, sr=44100):
    y, sr = librosa.load(mypath+soundFile, sr=44100)
    S, phase = librosa.magphase(librosa.stft(y=y))
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    H, P = librosa.decompose.hpss(S)
    return H, P

def HPR(mypath, soundFile, sr=44100):
    y, sr = librosa.load(mypath+soundFile, sr=44100)
    S, phase = librosa.magphase(librosa.stft(y=y))
    H, P = librosa.decompose.hpss(S) # harmonic, percussive separation
    S_e = np.power(np.abs(S),2)
    H_e = np.power(np.abs(H),2)
    P_e = np.power(np.abs(P),2)
    H_e_contrib = (np.sum(H_e, axis=0) / np.sum(S_e, axis=0)) * 100
    P_e_contrib = (np.sum(P_e, axis=0) / np.sum(S_e, axis=0)) * 100
    HPR_v=10*np.log10(H_e_contrib/P_e_contrib)
    HPR_v = HPR_v/np.max(HPR_v) # normalized
    HPRMed = np.median(HPR_v)
    if np.isnan(HPRMed):
        HPRMed = 0 
    HPRIqr = sp.stats.iqr(HPR_v)
    if np.isnan(HPRIqr):
        HPRIqr = 0 
    #times = librosa.frames_to_time(np.arange(S.shape[1]))
    #plt.plot(times, HNR_v/np.max(HNR_v))
    return HPRMed, HPRIqr

def mfcc(mypath, soundFile,sr=44100, n_coeff=16):
    y, sr = librosa.load(mypath+soundFile, sr=44100)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_coeff)
    return np.median(mfcc, axis=1)

def HNR(mypath, soundfile):
    # parselmouth sampling rate = 44100
    import parselmouth
    thresh = -20 # dB threshold on which we compute median and iqr [NOT OPTI]
    sound = parselmouth.Sound(mypath+soundfile)
    harm = sound.to_harmonicity()
    harm_vals  = harm.values[0]
    harm_vals = harm_vals[harm_vals > thresh]
    HNRMed = np.median(harm_vals)
    if np.isnan(HNRMed):
        HNRMed = 0 
    HNRIqr = sp.stats.iqr(harm_vals)
    if np.isnan(HNRIqr):
        HNRIqr = 0 

    return HNRMed, HNRIqr

def rms(mypath, soundFile,sr=44100):
    y, sr = librosa.load(mypath+soundFile, sr=44100)
    rms = librosa.feature.rms(y=y)
    rmsMed = np.median(rms)
    if np.isnan(rmsMed):
        rmsMed = 0 
    rmsIqr = sp.stats.iqr(rms)
    if np.isnan(rmsIqr):
        rmsIqr = 0 

    return rmsMed, rmsIqr

def cgsMean(mypath,soundFile, sr=44100, plot=True):
    import scipy as sp
    import librosa 
    import librosa.display
    import numpy as np

    y, sr = librosa.load(mypath+soundFile, sr=44100)
    S, phase = librosa.magphase(librosa.stft(y=y))
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    
    # weighting of centroid with energy 
    alpha = 0.0005
    X = sum(S)
    Energy = np.power(np.abs(X),2) # 
    eff_win = (1-np.exp(-alpha*(Energy))) # alpha*energy between 1200 and 0; --> eff_win between 0 and 1
    cent_new = cent*eff_win

    cutoff = 0.75 # cutoff for the selection of the values for computing the median CGS
    centMed=np.median(cent_new[cent_new > cutoff*max(cent)]) # select value near the maximum cgs (see pizz)  
    if np.isnan(centMed):
        centMed = 0 
    centStd = np.std(cent_new[cent_new > cutoff*max(cent)])
    if np.isnan(centStd):
        centStd = 0 
    centIqr = sp.stats.iqr(cent_new[cent_new > cutoff*max(cent)])
    if np.isnan(centIqr):
        centIqr = 0 
    # pitch 
    # plot
    times = librosa.times_like(cent)
    #duration_considered = round(len(cent_new[cent_new > cutoff*max(cent)])*times[-1]/len(times), 2)

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                             y_axis='log', x_axis='time', ax=ax)
        ax.plot(times, cent.T, label='Spectral centroid', color='w')
        ax.plot(times, cent_new.T, label='Spectral centroid corrected', color='b')   
    return centMed, centStd, centIqr


def zero_cross_rate_librosa(mypath,soundFile, sr=44100):
    filePath = mypath+soundFile
    y, sr = librosa.load(filePath, sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcrMed = np.median(zcr)
    if np.isnan(zcrMed):
        zcrMed = 0 
    zcrStd = np.std(zcr)
    if np.isnan(zcrStd):
        zcrStd = 0 
    zcrIqr = sp.stats.iqr(zcr)
    if np.isnan(zcrIqr):
        zcrIqr = 0 
    return zcrMed, zcrStd, zcrIqr

def spectral_bandwidth_librosa(mypath,soundFile, sr=44100):
    filePath = mypath+soundFile
    y, sr = librosa.load(filePath, sr)
    S = np.abs(librosa.stft(y))
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    bandwidthMed = np.median(bandwidth)
    if np.isnan(bandwidthMed):
        bandwidthMed = 0 
    bandwidthStd = np.std(bandwidth)
    if np.isnan(bandwidthStd):
        bandwidthStd = 0 
    bandwidthIqr = sp.stats.iqr(bandwidth)
    if np.isnan(bandwidthIqr):
        bandwidthIqr = 0 
    return bandwidthMed, bandwidthStd, bandwidthIqr

def spectral_contrast_librosa(mypath,soundFile, sr=44100):
    filePath = mypath+soundFile
    y, sr = librosa.load(filePath, sr)
    S = np.abs(librosa.stft(y))
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    contrastMed = np.median(contrast)
    if np.isnan(contrastMed):
        contrastMed = 0 
    contrastIqr = sp.stats.iqr(contrast)
    if np.isnan(contrastIqr):
        contrastIqr = 0 
    return contrastMed, contrastIqr

def spectral_flatness_librosa(mypath,soundFile, sr=44100):
    filePath = mypath+soundFile
    y, sr = librosa.load(filePath, sr)
    S = np.abs(librosa.stft(y))
    flatness = librosa.feature.spectral_flatness(S=S)
    flatnessMed = np.median(flatness)
    if np.isnan(flatnessMed):
        flatnessMed = 0 
    flatnessIqr = sp.stats.iqr(flatness)
    if np.isnan(flatnessIqr):
        flatnessIqr = 0 
    return flatnessMed, flatnessIqr

def spectral_rolloff_librosa(mypath,soundFile, sr=44100, roll_p = 0.85):
    filePath = mypath+soundFile
    y, sr = librosa.load(filePath, sr)
    S = np.abs(librosa.stft(y))
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=roll_p)
    rolloffMed = np.median(rolloff)
    if np.isnan(rolloffMed):
        rolloffMed = 0 
    rolloffIqr = sp.stats.iqr(rolloff)
    if np.isnan(rolloffIqr):
        rolloffIqr = 0 
    return rolloffMed, rolloffIqr

def spectral_crest(mypath, soundFile, sr=44100, eps=2.2204*pow(10, -16)):
    # from the formula in Peeters et al. (2011)
    y, sr = librosa.load(mypath+soundFile)
    S, phase = librosa.magphase(librosa.stft(y=y))
    K = float(np.shape(S)[0])
    arthMean_v = (1/K) *np.sum(S, axis=0)
    crest = np.max(S, axis=0) / (arthMean_v+eps)
    crestMed = np.median(crest)
    if np.isnan(crestMed):
        crestMed = 0 
    crestIqr = sp.stats.iqr(crest)
    if np.isnan(crestIqr):
        crestIqr = 0 
    return crestMed, crestIqr

def spectral_flatness_ttb(mypath, soundFile, sr=44100, eps=2.2204*pow(10, -16)):
    # from the formula in Peeters et al. (2011)
    # strongly correlats with librosa's flatness
    y, sr = librosa.load(mypath+soundFile)
    S, phase = librosa.magphase(librosa.stft(y=y))
    K = float(np.shape(S)[0])
    geoMean_v = np.exp((1/K) * np.sum(np.log(S+eps), axis=0))
    arthMean_v = (1/K) *np.sum(S, axis=0)
    flatness = geoMean_v/arthMean_v
    flatMed = np.median(flatness)
    if np.isnan(flatMed):
        flatMed = 0 
    flatIqr = sp.stats.iqr(flatness)
    if np.isnan(flatIqr):
        flatIqr = 0 
    return flatMed, flatIqr

def spectral_flux(mypath, soundFile):
    # compute the spectral flux as the sum of square distances
    y, sr = librosa.load(mypath+soundFile)
    S, phase = librosa.magphase(librosa.stft(y=y))
    sp_flux = []
    for i, frame in enumerate(S.T):
        if i >=1:
            prev_frame = S.T[i-1,:]
        else:
            prev_frame = np.zeros(len(frame)) # first frame
        # get fft magnitude
        curr_frame = np.abs(frame)
        prev_frame = np.abs(S.T[i-1,:])
        # normalize fft
        curr_frame = curr_frame / len(curr_frame)
        prev_frame = prev_frame / len(prev_frame)
        # sum 
        curr_frame_sum = np.sum(curr_frame + eps)
        prev_frame_sum = np.sum(prev_frame + eps)
        if i == S.shape[1]-1:
            sp_flux.append(0.0)
        else:
            sp_flux.append(np.sum(
                (curr_frame/curr_frame_sum - prev_frame/prev_frame_sum)**2))  
    sp_flux = sp_flux/max(sp_flux) # normalized      
    fluxMed = np.mean(sp_flux) 
    if np.isnan(fluxMed):
        fluxMed = 0
    fluxStd = np.std(sp_flux)
    if np.isnan(fluxStd):
        fluxStd = 0
    fluxIqr = sp.stats.iqr(sp_flux)
    if np.isnan(fluxIqr):
        fluxIqr = 0
    return fluxMed, fluxStd, fluxIqr


def feature_wrap(mypath, soundfile, sr=44100):
    tmp_lib_df = pd.Series()
    tmp_lib_df["sounds"] = soundfile
    tmp_lib_df["centMed"], tmp_lib_df["centStd"], tmp_lib_df["centIqr"] = cgsMean(mypath, soundfile, plot = False)
    tmp_lib_df["bandwidthMed"], tmp_lib_df["bandwidthStd"], tmp_lib_df["bandwidthIqr"] = spectral_bandwidth_librosa(mypath, soundfile)
    tmp_lib_df["contrastMed"], tmp_lib_df["contrastIqr"] = spectral_contrast_librosa(mypath, soundfile)
    tmp_lib_df["rolloffMed"], tmp_lib_df["rolloffIqr"] = spectral_rolloff_librosa(mypath, soundfile)
    tmp_lib_df["flatnessMed"], tmp_lib_df["flatnessIqr"] = spectral_flatness_librosa(mypath, soundfile)
    tmp_lib_df["zcrMed"], tmp_lib_df["zcrStd"], tmp_lib_df["zcrIqr"] = zero_cross_rate_librosa(mypath, soundfile)
    tmp_lib_df["crestMed"], tmp_lib_df["crestIqr"] = spectral_crest(mypath, soundfile)
    tmp_lib_df["fluxMed"], tmp_lib_df["fluxStd"], tmp_lib_df["fluxIqr"] = spectral_flux(mypath, soundfile)
    tmp_lib_df["HPRMed"], tmp_lib_df["HPRIqr"] = HPR(mypath, soundfile)
    tmp_lib_df["HNRMed"], tmp_lib_df["HNRIqr"] = HNR(mypath, soundfile)
    tmp_lib_df["rmsMed"], tmp_lib_df["rmsIqr"] = rms(mypath, soundfile)
    #tmp_lib_df["flatnessMed_ttb"], tmp_lib_df["flatnessIqr_ttb"] = spectral_flatness_ttb(mypath, soundfile)

    return tmp_lib_df


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


# META DATA 
def meta_data_ex(soundname_l, addMeta = True):
    import numpy as np
    import pandas as pd
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

    impact = ["guitare","harpa","ME","HH","HO","GA","pizz","pizz_lv","pizz_bartok","ES_Ha_sp-0", "ES_So_sp-0", "ES_A-Ha", "ES_A-So"]
    
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

    pitch_miss_d = {
                'n_ASax-ord_C7-ff.wav' : 2093.0, 
                'n_ASax-ord_C6-mf.wav' : 1046.5,
                'n_ASax-ord_C6-ff.wav' : 1046.5,
                'n_Vib_bow_sp-0_kurz_C6.wav' : 1046.5,
                'n_Vib_bow_sp-0_kurz_C4.wav' : 261.6,
                'n_ClBb-ord_C7-pp.wav' : 2093.0,
                'n_Vib_bow_sp-0_kurz_C5.wav' : 523.3,
                'n_ClBb-ord_C7-ff.wav' : 2093.0,
                'n_ASax-ord_C6-pp.wav' : 1046.5
                }
    pitch_d = {'C1': 32.7,
          'C2' : 65.4,
          'C3' : 130.8,
          'C4' : 261.6,
          'C5' : 523.3,
          'C6' : 1046.5,
          'C7' : 2093.0,
          'G7' : 3136.0,
          'C8' : 4186.0,
          'multi' : 0,
          '0' : 0}
    
    meta_df = pd.DataFrame(soundname_l, columns=["sounds"])
    meta_df["soundfile2"] = meta_df["sounds"].apply(lambda x : x[2:-4])
    meta_df["soundname"] = meta_df["soundfile2"].apply(lambda x : cleanName(x, dynamic, pitch))
    meta_df = meta_df.drop(columns="soundfile2")
    meta_df["instrument"] = meta_df["soundname"].apply(lambda x : instruments[x.split('-')[0]])
    meta_df["family"] = meta_df["soundname"].apply(lambda x : family[x.split('-')[0]])
    meta_df["technique"] = meta_df["soundname"].apply(lambda x : technique['-'.join(x.split('-')[1:-2])])
    meta_df["dynamic"] = meta_df["soundname"].apply(lambda x : dynamic[x.split('-')[-1]])
    meta_df["pitch"] = meta_df["soundname"].apply(lambda x : x.split('-')[-2])
    meta_df["pitch"]=meta_df["pitch"].apply(lambda x: x if len(x)<3 else 'multi')
    meta_df = meta_df.replace({"pitch":pitch_d})
    meta_df['pitch2'] = meta_df['sounds'].apply(lambda x : pitch_miss_d[x] if x in list(pitch_miss_d.keys()) else 0)
    meta_df['pitch']=meta_df['pitch2'].where(meta_df['pitch'] == 0, meta_df['pitch'])
    #meta_df['impact']=meta_df["sounds"].apply(lambda x: any(substring in x for substring in impact))
    meta_df = meta_df.drop(columns=["soundname","pitch2"])
    if addMeta == True:
        return meta_df
    else:
        return meta_df[["sounds","pitch"]]





