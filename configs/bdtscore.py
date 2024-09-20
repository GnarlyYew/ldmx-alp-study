import argparse
import importlib
import ROOT
from ROOT import TTree, TBranch, TH1F, TFile
ROOT.gSystem.Load("/Users/nathanjay/Desktop/SURF/ldmx-sw/install/lib/libFramework.so");
import os
import math
import sys
import csv
import numpy as np
import pandas as pd
from array import array
from optparse import OptionParser
#import matplotlib.pyplot as plt
#sys.path.insert(0, '../')
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import xgboost as xgb
# load the python modules we will use
import uproot # for data loading
import mplhep # style of plots
import pickle as pkl
mpl.style.use(mplhep.style.ROOT) # set the plot style
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split


def main(args):
    with uproot.open(f"ALP_m{args.mass}_{args.process}_ntuple.root") as f:
        signal = f['Features'].arrays(library='pd')
        num_signal = len(signal)

    with uproot.open("ALP_m__PN_ntuple.root") as f:
        bkgd = f['Features'].arrays(library='pd')

    #ofile = TFile('ALP_m'+str(args.mass)+'_'+str(args.process)+'_ntuple.root', 'READ')

    #old = ofile.Get("Features")

    
    nfile = TFile('ALP_m'+str(args.mass)+'_'+str(args.process)+'_scored_ntuple.root', 'RECREATE')
    ntree = TTree('Features', 'Features of test set')
    #new = old.CloneTree(0)
    

    #score branch
    BdtScore = array('f', [0])
    ntree.Branch("BdtScore", BdtScore, 'BdtScore/F')


    #run bdt
    



    #assigning labels
    signal['Label'] = 'signal'
    bkgd['Label'] = 'bkgd'

    signal['Label'] = signal.Label.astype('category')
    bkgd['Label'] = bkgd.Label.astype('category')


    data = pd.concat([bkgd, signal], ignore_index=True)

    data['Label'] = pd.Categorical(data['Label'])

    #want everything except isSignal and label and xs and ys and zs
    features = data.columns
    features = features.drop(['ZAv','XYAv','XYWidth',  'isSignal',
                              'e_NHits', 'e_ZLength', 'e_ZAverage_w', 'e_ZWidth_w', 'e_ZAv',
                              'e_ZWidth', 'e_XYAv', 'e_XYAv_w', 'e_XYWidth', 'e_XYWidth_w', 'e_EDensity', 'e_Eav', 'Label'])

    data = pd.concat([bkgd, signal], ignore_index=True)

    data['Label'] = pd.Categorical(data['Label'])
    data = data.drop_duplicates(features)
    X = data
    y = data['Label']

    #getting some data to test
    sampled_signal = signal
    sampled_background = bkgd.sample(n=num_signal, random_state=47, replace=False).reset_index(drop=True)

    # Combine the samples
    sampled_data = pd.concat([sampled_signal, sampled_background]).reset_index(drop=True)
    sampled_data['Label'] = sampled_data['Label'].astype('category')
   

    nbins=100
    #loading the model
    with open(f'./test_weights_{args.process}.pkl', 'rb') as model_file:
        gbm = pkl.load(model_file)
    

    test = xgb.DMatrix(data=sampled_data[features],label=sampled_data.Label.cat.codes,
                    missing=-999.0,feature_names=features, enable_categorical=True)

    predictions = gbm.predict(test)


    scores = predictions
    print(len(scores))
    print(num_signal)
    #get scores for each event????
 #fill the tree
    data_tree = sampled_data
    data_tree['Label'] = data_tree['Label'].map({'signal': 1.0, 'bkgd': 0.0})

    print(data_tree)
    branches = {}
    for column in data_tree.columns:
        branches[column] = array('f', [0])
        ntree.Branch(column, branches[column], f'{column}/F')
    

    for (i), (_, row) in enumerate(data_tree.iterrows()):

        for column in data_tree.columns:
            branches[column][0] = row[column]
        
        BdtScore[0] = scores[i]


        ntree.Fill()
    
    ntree.Write()

    nfile.Close()
    #ofile.Close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--process", help="Primakoff or Photon Fusion")
    parser.add_argument("--mass", help="ALP mass")
    args = parser.parse_args()
    (args) = parser.parse_args()
    main(args)

