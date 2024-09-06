import xgboost as xgb
# load the python modules we will use
import uproot # for data loading


import pandas as pd
import matplotlib as mpl # for plotting
import matplotlib.pyplot as plt # common shorthand
from mpl_toolkits.mplot3d import Axes3D
import mplhep # style of plots
import numpy as np
import argparse
import pickle as pkl

mpl.style.use(mplhep.style.ROOT) # set the plot style

from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

nbins = 100
#load the data
def main(args):
    with uproot.open(f"ALP_m{args.mass}_{args.process}_ntuple.root") as f:
        signal = f['Features'].arrays(library='pd')

    with uproot.open("ALP_m__PN_ntuple.root") as f:
        bkgd = f['Features'].arrays(library='pd')

    print(type(signal))


    #assigning labels
    signal['Label'] = 'signal'
    bkgd['Label'] = 'bkgd'

    signal['Label'] = signal.Label.astype('category')
    bkgd['Label'] = bkgd.Label.astype('category')

    print(signal)

    data = pd.concat([bkgd, signal], ignore_index=True)

    data['Label'] = pd.Categorical(data['Label'])

    #want everything except isSignal and label and xs and ys and zs
    features = data.columns[:-1]
    features = features.drop(['ZAv','XYAv','XYWidth',  'isSignal',
                              'e_NHits', 'e_ZLength', 'e_ZAverage_w', 'e_ZWidth_w', 'e_ZAv',
                              'e_ZWidth', 'e_XYAv', 'e_XYAv_w', 'e_XYWidth', 'e_XYWidth_w', 'e_EDensity', 'e_Eav'
                              ])


    data = pd.concat([bkgd, signal], ignore_index=True)

    data['Label'] = pd.Categorical(data['Label'])
    
    X = data
    y = data['Label']


    signal_data = data[data['Label'] == 'signal']
    background_data = data[data['Label'] == 'bkgd']

    param = {}

    # Booster parameters
    param['eta']              = 0.1 # learning rate
    param['max_depth']        = 10  # maximum depth of a tree
    param['subsample']        = 0.8# fraction of events to train tree on
    param['colsample_bytree'] = 1.0 # fraction of features to train tree on

    # Learning task parameters
    param['objective']   = 'binary:logistic' # objective function
    param['eval_metric'] = ['error']        # evaluation metric for cross validation
    #param['sampling_method'] = 'gradient_based'

    num_trees = 100  # number of trees to make

    sig_hists = []
    bkgd_hists = []


    for fold in range(1):

        #sampling 1000 events from each type
        sampled_signal = signal_data.sample(n=10000, random_state=42 + fold, replace=False).reset_index(drop=True)
        sampled_background = background_data.sample(n=10000, random_state=42 + fold, replace=False).reset_index(drop=True)

        # Combine the samples
        sampled_data = pd.concat([sampled_signal, sampled_background]).reset_index(drop=True)

        # Split into training and testing sets, preserving the 1:1 ratio
        train_set, test_set = train_test_split(sampled_data, test_size=0.5, stratify=sampled_data['Label'], random_state=42)

        print(len(test_set))

        train = xgb.DMatrix(data=train_set[features],label=train_set.Label.cat.codes,
                        missing=-999.0,feature_names=features)
        test = xgb.DMatrix(data=test_set[features],label=test_set.Label.cat.codes,
                    missing=-999.0,feature_names=features)
    
        

        booster = xgb.train(param,train,num_boost_round=num_trees)
        output = open(f'test_weights_{args.process}.pkl', 'wb')
        pkl.dump(booster, output)
        print(booster.eval(test))

        predictions = booster.predict(test)
 

            # plot all predictions (both signal and background)


        ax = xgb.plot_importance(booster,grid=False);
        plt.savefig(f'./bdt_output/m{args.mass}{args.process}/ft{fold}')


        # store histograms of signal and background separately
        sig_hist, _ = np.histogram(predictions[test.get_label().astype(bool)],bins=np.linspace(0,1,nbins),
                density=True);
        bkgd_hist, _ = np.histogram(predictions[~(test.get_label().astype(bool))],bins=np.linspace(0,1,nbins),
                density=True);

        print(len(predictions[test.get_label().astype(bool)]))
        sig_hists.append(sig_hist)
        bkgd_hists.append(bkgd_hist)

        # choose score cuts:\
    
    thresholds = np.linspace(0, 1, 100)

    labels = test_set.Label.cat.codes

    sig_efficiencies = []
    bkgd_efficiencies = []
    significances = []
    diffs = []
    optimal_threshold = None
    max_diff = 0
    max2_diff = 0
    optimal_sig = None
    optimal_bkgd = None
    next_thresh = None
    next_sig = None
    next_bkgd = None

    for threshold in thresholds:
    
        predicted_labels = (predictions >= threshold).astype(int)

        #true positives
        TP = np.sum((predicted_labels == 1) & (labels == 1))

        #false neg
        FN = np.sum((predicted_labels == 0) & (labels == 1))

        #true neg
        TN = np.sum((predicted_labels == 0) & (labels == 0))

        #false pos
        FP = np.sum((predicted_labels == 1) & (labels == 0))

        if (TP + FN) != 0:
            sig_efficiency = TP / (5000)
        else:
            sig_efficiency = 0
        
        
        if (FP + TN) != 0:
            bkgd_efficiency = FP / (5000)
        else:
            bkgd_efficiency = 0
        

        S = TP
        B = FP

        # Compute significance using Z = S / sqrt(S + B)
        if ( B) > 0:
            significance = S * (B)**(-1/3)
        else:
            significance = 0  # Handle cases where S + B is 0

        #compute max distance between efficiencies
        
        #diff = sig_efficiency - bkgd_efficiency
        diff = significance
        if diff > max_diff:
            max_diff = diff
            optimal_threshold = threshold
            optimal_sig = sig_efficiency
            optimal_bkgd = bkgd_efficiency
        elif diff > max2_diff:
            max2_diff = diff
            next_thresh = threshold
            next_sig = sig_efficiency
            next_bkgd = bkgd_efficiency





        # Append the significance value
        significances.append(significance)
        diffs.append(diff)
        sig_efficiencies.append(sig_efficiency)
        bkgd_efficiencies.append(bkgd_efficiency)


    #creating hist
    bins = np.linspace(0,1,nbins)
    mean_signal_hist = np.mean(sig_hists, axis=0)
    std_signal_hist = np.std(sig_hists, axis=0)

    mean_background_hist = np.mean(bkgd_hists, axis=0)
    std_background_hist = np.std(bkgd_hists, axis=0)

    bin_centers = (np.linspace(0, 1, nbins)[:-1] + np.linspace(0, 1, nbins)[1:]) / 2

    bin_width = bins[1] - bins[0]
    bin_centers = (bins[:-1] + bins[1:]) / 2

    plt.figure()

    # Plot signal histogram with error bars
    plt.bar(bin_centers, mean_signal_hist, width=bin_width, edgecolor='midnightblue', alpha=0.2, label=f'm{args.mass}_{args.process}_signal',  error_kw=dict(ecolor='black', capsize=3))

    # Plot background histogram with error bars
    plt.bar(bin_centers, mean_background_hist, width=bin_width, edgecolor='firebrick', alpha=0.2, label='background',  error_kw=dict(ecolor='black', capsize=3))
    plt.axvline(optimal_threshold, color='y', linestyle='--', label=f'Optimal Thresh: {optimal_threshold:.2f}')
    plt.xlabel('Prediction from BDT', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.yscale('log')
    plt.legend(frameon=False)
    plt.savefig(f'./bdt_output/m{args.mass}{args.process}/testing_{args.mass}_{args.process}')









    
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, sig_efficiencies, label=f'Signal Efficiency: {optimal_sig:.2f}', color='r')
    plt.plot(thresholds, bkgd_efficiencies, label=f'Background efficiency: {optimal_bkgd:.2f}', color='b')
   # plt.plot(thresholds, significances, label='Significances', color='g')
    plt.axvline(optimal_threshold, color='y', linestyle='--', label=f'Optimal Thresh: {optimal_threshold:.2f}')
    plt.xlabel('BDT score cuts')
    plt.ylabel('Efficiency')
    plt.title('Signal vs. Background Efficiency Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./bdt_output/m{args.mass}{args.process}/eff_{args.mass}_{args.process}')

        # plot efficiency vs. purity (ROC curve)



    #averaging out fold histograms



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--process", help="Primakoff or Photon Fusion")
    parser.add_argument("--mass", help="ALP mass")
    args = parser.parse_args()
    (args) = parser.parse_args()
    main(args)