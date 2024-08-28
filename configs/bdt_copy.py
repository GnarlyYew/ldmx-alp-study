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

mpl.style.use(mplhep.style.ROOT) # set the plot style

from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

nbins = 100
#load the data
def main(args):
    with uproot.open(f"ALP_m{args.mass}_{args.process}_ntuple.root") as f:
        signal_t = f['Features'].arrays(library='pd')

    with uproot.open("ALP_mall_PN_ntuple.root") as f:
        bkgd_t = f['Features'].arrays(library='pd')


    signal = signal_t[1]
    bkgd = bkgd_t[1]

    #assigning labels
    signal['Label'] = 'signal'
    bkgd['Label'] = 'bkgd'

    signal['Label'] = signal.Label.astype('category')
    bkgd['Label'] = bkgd.Label.astype('category')

    print(signal)

    data = pd.concat([bkgd, signal], ignore_index=True)

    data['Label'] = pd.Categorical(data['Label'])
    print(len(data))
    #want everything except isSignal and label and xs and ys and zs
    features = data.columns[:-1]
    features = features.drop(['Eav_cut_1',
    'Eav_cut_2','Eav_cut_3','XAverage','YAverage','ZAv','XYAv','XYWidth', 'isSignal', 'e_Es', 'e_NHits'])


    data = pd.concat([bkgd, signal], ignore_index=True)

    data['Label'] = pd.Categorical(data['Label'])
    data = data.drop_duplicates(features)
    X = data
    y = data['Label']

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    signal_data = data[data['Label'] == 'signal']
    background_data = data[data['Label'] == 'bkgd']

    param = {}

    # Booster parameters
    param['eta']              = 0.1 # learning rate
    param['max_depth']        = 2  # maximum depth of a tree
    param['subsample']        = 0.5 # fraction of events to train tree on
    param['colsample_bytree'] = 1.0 # fraction of features to train tree on

    # Learning task parameters
    param['objective']   = 'binary:logistic' # objective function
    param['eval_metric'] = ['error']        # evaluation metric for cross validation
    #param['sampling_method'] = 'gradient_based'

    num_trees = 100  # number of trees to make

    sig_hists = []
    bkgd_hists = []


    for fold in range(5):

        #sampling 1000 events from each type
        sampled_signal = signal_data.sample(n=1000, random_state=42 + fold).reset_index(drop=True)
        sampled_background = background_data.sample(n=1000, random_state=42 + fold).reset_index(drop=True)

        # Combine the samples
        sampled_data = pd.concat([sampled_signal, sampled_background]).reset_index(drop=True)

        # Split into training and testing sets, preserving the 1:1 ratio
        train_set, test_set = train_test_split(sampled_data, test_size=0.5, stratify=sampled_data['Label'], random_state=42)


        train = xgb.DMatrix(data=train_set[features],label=train_set.Label.cat.codes,
                        missing=-999.0,feature_names=features)
        test = xgb.DMatrix(data=test_set[features],label=test_set.Label.cat.codes,
                    missing=-999.0,feature_names=features)

        booster = xgb.train(param,train,num_boost_round=num_trees)

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

        sig_hists.append(sig_hist)
        bkgd_hists.append(bkgd_hist)


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

    plt.xlabel('Prediction from BDT', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.yscale('log')
    plt.legend(frameon=False)
    plt.savefig(f'./bdt_output/m{args.mass}{args.process}/testing_{args.mass}_{args.process}')





        # choose score cuts:

        # plot efficiency vs. purity (ROC curve)



    #averaging out fold histograms



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--process", help="Primakoff or Photon Fusion")
    parser.add_argument("--mass", help="ALP mass")
    args = parser.parse_args()
    (args) = parser.parse_args()
    main(args)