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

from sklearn.model_selection import StratifiedKFold, KFold


#load the data
def main(args):
    with uproot.open(f"ALP_m{args.mass}_{args.process}_ntuple.root") as f:
        signal = f['Features'].arrays(library='pd')

    with uproot.open("ALP_mall_PN_ntuple.root") as f:
        bkgd = f['Features'].arrays(library='pd')

    #assigning labels
    signal['Label'] = 'signal'
    bkgd['Label'] = 'bkgd'

    signal['Label'] = signal.Label.astype('category')
    bkgd['Label'] = bkgd.Label.astype('category')



    data = pd.concat([bkgd, signal], ignore_index=True)

    data['Label'] = pd.Categorical(data['Label'])
    print(len(data))
    #want everything except isSignal and label and xs and ys and zs
    features = data.columns[:-2]
    features = features.drop(['Xs', 'Ys', 'Zs', 'Es'])

    data = data.drop_duplicates(features)
    X = data
    y = data['Label']

    skf = StratifiedKFold(n_splits=3)

    param = {}

    # Booster parameters
    param['eta']              = 0.1 # learning rate
    param['max_depth']        = 10  # maximum depth of a tree
    param['subsample']        = 0.8 # fraction of events to train tree on
    param['colsample_bytree'] = 0.8 # fraction of features to train tree on

    # Learning task parameters
    param['objective']   = 'binary:logistic' # objective function
    param['eval_metric'] = ['error']        # evaluation metric for cross validation


    num_trees = 100  # number of trees to make

    sig_hists = []
    bkgd_hists = []
    efficiencies = []
    purities = []

    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"Fold {fold + 1}")
        
        # Splitting data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Check the distribution of labels in each set
        print("Train label distribution:\n", y_train.value_counts())
        print("Test label distribution:\n", y_test.value_counts())
        print("-" * 40)

        train = xgb.DMatrix(data=X_train[features],label=X_train.Label.cat.codes,
                        missing=-999.0,feature_names=features)
        test = xgb.DMatrix(data=X_test[features],label=X_test.Label.cat.codes,
                    missing=-999.0,feature_names=features)
        
        booster = xgb.train(param,train,num_boost_round=num_trees)

        print(booster.eval(test))

        predictions = booster.predict(test)

        # plot all predictions (both signal and background)


        # store histograms of signal and background separately

        sig_hist, _ = np.histogram(predictions[test.get_label().astype(bool)],bins=np.linspace(0,1,30),
                density=True);
        bkgd_hist, _ = np.histogram(predictions[~(test.get_label().astype(bool))],bins=np.linspace(0,1,30),
                density=True);
        
        sig_hists.append(sig_hist)
        bkgd_hists.append(bkgd_hist)



        # choose score cuts:
        cuts = np.linspace(0,1,500);
        nsignal = np.zeros(len(cuts));
        nbackground = np.zeros(len(cuts));
        for i,cut in enumerate(cuts):
            nsignal[i] = len(np.where(predictions[test.get_label().astype(bool)] > cut)[0]);
            nbackground[i] = len(np.where(predictions[~(test.get_label().astype(bool))] > cut)[0]);

        efficiency = nsignal / len(y_test[y_test == 'signal'])
        purity = nsignal / (nsignal + nbackground + 1e-10)  # Small epsilon to avoid division by zero
        
        # Store the results
        efficiencies.append(efficiency)
        purities.append(purity)    


        ax = xgb.plot_importance(booster,grid=False);
        plt.savefig(f'./bdt_output/m{args.mass}_{args.process}_features_{fold + 1}')


    #averaging out fold histograms
    bins = np.linspace(0,1,30)
    mean_signal_hist = np.mean(sig_hists, axis=0)
    std_signal_hist = np.std(sig_hists, axis=0)

    mean_background_hist = np.mean(bkgd_hists, axis=0)
    std_background_hist = np.std(bkgd_hists, axis=0)

    bin_centers = (np.linspace(0, 1, 30)[:-1] + np.linspace(0, 1, 30)[1:]) / 2

    bin_width = bins[1] - bins[0]
    bin_centers = (bins[:-1] + bins[1:]) / 2

    plt.figure()

    # Plot signal histogram with error bars
    plt.bar(bin_centers, mean_signal_hist, width=bin_width, edgecolor='midnightblue', alpha=0.2, label=f'm{args.mass}_{args.process}_signal', yerr=std_signal_hist, error_kw=dict(ecolor='black', capsize=3))

    # Plot background histogram with error bars
    plt.bar(bin_centers, mean_background_hist, width=bin_width, edgecolor='firebrick', alpha=0.2, label='background', yerr=std_background_hist, error_kw=dict(ecolor='black', capsize=3))

    plt.xlabel('Prediction from BDT', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.yscale('log')
    plt.legend(frameon=False)
    plt.savefig(f'./bdt_output/m{args.mass}_{args.process}_sep_avg.pdf')
    plt.show()

    mean_efficiency = np.mean(efficiencies, axis=0)
    mean_purity = np.mean(purities, axis=0)
    std_efficiency = np.std(efficiencies, axis=0)
    std_purity = np.std(purities, axis=0)

    plt.figure()
    plt.plot(mean_efficiency, mean_purity, color='blueviolet',
            label='Mean Efficiency-Purity')
    plt.fill_between(mean_efficiency, mean_purity - std_purity, mean_purity + std_purity,
                    color='blueviolet', alpha=0.2)

    # Plot labels and legend
    plt.xlabel('Efficiency')
    plt.ylabel('Purity')
    plt.legend(loc='lower left')
    plt.savefig(f'./bdt_output/m{args.mass}_{args.process}_roc')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--process", help="Primakoff or Photon Fusion")
    parser.add_argument("--mass", help="ALP mass")
    args = parser.parse_args()
    (args) = parser.parse_args()
    main(args)