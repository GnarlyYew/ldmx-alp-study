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


def main(args):

    with uproot.open(f"ALP_m{args.mass}_{args.process}_ntuple.root") as f:
        signal = f['Features'].arrays(library='pd')

    with uproot.open("ALP_m__PN_ntuple.root") as f:
        bkgd = f['Features'].arrays(library='pd')




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
    features = features.drop(['ZAv','XYAv','XYWidth',  'isSignal',
                              'e_NHits', 'e_ZLength', 'e_ZAverage_w', 'e_ZWidth_w', 'e_ZAv',
                              'e_ZWidth', 'e_XYAv', 'e_XYAv_w', 'e_XYWidth', 'e_XYWidth_w', 'e_EDensity', 'e_Eav'])

    data = pd.concat([bkgd, signal], ignore_index=True)

    data['Label'] = pd.Categorical(data['Label'])
    data = data.drop_duplicates(features)
    X = data
    y = data['Label']

    #getting some data to test
    sampled_signal = signal.sample(n=10000, random_state=25, replace=True).reset_index(drop=True)
    sampled_background = bkgd.sample(n=10000, random_state=47, replace=True).reset_index(drop=True)

    # Combine the samples
    sampled_data = pd.concat([sampled_signal, sampled_background]).reset_index(drop=True)
    sampled_data['Label'] = sampled_data['Label'].astype('category')
    nbins=100
    #loading the model
    with open(f'./test_weights_{args.process}.pkl', 'rb') as model_file:
        gbm = pkl.load(model_file)
    

    test = xgb.DMatrix(data=sampled_data[features],label=sampled_data.Label.cat.codes,
                    missing=-999.0,feature_names=features)

    predictions = gbm.predict(test)


    sig_hist, _ = np.histogram(predictions[test.get_label().astype(bool)],bins=np.linspace(0,1,nbins),
            density=True);
    bkgd_hist, _ = np.histogram(predictions[~(test.get_label().astype(bool))],bins=np.linspace(0,1,nbins),
            density=True);

    bins = np.linspace(0,1,nbins)
    bin_centers = (np.linspace(0, 1, nbins)[:-1] + np.linspace(0, 1, nbins)[1:]) / 2

    bin_width = bins[1] - bins[0]
    bin_centers = (bins[:-1] + bins[1:]) / 2

    plt.figure()

    # Plot signal histogram with error bars
    plt.bar(bin_centers, sig_hist, width=bin_width, edgecolor='midnightblue', alpha=0.2, label=f'm{args.mass}_{args.process}_signal',  error_kw=dict(ecolor='black', capsize=3))

    # Plot background histogram with error bars
    plt.bar(bin_centers, bkgd_hist, width=bin_width, edgecolor='firebrick', alpha=0.2, label='background',  error_kw=dict(ecolor='black', capsize=3))

    plt.xlabel('Prediction from BDT', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.yscale('log')
    plt.legend(frameon=False)
    plt.savefig(f'./bdt_output/m{args.mass}{args.process}/testing_{args.mass}_{args.process}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--process", help="Primakoff or Photon Fusion")
    parser.add_argument("--mass", help="ALP mass")
    args = parser.parse_args()
    (args) = parser.parse_args()
    main(args)