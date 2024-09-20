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
        num_signal = len(signal)

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
    print(len(predictions))
    print(num_signal)


    sig_hist, _ = np.histogram(predictions[test.get_label().astype(bool)],bins=np.linspace(0,1,nbins),
            density=True);
    bkgd_hist, _ = np.histogram(predictions[~(test.get_label().astype(bool))],bins=np.linspace(0,1,nbins),
            density=True);


    #choose score cuts

            # choose score cuts:\
    
    thresholds = np.linspace(0, 1, 100)

    labels = sampled_data.Label.cat.codes

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
            sig_efficiency = TP / (num_signal)
        else:
            sig_efficiency = 0
        
        
        if (FP + TN) != 0:
            bkgd_efficiency = FP / (num_signal)
        else:
            bkgd_efficiency = 0
        

        S = TP
        B = FP

        # Compute significance using Z = S / sqrt(S + B)
        if (B) > 0:
            significance = S * B**(-1/3)
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


    #setting up histogram
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
    plt.axvline(optimal_threshold, color='y', linestyle='--', label=f'Optimal Thresh: {optimal_threshold:.2f}')
    plt.yscale('log')
    plt.legend(frameon=False)
    plt.savefig(f'./bdt_output/m{args.mass}{args.process}/testing_{args.mass}_{args.process}')


    #cut efficiencies
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, sig_efficiencies, label=f'Signal Efficiency: {optimal_sig:.2f}', color='r')
    plt.plot(thresholds, bkgd_efficiencies, label=f'Background efficiency: {optimal_bkgd:.2f}', color='b')
    #plt.plot(thresholds, significances, label='Significances', color='g')
    plt.axvline(optimal_threshold, color='y', linestyle='--', label=f'Optimal Thresh: {optimal_threshold:.2f}')
    plt.xlabel('BDT score')
    plt.ylabel('Efficiency')
    plt.title('Signal vs. Background Efficiency Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./bdt_output/m{args.mass}{args.process}/eff_{args.mass}_{args.process}')   



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--process", help="Primakoff or Photon Fusion")
    parser.add_argument("--mass", help="ALP mass")
    args = parser.parse_args()
    (args) = parser.parse_args()
    main(args)