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
from array import array
from optparse import OptionParser
#import matplotlib.pyplot as plt
#sys.path.insert(0, '../')
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def loop(self):
    f = TFile('ALP_m'+str(self.mass)+'_'+self.label+'_ntuple.root', 'UPDATE')