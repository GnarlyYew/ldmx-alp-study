#!/usr/bin/python

# ldmx python3 MakeRootTree.py --ifile reco.root
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


class GetPart:

    def __init__(self, fn1, ofn, label, mass, tag):

        self.label = label
        self.mass = mass
        #input files:
        self.fin1 = ROOT.TFile(fn1);
        self.tin1 = self.fin1.Get("LDMX_Events")
        self.tag = int(tag);

        # output files:
        #self.fn_out = ofn;
        #self.fout = ROOT.TFile("hist_"+self.fn_out,"RECREATE");

        #list of branches:
        self.evHeader1 = ROOT.ldmx.EventHeader()
        self.hcalRecHits = ROOT.std.vector('ldmx::HcalHit')();
        self.ecalRecHits = ROOT.std.vector('ldmx::EcalHit')();
        self.tin1.SetBranchAddress("EventHeader",  ROOT.AddressOf( self.evHeader1 ));
        self.tin1.SetBranchAddress("HcalRecHits_v14",  ROOT.AddressOf( self.hcalRecHits ));
        self.tin1.SetBranchAddress("EcalRecHits_v14",  ROOT.AddressOf( self.ecalRecHits ));

        # loop and save:
        self.loop();


    def loop(self):
        f = TFile('ALP_m'+str(self.mass)+'_'+self.label+'_ntuple_ecal.root', 'RECREATE')
        Features = TTree( 'Features', 'Information about events' )

        NHits = array('f',[0])
        Features.Branch("NHits",  NHits,  'NHits/F')
        ZLength = array('f',[0])
        Features.Branch("ZLength",  ZLength,  'ZLength/F')
        ZAverage_w = array('f',[0])
        Features.Branch("ZAverage_w",  ZAverage_w,  'ZAverage_w/F')


        Ys = ROOT.std.vector('float')()
        Features.Branch("Ys", Ys)
        Xs = ROOT.std.vector('float')()
        Features.Branch("Xs", Xs)
        Zs = ROOT.std.vector('float')()
        Features.Branch('Zs', Zs)
        Es = ROOT.std.vector('float')()
        Features.Branch("Es", Es)

        ZWidth_w = array('f',[0])
        Features.Branch("ZWidth_w",  ZWidth_w,  'ZWidth_w/F')

        ZAv = array('f',[0])
        Features.Branch("ZAv", ZAv, 'ZAv/F')

        ZWidth = array('f',[0])
        Features.Branch("ZWidth", ZWidth, 'ZWidth/F')

        XYAv = array('f',[0])
        Features.Branch("XYAv",  XYAv,  'XYAv/F')

        XYAv_w = array('f',[0])
        Features.Branch("XYAv_w", XYAv_w, 'XYAv_w/F')

        XYWidth = array('f',[0])
        Features.Branch("XYWidth",  XYWidth,  'XYWidth/F')

        XYWidth_w =  array('f',[0])
        Features.Branch("XYWidth_w", XYWidth_w, 'XYWidth_w/F')

        Eav = array('f',[0])
        Features.Branch("Eav",  Eav,  'Eav/F')

        EDensity = array('f',[0])
        Features.Branch("EDensity",  EDensity,  'EDensity/F')


        isSignal = array('i',[0])
        Features.Branch("isSignal",  isSignal,  'isSignal/I')

        nent = self.tin1.GetEntriesFast();

        for i in range(nent):
            self.tin1.GetEntry(i);
            NHits[0] = 0
            ZLength[0] = 0.
            ZAverage_w[0] = 0.
            ZWidth_w[0] = 0.
            ZAv[0] = 0.
            ZWidth[0] = 0.
            XYAv[0] = 0.
            XYAv_w[0] = 0.
            Eav[0] = 0.
            EDensity[0] = 0.
            isSignal[0] = 0
            sumE = 0

            x_positions = []
            y_positions = []
            z_positions = []
            distances = []
            weighted_z = []
            weighted_dist = []
            energies = []
            weights = []


            Xs.clear()
            Ys.clear()
            Zs.clear()
            Es.clear()


            for ih,hit in enumerate(self.ecalRecHits):
                NHits[0] += 1
                energies.append(hit.getEnergy())
                x_positions.append(hit.getXPos())
                y_positions.append(hit.getYPos())
                z_positions.append(hit.getZPos())
                sumE += hit.getEnergy()

            if len(energies) != 0:
            # for Z length
                first_z = np.min(z_positions)
                last_z = np.max(z_positions)


            # loop over hits, weight by fraction of energy in that hit  
                for p,q in enumerate(energies):
                    w = q/sumE
                    weights.append(w)
                    w_z = z_positions[p]*w
                    weighted_z.append(w_z)
                
                
                    
                # loop over hits,get transverse shower Information
                for p,q in enumerate(x_positions):
                    distance = math.sqrt(x_positions[p]*x_positions[p] + y_positions[p]*y_positions[p])
                    distances.append(distance)
                    weighted_dist.append(distance*energies[p]/sumE)
                    Xs.push_back(x_positions[p])
                    Ys.push_back(y_positions[p])
                    Zs.push_back(z_positions[p])
                    Es.push_back(energies[p])

            
                z_variations = []
                xy_variations = []
                #loop over again, to get standard deviation weighted
                w_av_z = np.sum(weighted_z)
                w_av_xy = np.sum(weighted_dist)
                for i in range(len(weights)):
                    z_variations.append(weights[i]*((z_positions[i] - w_av_z)**2))
                    xy_variations.append(weights[i]*((distances[i] - w_av_xy)**2))

                #weighted std
                w_std_dist = math.sqrt(np.sum(xy_variations))
                
                w_std_z = math.sqrt(np.sum(z_variations))

        


                XYAv[0] = np.mean(distances)
                XYWidth[0] = np.std(distances)
                XYAv_w[0] = np.sum(weighted_dist)
                XYWidth_w[0] = w_std_dist


            # get the mean and stddev for this event

                weighted_z = 0
                ZAverage_w[0] = (w_av_z)
                ZAv[0] = np.mean(z_positions)
                ZWidth_w[0] = (w_std_z)
                ZWidth[0] = np.std(z_positions)
                ZLength[0] = (abs(last_z - first_z))

                Eav[0] = (np.mean(energies))

                if (last_z - first_z) != 0:
                    EDensity[0] = (np.sum(energies))/(abs(last_z - first_z))


                isSignal[0] = 1


 
                Features.Fill()
        f.Write();
        f.Close();

def main(options,args) :
    sc = GetPart(options.ifile,options.ofile,options.label, options.mass, options.tag);
    #sc.fout.Close();

if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option('-b', action='store_true', dest='noX', default=False, help='no X11 windows')
    parser.add_option('-a','--ifile', dest='ifile', default = 'file.root',help='directory with data1', metavar='idir')
    parser.add_option('-o','--ofile', dest='ofile', default = 'ofile.root',help='directory to write plots', metavar='odir')
    parser.add_option('--label', dest='label', default = 'primakoff',help='production model', metavar='label')
    parser.add_option('--mass', dest='mass', default = '10',help='mass of ALP', metavar='mass')
    parser.add_option('--tag', dest='tag', default = '1',help='file tag', metavar='tag')

    (options, args) = parser.parse_args()


    ROOT.gStyle.SetPadTopMargin(0.10)
    ROOT.gStyle.SetPadLeftMargin(0.16)
    ROOT.gStyle.SetPadRightMargin(0.10)
    ROOT.gStyle.SetPalette(1)
    ROOT.gStyle.SetPaintTextFormat("1.1f")
    ROOT.gStyle.SetOptFit(0000)
    ROOT.gROOT.SetBatch()
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)

    main(options,args);
