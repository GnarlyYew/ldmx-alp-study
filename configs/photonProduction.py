#!/usr/bin/python
import argparse
import importlib
import ROOT
from ROOT import TTree, TBranch, TH1F
ROOT.gSystem.Load("/Users/user/ldmx-sw/install/lib/libHcal_Event.so")	;
ROOT.gSystem.Load("/Users/user/ldmx-sw/install/lib/libEcal_Event.so")	;
ROOT.gSystem.Load("/Users/user/ldmx-sw/install/lib/libRecon_Event.so")	;
ROOT.gSystem.Load("/Users/user/ldmx-sw/install/lib/libSimCore_Event.so")	;
ROOT.gSystem.Load("/Users/user/ldmx-sw/install/lib/libFramework.so");
import os
import math
import sys
import csv
from array import array
from optparse import OptionParser
#import matplotlib.pyplot as plt
#sys.path.insert(0, '../')
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
path_to_files = "/Users/user/ldmx-sw/ALPSamples/primakoff_rootfiles/"
filenames = ["PN.root","ALP_m10.root","ALP_m50.root","ALP_m100.root","ALP_m150.root","ALP_m200.root"]
nametag= ["PN BKG","m = 10","m = 50","m = 100","m = 150", "m = 200"]
colors = ["black","cyan","magenta","green","blue","red"]
bars = ["k.","c.","m.","g.","b.","r."]
energy = []
mom_x = []
mom_y = []
mom_z = []
phi = []
polar = []
processname = "primakoff"
class GetPhotons:

    def __init__(self, fn1, ofn, tag):

        self.processname = "primakoff"
        #input files:
        self.fin1 = ROOT.TFile(str(path_to_files)+fn1);
        self.tin1 = self.fin1.Get("LDMX_Events")
        self.tag = int(tag);

        # output files:
        self.fn_out = ofn;
        self.fout = ROOT.TFile("hist_"+self.fn_out,"RECREATE");

        #list of branches:
        self.evHeader1 = ROOT.ldmx.EventHeader()
        self.simParticles1 = ROOT.std.map(int, 'ldmx::SimParticle')();
        self.hcalHits1 = ROOT.std.vector('ldmx::HcalHit')();
        self.hcalSimHits1 = ROOT.std.vector('ldmx::SimCalorimeterHit')();
        self.ecalHits1 = ROOT.std.vector('ldmx::EcalHit')();
        self.ecalSimHits1 = ROOT.std.vector('ldmx::SimCalorimeterHit')();
        self.targetSimHits1 = ROOT.std.vector('ldmx::SimCalorimeterHit')();
        self.hcalScoringPlaneHits1 = ROOT.std.vector('ldmx::SimTrackerHit')();
        self.ecalScoringPlaneHits1 = ROOT.std.vector('ldmx::SimTrackerHit')();
        self.targetScoringPlaneHits1 = ROOT.std.vector('ldmx::SimTrackerHit')();
        self.tin1.SetBranchAddress("EventHeader",  ROOT.AddressOf( self.evHeader1 ));
        self.tin1.SetBranchAddress("HcalSimHits_v12",  ROOT.AddressOf( self.hcalSimHits1 ));
        self.tin1.SetBranchAddress("HcalRecHits_v12",  ROOT.AddressOf( self.hcalHits1 ));
        self.tin1.SetBranchAddress("SimParticles_v12",  ROOT.AddressOf( self.simParticles1 ));
        self.tin1.SetBranchAddress("EcalSimHits_v12",  ROOT.AddressOf( self.ecalSimHits1 ));
        self.tin1.SetBranchAddress("TargetSimHits_v12",  ROOT.AddressOf( self.targetSimHits1 ));
        #self.tin1.SetBranchAddress("EcalRecHits_test",  ROOT.AddressOf( self.ecalHits1 ));
        self.tin1.SetBranchAddress("HcalScoringPlaneHits_v12",  ROOT.AddressOf(self.hcalScoringPlaneHits1));
        self.tin1.SetBranchAddress("EcalScoringPlaneHits_v12",  ROOT.AddressOf(self.ecalScoringPlaneHits1));
        self.tin1.SetBranchAddress("TargetScoringPlaneHits_v12",  ROOT.AddressOf(self.targetScoringPlaneHits1));

        # loop and save:
        self.loop();
        self.writeOutHistos();

    def writeOutHistos(self):
        self.fout.cd();

    def loop(self):
        nent = self.tin1.GetEntriesFast();
        print("Entries ",nent);
        f = open('data.csv', 'w')

        print("Energy [MeV],Mom x [MeV/c], Mom y [MeV/c], Mom z [MeV/c]")
        f.write("Event No.,Energy [MeV],Mom x [MeV/c], Mom y [MeV/c], Mom z [MeV/c]\n")
        nHasHits = 0;
        energy_i = []
        mom_x_i = []
        mom_y_i = []
        mom_z_i = []
        phi_i = []
        polar_i = []
        for i in range(nent):
            self.tin1.GetEntry(i);

            #print("==============================================")
            """
            for ih,hit in enumerate(self.targetSimHits1):
                for ik in range(0,hit.getNumberOfContribs()):
                    print(hit.getContrib(ik).pdgCode)
            """

            if(len(self.hcalHits1) > 0):
                nHasHits +=1
            for ih,hit in enumerate(self.targetScoringPlaneHits1):

                hitID = hit.getID() & 0xFFF ;

                if hitID ==4 and hit.getPdgID() == 22:# and hit.getTrackID()==1:
                    #print(hit.getEnergy(),",",hit.getMomentum()[0],",",hit.getMomentum()[1],",",hit.getMomentum()[2])
                    #print(hit.getPdgID(),hit.getEnergy())
                    energy_i.append(hit.getEnergy())
                    mom_x_i.append(hit.getMomentum()[0])
                    mom_y_i.append(hit.getMomentum()[1])
                    mom_z_i.append(hit.getMomentum()[2])
                    phi_i.append(math.atan2(hit.getMomentum()[1],hit.getMomentum()[0]))
                    polar_i.append(math.acos(hit.getMomentum()[2]/math.sqrt(hit.getMomentum()[0]*hit.getMomentum()[0] + hit.getMomentum()[1]*hit.getMomentum()[1] + hit.getMomentum()[2]*hit.getMomentum()[2])))
                    f.write(str(i)+","+str(hit.getEnergy())+","+str(hit.getMomentum()[0])+","+str(hit.getMomentum()[1])+","+str(hit.getMomentum()[2])+"\n")
        #print("hitID = ",hitID, hit.getPdgID())
        energy.append(energy_i)
        mom_x.append(mom_x_i)
        mom_y.append(mom_y_i)
        mom_z.append(mom_z_i)
        phi.append(phi_i)
        polar.append(polar_i)
        print(nHasHits)
def main(options,args) :
    for i,fil in enumerate(filenames):
        sc = GetPhotons(fil,options.ofile,options.tag);
    fig, ax = plt.subplots(1,1)
    print("LEngth",len(energy))
    for k, en in enumerate(energy):
        n, bins, patches = ax.hist(en,
                bins=100, range=(0,4000), alpha=0.5, histtype='step', label=nametag[k], color = colors[k])
    plt.yscale('log')
    ax.set_xlabel('Energy [MeV]')
    ax.set_ylabel('Entries per bin')
    plt.legend(loc="upper right")
    fig.savefig("energy_"+str(sc.processname)+".pdf")

    fig, ax = plt.subplots(1,1)
    for k, en in enumerate(mom_x):
        n, bins, patches = ax.hist(en,
                bins=100, range=(0,500), alpha=0.5, histtype='step', label=nametag[k], color = colors[k])
    plt.yscale('log')
    ax.set_xlabel('Mom x [MeV/c]')
    ax.set_ylabel('Entries per bin')
    plt.legend(loc="upper right")
    fig.savefig("momx_"+str(sc.processname)+".pdf")

    fig, ax = plt.subplots(1,1)
    for k, en in enumerate(mom_y):
        n, bins, patches = ax.hist(en,
                bins=100, range=(0,500), alpha=0.5, histtype='step', label=nametag[k], color = colors[k])
    plt.yscale('log')
    ax.set_xlabel('Mom y [MeV/c]')
    ax.set_ylabel('Entries per bin')
    plt.legend(loc="upper right")
    fig.savefig("momy_"+str(sc.processname)+".pdf")

    fig, ax = plt.subplots(1,1)
    for k, en in enumerate(mom_z):
        n, bins, patches = ax.hist(en,
                bins=100, range=(0,4000), alpha=0.5, histtype='step', label=nametag[k], color = colors[k])
    plt.yscale('log')
    ax.set_xlabel('Mom z [MeV/c]')
    ax.set_ylabel('Entries per bin')
    plt.legend(loc="upper right")
    fig.savefig("momz_"+str(sc.processname)+".pdf")

    fig, ax = plt.subplots(1,1)
    for k, en in enumerate(phi):
        n, bins, patches = ax.hist(en,
                bins=100, range=(0,3.1415), alpha=0.5, histtype='step', label=nametag[k], color = colors[k])
    plt.yscale('log')
    ax.set_xlabel('Phi [rad]')
    ax.set_ylabel('Entries per bin')
    plt.legend(loc="upper right")
    fig.savefig("phi_"+str(sc.processname)+".pdf")

    fig, ax = plt.subplots(1,1)
    for k, en in enumerate(polar):
        n, bins, patches = ax.hist(en,
                bins=100, range=(0,3.1415), alpha=0.5, histtype='step', label=nametag[k], color = colors[k])
    plt.yscale('log')
    ax.set_xlabel('Polar [rad]')
    ax.set_ylabel('Entries per bin')
    plt.legend(loc="upper right")
    fig.savefig("polar_"+str(sc.processname)+".pdf")


    for k, en in enumerate(energy): #one per filenmame
        fig, ax = plt.subplots(1,1)
        plt.hist2d(polar[k],en,cmin=1,bins=[50,50])
        print("name ",nametag[k])
        plt.legend()
        cmap = plt.cm.nipy_spectral
        plt.colorbar()
        ax.set_xlabel('Polar Angle [rad]')
        ax.set_ylabel('Photon Energy [MeV]')
        plt.text(2.5,1500, str(nametag[k]), fontsize=12)
        fig.savefig(str(processname)+"_theta_v_e"+str(k)+".pdf")

    sc.fout.Close();

if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option('-b', action='store_true', dest='noX', default=False, help='no X11 windows')
    parser.add_option('-a','--ifile1', dest='ifile1', default = 'file1.root',help='directory with data1', metavar='idir1')
    parser.add_option('-o','--ofile', dest='ofile', default = 'ofile.root',help='directory to write plots', metavar='odir')
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
    # Get the Event library
    ROOT.gSystem.Load("/Users/user/ldmx-sw/install/lib/libHcal_Event.so")	;
    ROOT.gSystem.Load("/Users/user/ldmx-sw/install/lib/libFramework.so");
    ROOT.gSystem.Load("/Users/user/ldmx-sw/install/lib/libEcal_Event.so")	;
    ROOT.gSystem.Load("/Users/user/ldmx-sw/install/lib/libRecon_Event.so")	;
    ROOT.gSystem.Load("/Users/user/ldmx-sw/install/lib/libSimCore_Event.so")	;

    main(options,args);
