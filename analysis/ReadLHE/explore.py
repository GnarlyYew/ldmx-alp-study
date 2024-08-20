from lhereader import readLHEF
from ROOT import TCanvas, TH1F, TH2F, TLorentzVector, TF1
import math
import matplotlib.pyplot as plt
import numpy as np
import argparse

path = "/Users/nathanjay/Desktop/SURF/ALP-8GeV/Primakoff/edited-100K-prima/"

def main(args):

    # Extract particles:
    data=[]
    data=readLHEF(str(args.fullfilename))
    photons=data.getParticlesByIDs([22])
    ALPs=data.getParticlesByIDs([666])
    electrons=data.getParticlesByIDs([11])

    pt = []
    pz = []
    pz_com = []
    e = []
    angle = []
    zangle = []
    nphoton = 0

    # TLorentzVector for the two photons
    gamma1_4mom = TLorentzVector()
    gamma2_4mom = TLorentzVector()

    for g in photons:
        # Outgoing photons (status ==1):
        if (g.status == 1):
            nphoton+=1
            # all photons
            pt.append(g.p4.Pt())
            pz.append(g.pz)
            zangle.append(g.p4.Angle((0., 0., 1.)))
            # to get details of each photon (assume two per event in even structure - this should be OK)
            if nphoton%2!=0:
                print(nphoton, "this is the first photon in event")
                gamma1_4mom = g.p4
                pz1 = g.pz
            if nphoton%2==0:
                print(nphoton,"this is the second photon in event")
                gamma2_4mom = g.p4
                pz2 = g.pz
                # angle between the two photons
                print("---------------------")
                angle.append(gamma1_4mom.Angle(gamma2_4mom.Vect()))
                pz_com.append(pz1 + pz2)

    
    alp_e = []
    alp_pz = []
    for a in ALPs:
        alp_e.append(a.p4.E())
        alp_pz.append(a.pz)
    
    fig, ax = plt.subplots(1,1)
    plt.title("Energy "+str(args.process)+" mALP = "+str(args.mass)+"MeV")
    n, bins, patches = ax.hist(pt,
                            bins=100,
                            
                            label="ALPs")
    
    nentries = len(alp_e)
    e_mean = np.mean(alp_e)

    plt.text(6,40000, 'nentries = '+str(nentries), fontsize = 8)
    plt.text(6,20000, 'mean = '+str(np.round(e_mean,2)), fontsize = 8)
    ax.set_yscale('log')
    ax.set_ylabel('events per bin')
    ax.set_xlabel('E of virtual ALP [MeV]')
    fig.savefig(f'./energies/ALP_E{args.mass}_{args.process}.pdf')

    fig, ax = plt.subplots(1,1)
    plt.title("Angle Between Photons "+str(args.process)+" mALP = "+str(args.mass)+"MeV/c")
    n, bins, patches = ax.hist(angle,
                            bins=100,
                            range=(0,2*math.pi),
                            label="photons")
    nentries = len(angle)
    mean = np.mean(angle)
    #rms = np.sqrt(np.mean(pt))
    #adding text inside the plot
    plt.text(5,30000, 'nentries = '+str(nentries), fontsize = 8)
    plt.text(6,10000, 'mean = '+str(np.round(mean,2)), fontsize = 8)
    ax.set_yscale('log')
    ax.set_ylabel('events per bin')
    ax.set_xlabel('angle between photons')
    fig.savefig(f'./angles/photon_angle{args.mass}.pdf')

    #2d histogram of pz and angle
    fig, ax = plt.subplots(1,1)
    h, xedges, yedges, image = ax.hist2d(pz, zangle, bins=100, range=[[0,8],[0,0.5*math.pi]], cmap=plt.cm.viridis)
    cbar = plt.colorbar(image, ax=ax)
    cbar.set_label("Counts")
    ax.set_xlabel("Longitudinal momentum of photons (GeV/c)")
    ax.set_ylabel("Angle of photons")
    plt.title(f"Angles and momentum of photons, {args.mass} Mev/c^2 in {args.process}")
    fig.savefig(f'./angles/2dhist_angles_to_pz_{args.process}_{args.mass}.pdf')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fullfilename", help="full filename with path", default="/Users/nathanjay/Desktop/SURF/ALP-8GeV/all/m10_prima.lhe")
    parser.add_argument("--process", help="Primakoff or Photon Fusion")
    parser.add_argument("--mass", help="ALP mass")
    args = parser.parse_args()
    (args) = parser.parse_args()
    main(args)
