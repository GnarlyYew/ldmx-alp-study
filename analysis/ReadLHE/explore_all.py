from lhereader import readLHEF
from ROOT import TCanvas, TH1F, TH2F, TLorentzVector, TF1
import math
import matplotlib.pyplot as plt
import numpy as np
import argparse

path = "/Users/nathanjay/Desktop/SURF/ALP-8GeV/all/"

def main(args):
    pts = []
    pzs = []
    angles = []
    mean_angles = []
    energies = []
    mean_energies = []
    e_pts = []
    e_pzs = []
    e_angles = []
    e_energies = []
    alp_pzs = []

    for i, data in enumerate(args.fullfilename):
    
    # Extract particles:
        data=[]
        data=readLHEF(str(path)+str(args.fullfilename[i]))
        photons=data.getParticlesByIDs([22])
        ALPs=data.getParticlesByIDs([666])
        electrons=data.getParticlesByIDs([11])

        pt = []
        pz = []
        angle = []
        e_pt = []
        e_pz = []
        e_angle = []
        nphoton = 0

        # TLorentzVector for the two photons
        gamma1_4mom = TLorentzVector()
        gamma2_4mom = TLorentzVector()
        e_4mom = TLorentzVector()
        lorentz_vector = TLorentzVector()
        lorentz_vector.SetPxPyPzE(0.0, 0.0, 1.0, 0.0)

        for g in photons:
            # Outgoing photons (status ==1):
            if (g.status == 1):
                nphoton+=1
                # all photons
                pt.append(g.p4.Pt())
                pz.append(g.pz)
                # to get details of each photon (assume two per event in even structure - this should be OK)
                if nphoton%2!=0:
                    print(nphoton, "this is the first photon in event")
                    gamma1_4mom = g.p4
                if nphoton%2==0:
                    print(nphoton,"this is the second photon in event")
                    gamma2_4mom = g.p4
                    # angle between the two photons
                    print("---------------------")
                    angle.append(gamma1_4mom.Angle(gamma2_4mom.Vect()))

        for e in electrons:
            if (e.status == 1):
                #outgoing
                e_pt.append(e.p4.Pt())
                e_pz.append(e.pz)
                e_4mom = e.p4
                e_angle.append(e_4mom.Angle(lorentz_vector.Vect()))

        
        alp_e = []
        alp_pz = []
        for a in ALPs:
            alp_e.append(a.p4.E())
            alp_pz.append(a.pz)
        
        pts.append(pt)
        pzs.append(pz)
        angles.append(angle)
        energies.append(alp_e)

        e_pts.append(e_pt)
        e_pzs.append(e_pz)
        e_angles.append(e_angle)

        alp_pzs.append(alp_pz)

        e_mean = np.mean(alp_e)
        mean_energies.append(e_mean)

        angle_mean = np.mean(angle)
        mean_angles.append(angle_mean)

    #Photon momentum
    fig, ax = plt.subplots(1,1)
    for i, data in enumerate(pts):

        plt.title("Photon Transverse Momentum")
        n, bins, patches = ax.hist(pts[i],
       
                                   histtype = 'step',
                                   label=str(args.fullfilename[i]))
    plt.legend()
    ax.set_yscale('log')
    ax.set_ylabel('events per bin')
    ax.set_xlabel('pt of all photons [GeV/c]')
    fig.savefig(f'./momenta/pt_{args.process}.pdf')

    #electron momentum
    fig, ax = plt.subplots(1,1)
    for i, data in enumerate(e_pts):

        plt.title("Electron Transverse Momentum")
        n, bins, patches = ax.hist(e_pts[i],
       
                                   histtype = 'step',
                                   label=str(args.fullfilename[i]))
    plt.legend()
    ax.set_yscale('log')
    ax.set_ylabel('events per bin')
    ax.set_xlabel('pt of outgoing electrons [GeV/c]')
    fig.savefig(f'./momenta/e_pt_{args.process}.pdf')

    #photon momentum
    fig, ax = plt.subplots(1,1)
    for i, data in enumerate(pzs):

        plt.title("Photon Longitudinal Momentum")
        n, bins, patches = ax.hist(pzs[i],
       
                                   histtype = 'step',
                                   label=str(args.fullfilename[i]))
    plt.legend()
    ax.set_yscale('log')
    ax.set_ylabel('events per bin')
    ax.set_xlabel('pz of all photons [GeV/c]')
    fig.savefig(f'./momenta/pz_{args.process}.pdf')

    #electron momentum
    fig, ax = plt.subplots(1,1)
    for i, data in enumerate(e_pzs):

        plt.title("Photon Longitudinal Momentum")
        n, bins, patches = ax.hist(e_pzs[i],
       
                                   histtype = 'step',
                                   label=str(args.fullfilename[i]))
    plt.legend()
    ax.set_yscale('log')
    ax.set_ylabel('events per bin')
    ax.set_xlabel('pz of electrons [GeV/c]')
    fig.savefig(f'./momenta/e_pz_{args.process}.pdf')

    #photon angles 
    styles = ["solid", "solid", "solid", "solid", "dashed", "dashed", "dashed", "dashed"]
    colors = ["green", "blue", "red", "purple", "green", "blue", "red", "purple"]
    fig, ax = plt.subplots(1,1)
    for i, data in enumerate(angles):

        plt.title("Angle between photons")
        n, bins, patches = ax.hist(angles[i],
                                   bins=60,
                                   range=(0,math.pi),
                                   histtype = 'step',
                                   linestyle = styles[i],
                                   color = colors[i],
                                   label=str(args.fullfilename[i]))
    plt.legend(prop={'size': 8})
    ax.set_yscale('log')
    ax.set_ylabel('events per bin')
    ax.set_xlabel('angle between photons (radians)')
    fig.savefig(f'./angles/angle_all{args.process}.pdf')

    #electron angle
    fig, ax = plt.subplots(1,1)
    for i, data in enumerate(e_angles):

        plt.title("Angle between photons")
        n, bins, patches = ax.hist(e_angles[i],
                                   bins=50,
                                   #range=(-1,2*math.pi),
                                   histtype = 'step',
                                   label=str(args.fullfilename[i]))
    plt.legend()
    ax.set_yscale('log')
    ax.set_ylabel('events per bin')
    ax.set_xlabel('angle of outgoing electron (radians)')
    fig.savefig(f'./angles/e_angle_all{args.process}.pdf')

    masses = [10, 200, 500]
    fig, ax = plt.subplots(1, 1)
    ax.scatter(masses, mean_energies)
    
    ax.set_ylabel("Mean energy from sample (GeV)")
    ax.set_xlabel("Theoretical mass (MeV / c^2)")
    fig.savefig(f'./energies/mean_energy_{args.process}.pdf')

    #angles(mass)
    fig, ax = plt.subplots(1, 1)
    ax.scatter(masses, mean_angles)
    
    ax.set_ylabel("angle between photons")
    ax.set_xlabel("Theoretical mass (MeV / c^2)")
    fig.savefig(f'./angles/mean_angle_{args.process}.pdf')


    #for repeating masses for 2d hist
    #hist_masses = []
    #for i in range(len(angles[1])):
     #   for mass in masses:
     #       hist_masses.append(mass)
    
    
    one_d_angles = [item for sublist in angles for item in sublist]
    one_d_pz = [item for sublist in alp_pzs for item in sublist]
    fig, ax = plt.subplots(1,1)
    colors = ['Blues', "Greens", 'Reds']
    #h, xedges, yedges, image = ax.hist2d(one_d_pz, one_d_angles, bins=100, cmap=plt.cm.viridis)
    #for i in range(len(angles)):
        
    #     , alpha=0.5, label=f'Mass {masses[i]} MeV/c^2')

    cbar = plt.colorbar(image, ax=ax)
    cbar.set_label("Counts")
    ax.set_xlabel("Longitudinal momentum of ALP (GeV/c)")
    ax.set_ylabel("Angle between photons")
    plt.title(f"Angles and momentum in {args.process}")
    ax.legend()
    fig.savefig(f'./angles/2dhist_angles_pz_{args.process}.pdf')
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    files = []
    masses = [10, 150, 300, 500]
    processes = ["pf", "prima"]
    for process in processes:
        for mass in masses:
            files.append(f'm{mass}_{process}.lhe')
    
    parser.add_argument("--fullfilename", help="full filename with path", default=files)
    parser.add_argument("--process", help="Primakoff or Photon Fusion")
    parser.add_argument("--mass", help="ALP mass")
    args = parser.parse_args()
    (args) = parser.parse_args()
    main(args)
