from lhereader import readLHEF
from ROOT import TCanvas, TH1F, TH2F, TLorentzVector, TF1
import math
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd

path = "/Users/nathanjay/Desktop/SURF/ALP-8GeV/displaced/"

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

    #TLorentzVector for the two photons
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
    alp_px = []
    alp_py = []
    alp_pz = []
    alp_p = []
    alp_vtims = []
    alp_mass = []
    alp_gammas = []
    alp_dz = []
    alp_dis = []
    velocities = []
    vtims = []
    for a in ALPs:
        alp_e.append(a.p4.E())
        alp_pz.append(a.pz)
        alp_vtims.append(a.vtim)
        alp_mass.append(a.mass)
        alp_gammas.append(a.p4.Gamma())
        alp_p.append(a.p4.P())
        velocity = a.p4.Vect() / (a.p4.Gamma() * a.mass) #in units of c?
        velocities.append(velocity[2])
        displacement = velocity * a.vtim   #in mm
        vtims.append(a.vtim)
        #calculate magnitude
        alp_dis.append(np.sqrt(displacement[0]**2 + displacement[1]**2 + displacement[2]**2))
        alp_dz.append(displacement[2])

        #check that displacement in z makes sense
    processes = ['prima', 'pf']
    #transferring signal yield here   
    sig_yield = {'Process': processes}
    sig_yield[10] = [3071, 2097]
    sig_yield[100] = [4822, 888]
    sig_yield[150] = [4215, 616]
    sig_yield[200] = [3505, 448]
    sig_yield[300] = [2316, 249]
    sig_yield[400] = [1470, 140]
    sig_yield[500] = [927, 80]

    sig_yield = pd.DataFrame(sig_yield)

    #doing displacement scaling
    dz_median = np.median(alp_dz)
    coup_d = []
    coup_d_med = []
    couplings = np.logspace(-5, -3, 50)
    alp_dz = np.array(alp_dz)
    #dis scaling
    dis_factor = (1e-3 / couplings)**2
    xsec_factor = (couplings / 1e-3)**2


    for factor in dis_factor:
        coup_d.append(alp_dz * factor)
        coup_d_med.append(dz_median * factor)
    

    produced = sig_yield[int(args.mass)][0] * xsec_factor
    fig, ax1 = plt.subplots()

# Plot the first graph with ax1
    ax1.plot(couplings, coup_d_med, color='blue', label='Median displacement')
    #ax1.axhline(y=1000)
    ax1.set_xlabel('Coupling')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel('Median d', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create the second y-axis with twinx and plot the second graph
    ax2 = ax1.twinx()   
    ax2.plot(couplings, produced, color='red', label='Reconstructed ALPs')
    ax2.axhline(y=1)
    ax2.set_ylabel('Reconstructed ALPs', color='red')
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor='red')

    # Add a title and show the plot
    plt.title(f'Displacement and # of ALPs for {args.mass} Prima')
    fig.tight_layout()  # Adjust layout to avoid overlap
    plt.show()
    fig.savefig(f'./decay/comb_{args.mass}')


    print(coup_d_med[-1])
    coup_d = np.array(coup_d)
    #coup_d is a 2d array of displacements at different couplings
    hist_data = []
    for i, coupling in enumerate(couplings):
        hist, bin_edges = np.histogram(coup_d[i, :], bins=100, density=True)
        hist_data.append(hist)
    fig, ax = plt.subplots(1, 1)
    hist_data = np.array(hist_data)
    X, Y = np.meshgrid(bin_edges[:-1], couplings)
    non_zero_hist_data = np.where(hist_data == 0, np.nan, hist_data)  # Replace zero entries with NaN

    # Plot the heatmap without zero entries
    c = ax.pcolormesh(X, Y, non_zero_hist_data, shading='auto', cmap='viridis')
    fig.colorbar(c, ax=ax, label='Frequency')
    ax.set_xlabel('Displacement')
    ax.set_ylabel('Coupling')
    ax.set_title('Displacement Histograms (without zeros)')
    fig.savefig(f'./decay/coup_{args.mass}')

    fig, ax = plt.subplots(1, 1)
    ax.plot(couplings, coup_d_med)
    ax.set_xlabel('Coupling')
    ax.set_ylabel('Median displacement')
    ax.set_xscale('log')
    #ax.set_yscale('log')
    fig.savefig(f'./decay/coup_med_{args.mass}')

    #plotting cross sections and median displacement

    
    
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

    fig, ax = plt.subplots(1, 1)
    ax.hist(alp_vtims, bins=100, range=[0, 5000])
    fig.savefig(f'./decay/vtims_{args.process}_{args.mass}')

    fig, ax = plt.subplots(1, 1)
    ax.hist(alp_dz, range=[0, 5000], bins=100)
    ax.set_yscale('log')
    ax.set_xlabel("displacement in z [mm]")
    plt.title(f'Displacement for {args.mass}, {args.process}')
    fig.savefig(f'./decay/dis_{args.process}_{args.mass}')

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fullfilename", help="full filename with path", default="/Users/nathanjay/Desktop/SURF/ALP-8GeV/displaced/DP_m10_prima.lhe")
    parser.add_argument("--process", help="Primakoff or Photon Fusion")
    parser.add_argument("--mass", help="ALP mass")
    args = parser.parse_args()
    (args) = parser.parse_args()
    main(args)
