import os

from LDMX.Framework import ldmxcfg
from LDMX.SimCore import generators
from LDMX.SimCore import simulator

p=ldmxcfg.Process("v14")

from LDMX.Hcal import hcal
from LDMX.Hcal import digi
from LDMX.Ecal import digi as ecaldigi
from LDMX.Ecal import ecalClusters
from LDMX.DetDescr.HcalGeometry import HcalGeometry
#from LDMX.DetDescr.EcalHexReadout import EcalHexReadoutGeometry
from LDMX.Ecal import EcalGeometry
from LDMX.Hcal import HcalGeometry
from LDMX.Hcal import hcal_hardcoded_conditions
from LDMX.Ecal import ecal_hardcoded_conditions
from LDMX.Ecal import vetos as ecal_veto

# edit these for each run:
filepath = "/Users/nathanjay/Desktop/SURF/ALP-8GeV/all/"
infilename = "m100_prima.lhe"
outfilename = "ALP_m100_prima_reco.root"
nevents = 50000

#from LDMX.Tools.HgcrocEmulator import HgcrocEmulator
# Instantiate the simulator
sim = simulator.simulator("signal")

# Set the detector to use and enable the scoring planes
sim.setDetector( 'ldmx-det-v14-8gev', True)

# Set the run number
#p.runNumber = {{ run }}

# Set a description of what type of run this is.
sim.description = "Signal generated using the v14 detector."

# Set the random seeds
sim.randomSeeds = [ 1,2 ]

# Smear the beamspot
sim.beamSpotSmear = [ 20., 80., 0 ]

# Enable the LHE generator
sim.generators.append(generators.lhe( "Signal Generator", (str(filepath)+str(infilename) )))

#ecalrec = ecaldigi.EcalRecProducer()
hcalDigis = digi.HcalDigiProducer()
ecalDigis = ecaldigi.EcalDigiProducer()
hcalDigis.hgcroc.noise = False
hcalClusters = hcal.HcalClusterProducer()
#hcalNewClusters = hcal.HcalNewClusterProducer()
#hcalOldDigis = hcal.HcalOldDigiProducer()
hcalrec = digi.HcalRecProducer()
ecalrec = ecaldigi.EcalRecProducer()
hcalWABVeto  = hcal.HcalWABVetoProcessor()
ecalVeto = ecal_veto.EcalVetoProcessor()
hcalVeto   = hcal.HcalVetoProcessor('hcalVeto')
geom = HcalGeometry.HcalGeometryProvider.getInstance()
p.sequence=[ sim, ecalDigis,  ecalrec, hcalDigis, hcalrec, hcalClusters]#, hcalVeto, hcalWABVeto, ecalVeto]#, hcalNewClusters]

p.outputFiles = [ str(outfilename)]

p.maxEvents = nevents #100,8418
p.logFrequency = 10
p.lheFilePath = (str(filepath)+str(infilename) )
#p.pause()
