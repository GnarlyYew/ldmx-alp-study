from LDMX.Framework import ldmxcfg
p = ldmxcfg.Process('gun')

from LDMX.SimCore import simulator as sim
mySim = sim.simulator( "mySim" )
mySim.description = 'Hcal'
mySim.setDetector( 'ldmx-det-v14',True )
from LDMX.SimCore import generators as gen

myGun = gen.gun('myGun')
myGun.particle = "gamma" #'neutron','mu-','pi-','e-','gamma'
myGun.position = [ 0., 0., 0. ]
myGun.direction = [ 0., 0., 1.]
myGun.energy = 1.0 #GeV

#from LDMX.Detectors.makePath import *
#from LDMX.SimCore import simcfg
#mySims.scoringPlanes = makeScoringPlanesPath('ldmx-det-v14')
mySim.generators = [ myGun ]
# mySim.verbosity = 1

p.sequence = [ mySim ]

##################################################################
# Below should be the same for all sim scenarios

import os
import sys

p.run = 1
p.maxEvents = 100
#p.logFrequency = 100
p.histogramFile = 'hist.root'
p.outputFiles = ['neutrons_1GeV.root']

import LDMX.Ecal.EcalGeometry
import LDMX.Ecal.ecal_hardcoded_conditions
import LDMX.Hcal.HcalGeometry
import LDMX.Hcal.hcal_hardcoded_conditions
import LDMX.Ecal.digi as ecal_digi
import LDMX.Ecal.vetos as ecal_vetos
import LDMX.Hcal.digi as hcal_digi

from LDMX.DQM import dqm
hcalDigis = hcal_digi.HcalDigiProducer()
hcalDigis.hgcroc.noise = False
hcalrec = hcal_digi.HcalRecProducer()
p.sequence.extend([
        hcalDigis,
        hcalrec,
        ])
