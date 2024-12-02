import xml.etree.ElementTree as ET
from ROOT import TLorentzVector

class Particle:
    def __init__(self, eventid, pdgid, spin, px=0, py=0, pz=0, energy=0, mass=0, vtim=0, status=0):
        self.eventid=eventid
        self.pdgid=pdgid
        self.px=px
        self.py=py
        self.pz=pz
        self.energy=energy
        self.mass=mass
        self.spin=spin
        self.status=status
        self.vtim=vtim

    @property
    def p4(self):
        return TLorentzVector(self.px,self.py,self.pz,self.energy)

    @p4.setter
    def p4(self,value):
        self.px=value.Px()
        self.py=value.Py()
        self.pz=value.Pz()
        self.energy=value.E()
        self.mass=value.M()

    @property
    def p(self):
        return self.p4.P()

    @property
    def eta(self):
        return self.p4.Eta()

    @property
    def pt(self):
        return self.p4.Pt()



class Event:
    def __init__(self,num_particles):
        self.num_particles=num_particles
        self.particles=[]
        self.vertex_x = None
        self.vertex_y = None
        self.vertex_z = None
        self.vertex_distance = None

    def __addParticle__(self,particle):
        self.particles.append(particle)
    
    def get_vertex(self):
        return (self.vertex_x, self.vertex_y, self.vertex_z, self.vertex_distance)


    def getParticlesByIDs(self,idlist):
        partlist=[]
        for pdgid in idlist:
            for p in self.particles:
                if p.pdgid==pdgid:
                    partlist.append(p)
        return partlist

class LHEFData:
    def __init__(self,version):
        self.version=version
        self.events=[]

    def __addEvent__(self,event):
        self.events.append(event)

    def getParticlesByIDs(self,idlist):
        partlist=[]
        for event in self.events:
            partlist.extend(event.getParticlesByIDs(idlist))
        return partlist

    def getVertices(self):
        vertices = []
        for event in self.events:
            vertices.extend(event.get_vertex())
        return vertices

def readLHEF(name):
    tree = ET.parse(name)
    root=tree.getroot()
    lhefdata=LHEFData(float(root.attrib['version']))
    for n, child in enumerate(root):
        if(child.tag == 'event'):
            lines=child.text.strip().split('\n')
            event_header=lines[0].strip()
            num_part=int(event_header.split()[0].strip())
            e=Event(num_part)
            for i in range(1,num_part+1):
                part_data = lines[i].strip().split()
                p = Particle(int(n),int(part_data[0]), float(part_data[12]), float(part_data[6]), float(part_data[7]), float(part_data[8]), float(part_data[9]), float(part_data[10]), float(part_data[11]), int(part_data[1]))
                e.__addParticle__(p)
            
             # Check if there's a vertex line after particle data
            if lines[-1].startswith("#vertex"):
                vertex_data = lines[-1].split()
                vertex_x = float(vertex_data[1])
                vertex_y = float(vertex_data[2])
                vertex_z = float(vertex_data[3])
                vertex_distance = float(vertex_data[4].strip("[]"))
                e.set_vertex(vertex_x, vertex_y, vertex_z, vertex_distance)

            lhefdata.__addEvent__(e)

    return lhefdata
