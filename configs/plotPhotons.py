

# import all modules
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# Read in the DataFrame
df = pd.read_csv('secPhot.csv')

# creating a histogram
entries, edges, _ = plt.hist(df['Energy [MeV]'],bins=80,histtype='step',color='blue')
bin_centers = 0.5 *(edges[:-1] + edges[1:])
plt.errorbar(bin_centers,entries,yerr=np.sqrt(entries),fmt='blue',marker = '.',linestyle='')
ax.set_yscale('log')
plt.xlabel('Photon Energy [MeV]')
plt.ylabel('Entries per bin')
plt.show()
