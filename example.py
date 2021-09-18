import matplotlib.pyplot as plt
import numpy as np
from healpy.newvisufunc import projview, newprojplot

import healpy as hp

import matplotlib.colors as col
import matplotlib as mpl

def scale(data, linthresh=2e-4, logmax=1e3):
    m = np.copy(data)
    inds = (data > linthresh)
    m = (data/linthresh+1)/2
    m[inds] = 1 + 0.5*np.log10(data[inds]/linthresh)
    return m



cmap = col.ListedColormap(np.loadtxt('planck_cmap_logscale.dat')
    / 255.0, "planck_log")
mpl.cm.register_cmap(name='planck_log', cmap=cmap)



import my_projections


# The globe will be 13 inches in diameter, so roughly that will be the width,
# and the height will be 6.5 inches. dpi of 300 is probably fair.
fig = plt.figure(figsize=(13, 6.5))

p143 = hp.read_map('/mn/stornext/d16/cmbco/ola/npipe/freqmaps/npipe6v20_143_map_K.fits')
p143 = hp.remove_dipole(p143, gal_cut=30)


hp.cartview(scale(p143), min=0, max=2, cmap='planck_log', xsize=8000,
    cbar=False, title='', coord='G', fig=fig)

hp.graticule(coord='E')

#lon0 = 255.8
#lat0 = -0.91
#theta = np.linspace(0,2*np.pi)
#lon = 15*np.cos(theta) + lon0
#lat = 15*np.sin(theta) + lat0
#hp.projplot(lon, lat, lonlat=True)
#hp.projtext(lon0, lat0, 'Gum Nebula', lonlat=True, horizontalalignment='center',
#    direct=True)

#hp.projtext(58.0791, 87.9577, 'Coma Cluster', lonlat=True,
#    horizontalalignment='left', direct=False)


#plt.show()


projview(scale(p143), coord=["G"], projection_type="sinusoidal", min=0, max=2, 
    cmap='planck_log', xsize=8000)

plt.savefig('test_sinusoidal.png', bbox_inches='tight', dpi=300)

projview(scale(p143), coord=["G"], projection_type="polyconic", min=0, max=2, 
    cmap='planck_log', xsize=8000)

plt.savefig('test_polyconic.png', bbox_inches='tight', dpi=300)

projview(scale(p143), coord=["G"], projection_type="cassini", min=0, max=2, 
    cmap='planck_log', xsize=8000)

plt.savefig('test_cassini.png', bbox_inches='tight', dpi=300)
plt.show()
