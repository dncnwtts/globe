import matplotlib.pyplot as plt
import numpy as np

import healpy as hp

# The globe will be 13 inches in diameter, so roughly that will be the width,
# and the height will be 6.5 inches. dpi of 300 is probably fair.
fig = plt.figure(figsize=(13, 6.5))

p143 = hp.read_map('/mn/stornext/d16/cmbco/ola/npipe/freqmaps/npipe6v20_143_map_K.fits')
p143 = hp.remove_dipole(p143, gal_cut=30)
hp.cartview(p143, min=-3.5e-4, max=3.5e-4, cmap='RdBu_r', xsize=8000,
    cbar=False, title='', coord='G', fig=fig)

hp.graticule(coord='E')

lon0 = 255.8
lat0 = -0.91
theta = np.linspace(0,2*np.pi)
lon = 15*np.cos(theta) + lon0
lat = 15*np.sin(theta) + lat0
hp.projplot(lon, lat, lonlat=True)
hp.projtext(lon0, lat0, 'Gum Nebula', lonlat=True, horizontalalignment='center',
    direct=False)


plt.show()
