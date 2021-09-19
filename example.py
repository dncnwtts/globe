import matplotlib.pyplot as plt
import numpy as np
from healpy.newvisufunc import projview, newprojplot

import healpy as hp

import matplotlib.colors as col
import matplotlib as mpl

def scale2(data, linthresh=2e-4):
    m = np.copy(data)
    inds = (data > linthresh)
    f = 256/345
    m = (data/linthresh+1)*f
    m[inds] = 2*f + (1-f)*np.log10(data[inds]/linthresh)/2
    return m/2

def scale(data, linthresh=2e-4):
    x = np.copy(data)
    vmin = -1e3
    vmax = 1e7
    x = x*1e6
    x =  np.log10(0.5*(x + np.sqrt(x**2 + 4)))
    x = (x + 3)/10.
    return x


cm1 = np.loadtxt('/mn/stornext/u3/duncanwa/c3pp/src/planck_cmap.dat')
cm2 = np.loadtxt('planck_cmap_logscale.dat')

cm3 = np.concatenate((cm1, cm2[167:,:]))

inds = np.arange(len(cm1))

x = np.concatenate((inds, inds[167:]*256/167))
from scipy.interpolate import interp1d

f0 = interp1d(x, cm3[:,0])
f1 = interp1d(x, cm3[:,1])
f2 = interp1d(x, cm3[:,2])

inds = np.linspace(0,345, 1000)
cm4 = np.array([f0(inds), f1(inds), f2(inds)]).T


#cmap = col.ListedColormap(np.loadtxt('planck_cmap_logscale.dat')
#    / 255.0, "planck_log")
cmap = col.ListedColormap(cm4/255)
mpl.cm.register_cmap(name='planck_log', cmap=cmap)

cmap = col.ListedColormap(np.loadtxt('/mn/stornext/u3/duncanwa/c3pp/src/planck_cmap.dat')
        / 255.0, "planck")
mpl.cm.register_cmap(name='planck', cmap=cmap)




import my_projections


# The globe will be 13 inches in diameter, so roughly that will be the width,
# and the height will be 6.5 inches. dpi of 300 is probably fair.

p143 = hp.read_map('/mn/stornext/d16/cmbco/ola/npipe/freqmaps/npipe6v20_143_map_K.fits')


p143 = hp.remove_dipole(p143, gal_cut=30)

projview(scale2(p143), projection_type="mollweide", min=0, max=1, 
    cmap='planck_log', xsize=800, cbar=True)

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

projview(scale(p143), projection_type="cassini", min=0, max=1, 
    cmap='planck_log', xsize=800, cbar=False)

projview(scale2(p143), projection_type="cassini", min=0, max=1, 
    cmap='planck_log', xsize=800, cbar=False)

projview(p143, projection_type="cassini", min=-2.5e-4, max=2.5e-4, 
    cmap='planck', xsize=800, cbar=False)

plt.show()
