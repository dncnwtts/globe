import matplotlib.pyplot as plt
import numpy as np
from healpy.newvisufunc import projview, newprojplot

import healpy as hp

import matplotlib.colors as col
import matplotlib as mpl

xsize = 8000*2
dpi = 600

def scale2(data, linthresh=2.5e-4):
    m = np.copy(data)
    inds = (data > linthresh)
    f = 256/345
    m = (data/linthresh+1)*f
    m[inds] = 2*f + (1-f)*np.log10(data[inds]/linthresh)/2
    return m/2

def scale(data, linthresh=2.5e-4):
    x = np.copy(data)
    vmin = -1e3
    vmax = 1e7
    x = x*1e6
    x =  np.log10(0.5*(x + np.sqrt(x**2 + 4)))
    x = (x + 3)/10.
    return x

def plot_objects(text=False):
    objects = {}
    sm = 2
    objects['LMC'] = [279.465165, -31.671889, sm]
    objects['Gum Nebula'] = [255.8, -0.91, 15]
    objects['Coma'] = [58.0791, 87.9577, sm]
    objects['Cyg A'] = [76.18988, 5.755388, sm]
    objects['3C 273'] = [289.95, 64.3599, sm]
    objects['3C 279'] = [305.104, 57.062, sm]
    objects['3C 345'] = [63.455, 40.9488, sm]
    objects['3C 84'] = [150.5758, -13.26, sm]
    objects['Carina Nebula'] = [287.60, -00.64452, sm]
    objects['Cas A'] = [111.73, -0.2129, sm]
    objects['Cen A'] = [309.516, 19.4, sm]
    objects['Cyg A'] = [76, 5.755, sm]
    objects['Fornax A'] = [240.162, -56.69, sm]
    objects['Crab Nebula'] = [184.55746, -5.78436, sm]
    objects['W3'] = [133.94761, 0.106505, sm]
    objects['Virgo A'] = [283.7778, 74.49115, sm]
    objects['Rosette'] = [206.49814, -2.02319, sm]
    objects['NGC 1499'] = [160.6032, -12.0513, sm]
    objects['IC 348'] = [160.4899, -17.8022, sm]
    objects[r'$\rho$ Oph'] = [353.68597, 17.68675, sm]
    objects['OV-236'] = [9.344, -19.607, sm]


    for obj in objects:
        lon0, lat0, rad = objects[obj]

        theta = np.linspace(0,2*np.pi, 20*rad)

        lon = rad*np.cos(theta) + lon0
        lat = rad*np.sin(theta) + lat0
        inds = (lon > 180)
        lon[inds] = 360 - lon[inds]
        lon[~inds] = -lon[~inds]
        if lon0 > 180:
            lon0 = 360 - lon0
        else:
            lon0 = -lon0

    
        #plt.plot(np.deg2rad(lon), np.deg2rad(lat), color='k', lw=5)

        x0 = np.cos(np.deg2rad(lon0))*np.sin(np.pi/2 - np.deg2rad(lat0))
        y0 = np.sin(np.deg2rad(lon0))*np.sin(np.pi/2 - np.deg2rad(lat0))
        z0 = np.cos(np.pi/2 - np.deg2rad(lat0))
        #print(x0, y0, z0)
        r = np.deg2rad(rad)

        p = np.array([x0,y0,z0])*np.cos(r)
        dr = np.sin(r)
        v1 = p
        v2 = np.array([1,0,0])
        v3 = np.array([0,1,0])

        u1 = np.copy(v1)
        u2 = v2 - u1*(u1.dot(v2)/u1.dot(u1))
        u3 = v3 - (u1.dot(v3)/u1.dot(u1))*u1 - (u2.dot(v3)/u2.dot(u2))*u2

        e1 = u1/u1.dot(u1)**0.5
        e2 = u2/u2.dot(u2)**0.5
        e3 = u3/u3.dot(u3)**0.5

        r = [p[i] + dr*(np.cos(theta)*e2[i] + np.sin(theta)*e3[i]) for i in range(3)]

        phi = np.arctan2(r[1], r[0])
        th = np.arccos(r[2])
        inds = (phi > np.pi)
        plt.plot(phi, np.pi/2-th, '.', color='k', lw=1)





        if text:
            plt.text(np.deg2rad(lon0), np.deg2rad(lat0),
                obj, bbox=dict(facecolor='red', alpha=0.5))

    return

cm1 = np.loadtxt('/mn/stornext/u3/duncanwa/c3pp/src/planck_cmap.dat')
cm2 = np.loadtxt('planck_cmap_logscale.dat')

cm3 = np.concatenate((cm1, cm2[167:,:]))

# Need to smooth the r jump a bit

r0 = cm3[255, 0]
r1 = cm3[268, 0]
m = (r1-r0)/(268-255)



inds = np.arange(len(cm1))

x = np.concatenate((inds, inds[167:]*256/167))
from scipy.interpolate import interp1d

f0 = interp1d(x, cm3[:,0])
f1 = interp1d(x, cm3[:,1])
f2 = interp1d(x, cm3[:,2])

inds = np.linspace(0,345, 1000)
cm3[255:269,0] = m*(inds[255:269] - inds[255]) + cm3[255,0]
cm4 = np.array([f0(inds), f1(inds), f2(inds)]).T


cmap = col.ListedColormap(cm4/255)
#cmap = col.ListedColormap(cm2/255)
mpl.cm.register_cmap(name='planck_log', cmap=cmap)

cmap = col.ListedColormap(np.loadtxt('/mn/stornext/u3/duncanwa/c3pp/src/planck_cmap.dat')
        / 255.0, "planck")
mpl.cm.register_cmap(name='planck', cmap=cmap)




import my_projections


# The globe will be 13 inches in diameter, so roughly that will be the width,
# and the height will be 6.5 inches. dpi of 300 is probably fair.

#p143 = hp.read_map('/mn/stornext/d16/cmbco/ola/npipe/freqmaps/npipe6v20_217_map_K.fits')
p143 = hp.read_map('/mn/stornext/d16/cmbco/bp/delivery/v8.00/BP_release_files/BP_070_IQU_n1024_v1.fits')
#p143 = hp.read_map('/mn/stornext/d16/cmbco/bp/delivery/v8.00/BP_release_files/BP_044_IQU_n0512_v1.fits')
p143 *= 1e-6


p143 = hp.remove_dipole(p143, gal_cut=30)


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

#projview(scale(p143), projection_type="cassini", min=0, max=1, 
#    cmap='planck_log', xsize=800, cbar=False)
#
projview(scale2(p143), projection_type="cassini", min=0, max=1, 
    cmap='planck_log', xsize=xsize, cbar=False)

plt.savefig(f'cassini_dpi{dpi}.pdf', dpi=dpi)
projview(p143, projection_type="cassini", min=-2.5e-4, max=2.5e-4, 
    cmap='planck', xsize=xsize, cbar=False)

plt.show()
#plot_objects(text=False)

plt.close('all')

#projview(p143, projection_type="mollweide", min=-2.5e-4, max=2.5e-4,
#    cmap='planck', xsize=xsize, cbar=False, graticule=False)
#
#plt.savefig('mollweide.png', bbox_inches='tight', dpi=dpi)
#plt.show()
