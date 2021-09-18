import matplotlib
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.path import Path
from matplotlib.ticker import NullLocator, Formatter, FixedLocator
from matplotlib.transforms import Affine2D, BboxTransformTo, Transform
from matplotlib.projections import register_projection
import matplotlib.spines as mspines
import matplotlib.axis as maxis
import numpy as np

from custom_projections import GeoAxes

class HealpixAxes(GeoAxes):
    name = 'healpix'

    class HealpixTransform(Transform):
        input_dims = output_dims = 2

        def __init__(self, resolution):
            Transform.__init__(self)
            self._resolution = resolution

        def transform_non_affine(self, ll):
            longitude, latitude = ll.T

            # Pre-compute some values
            H = 4
            K = 3
            theta_x = np.arcsin((K-1)/K)

            x = longitude
            y = np.pi/2*K/H*np.sin(latitude)

            cap = (latitude > theta_x)
            base = (latitude < -theta_x)
            poles = (cap | base)
            sigma = np.sqrt(K*(1-np.abs(np.sin(latitude))))
            if (K % 2 == 0):
              omega = 0
            else:
              omega = 1
            y[cap] = np.pi/H*(2-sigma[cap])
            y[base] = -np.pi/H*(2-sigma[base])

            phi_c = -np.pi + (2*np.floor((longitude+np.pi)*H/(2*np.pi)+(1-omega)/2) + omega)*np.pi/H

            poles = base
            x[poles] = phi_c[poles] + (longitude[poles]-phi_c[poles])*sigma[poles]
            poles = cap
            x[poles] = phi_c[poles] + (longitude[poles]-phi_c[poles])*sigma[poles]

            return np.column_stack([x, y])

        def transform_path_non_affine(self, path):
            # vertices = path.vertices
            ipath = path.interpolated(self._resolution)
            return Path(self.transform(ipath.vertices), ipath.codes)

        def inverted(self):
            return HealpixAxes.InvertedHealpixTransform(self._resolution)

    class InvertedHealpixTransform(Transform):
        input_dims = output_dims = 2

        def __init__(self, resolution):
            Transform.__init__(self)
            self._resolution = resolution

        def transform_non_affine(self, xy):
            x, y = xy.T
            H = 4
            K = 3
            if (K % 2 == 0):
              omega = 0
            else:
              omega = 1

            y_x = np.pi/2*(K-1)/H
            cap = (y > y_x)
            base = (y < -y_x)
            poles = (cap | base)

            longitude = x
            latitude = np.zeros_like(y)
            latitude[~poles] = np.arcsin(y[~poles]*H/(np.pi*K/2))

            sigma = (K+1)/2 - np.abs(y*H)/np.pi
            x_c = -np.pi + (2*np.floor((x+np.pi)*H/(2*np.pi) + (1-omega)/2) + omega)*np.pi/H

            longitude[poles] = (x_c + (x - x_c)/sigma)[poles]

            latitude[cap] = np.arcsin(1-sigma[cap]/K)
            latitude[base] = -np.arcsin(1-sigma[base]/K)

            return np.column_stack([longitude, latitude])

        def inverted(self):
            return HealpixAxes.HealpixTransform(self._resolution)

    def __init__(self, *args, **kwargs):
        self._longitude_cap = np.pi / 2.0
        super().__init__(*args, **kwargs)
        self.set_aspect(0.5, adjustable='box', anchor='C')
        self.cla()

    def _get_core_transform(self, resolution):
        return self.HealpixTransform(resolution)

    def _gen_axes_spines(self):
        x = np.array([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,
                      2, 1.75, 1.5, 1.25, 1, 0.75, 0.5, 0.25, 0, 0])/2
        y = np.array([0.75, 1, 0.75, 1, 0.75, 1, 0.75, 1, 0.75,
                      0.25, 0, 0.25, 0, 0.25, 0, 0.25, 0, 0.25, 0.75])
        polygon = Polygon(np.array([x,y]).T)

        path = polygon.get_path()
        spine = mspines.Spine(axes=self, spine_type='circle', path=path)
        spine.set_transform(self.transAxes)
        return {'geo': spine}

    def _gen_axes_patch(self):
        """
        Override this method to define the shape that is used for the
        background of the plot.  It should be a subclass of Patch.

        In this case, it is a Circle (that may be warped by the axes
        transform into an ellipse).  Any data and gridlines will be
        clipped to this shape.
        """
        x = np.array([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,
                      2, 1.75, 1.5, 1.25, 1, 0.75, 0.5, 0.25, 0, 0])/2
        y = np.array([0.75, 1, 0.75, 1, 0.75, 1, 0.75, 1, 0.75,
                      0.25, 0, 0.25, 0, 0.25, 0, 0.25, 0, 0.25, 0.75])
        return Polygon(np.array([x,y]).T)

register_projection(HealpixAxes)

class EquirectangularAxes(GeoAxes):
    name = 'equirectangular'

    class EquirectangularTransform(Transform):
        input_dims = output_dims = 2

        def __init__(self, resolution):
            Transform.__init__(self)
            self._resolution = resolution

        def transform_non_affine(self, ll):
            l, phi = ll.T
            R = 1
            phi_0 = 0
            phi_1 = 0
            l_0 = 0

            x = R*(l - l_0)*np.cos(phi_1)
            y = R*(phi - phi_0)

            return np.column_stack([x, y])

        def transform_path_non_affine(self, path):
            # vertices = path.vertices
            ipath = path.interpolated(self._resolution)
            return Path(self.transform(ipath.vertices), ipath.codes)

        def inverted(self):
            return EquirectangularAxes.InvertedEquirectangularTransform(self._resolution)

    class InvertedEquirectangularTransform(Transform):
        input_dims = output_dims = 2

        def __init__(self, resolution):
            Transform.__init__(self)
            self._resolution = resolution

        def transform_non_affine(self, xy):
            x, y = xy.T
            R = 1
            phi_0 = 0
            phi_1 = 0
            l_0 = 0

            l = x/(R*np.cos(phi_1)) + l_0
            phi = y/R + phi_0

            return np.column_stack([l, phi])

        def inverted(self):
            return EquirectangularAxes.EquirectangularTransform(self._resolution)

    def __init__(self, *args, **kwargs):
        self._longitude_cap = np.pi / 2.0
        super().__init__(*args, **kwargs)
        self.set_aspect(0.5, adjustable='box', anchor='C')
        self.cla()

    def _get_core_transform(self, resolution):
        return self.EquirectangularTransform(resolution)

    def _gen_axes_patch(self):
        """
        Override this method to define the shape that is used for the
        background of the plot.  It should be a subclass of Patch.

        In this case, it is a Circle (that may be warped by the axes
        transform into an ellipse).  Any data and gridlines will be
        clipped to this shape.
        """
        return Rectangle((0, 0), 1, 1)

    def _gen_axes_spines(self):
        x = np.array([0,1,1,0,0])
        y = np.array([0,0,1,1,0])
        polygon = Polygon(np.array([x,y]).T)

        path = polygon.get_path()
        spine = mspines.Spine(axes=self, spine_type='circle', path=path)
        spine.set_transform(self.transAxes)
        return {'geo': spine}

register_projection(EquirectangularAxes)

class SinusoidalAxes(GeoAxes):
    name = 'sinusoidal'

    class SinusoidalTransform(Transform):
        input_dims = output_dims = 2

        def __init__(self, resolution):
            Transform.__init__(self)
            self._resolution = resolution

        def transform_non_affine(self, ll):
            l, phi = ll.T
            x = np.zeros_like(l)
            y = np.zeros_like(l)
            n = 12
            l0s = np.arange(-np.pi, np.pi, 2*np.pi/n) + np.pi/n
            for i in range(n):
                inds = (l >= (l0s[i]-np.pi/n)) & (l <= (l0s[i]+np.pi/n))
                l0 = l0s[i]
                x[inds] = (l[inds]-l0)*np.cos(phi[inds]) + l0
                y[inds] = phi[inds]

            # Nasty stuff happens at the boundaries
            inds = (l == -np.pi)
            x[inds] = -(l[inds] - l0s[0]) + l0s[0]
            y[inds] = phi[inds]

            inds = (l == np.pi)
            x[inds] = (l[inds] - l0s[-1]) + l0s[-1]
            y[inds] = phi[inds]

            inds = (l < -np.pi)
            x[inds] = -np.pi
            y[inds] = phi[inds]

            inds = (l > np.pi)
            x[inds] = np.pi
            y[inds] = phi[inds]

            return np.column_stack([x, y])

        def transform_path_non_affine(self, path):
            # vertices = path.vertices
            ipath = path.interpolated(self._resolution)
            return Path(self.transform(ipath.vertices), ipath.codes)

        def inverted(self):
            return SinusoidalAxes.InvertedSinusoidalTransform(self._resolution)

    class InvertedSinusoidalTransform(Transform):
        input_dims = output_dims = 2

        def __init__(self, resolution):
            Transform.__init__(self)
            self._resolution = resolution

        def transform_non_affine(self, xy):
            x, y = xy.T

            phi = y
            l = x

            return np.column_stack([l, phi])

        def inverted(self):
            return SinusoidalAxes.SinusoidalTransform(self._resolution)

    def __init__(self, *args, **kwargs):
        self._longitude_cap = np.pi / 2.0
        super().__init__(*args, **kwargs)
        self.set_aspect(0.5, adjustable='box', anchor='C')
        self.cla()

    def _get_core_transform(self, resolution):
        return self.SinusoidalTransform(resolution)

    def _gen_axes_patch(self):
        """
        Override this method to define the shape that is used for the
        background of the plot.  It should be a subclass of Patch.

        In this case, it is a Circle (that may be warped by the axes
        transform into an ellipse).  Any data and gridlines will be
        clipped to this shape.
        """
        x = []
        y = []

        phi_l = np.linspace(0, np.pi/2)
        phi_r = np.linspace(0, np.pi/2)[::-1]
        phi = np.concatenate((phi_l, phi_r))

        n = 12
        l0s = np.arange(-np.pi, np.pi, 2*np.pi/n) + np.pi/n
        for i in range(n):
            l_l = np.ones_like(phi_l)*(l0s[i]-np.pi/n)
            l_r = np.ones_like(phi_l)*(l0s[i]+np.pi/n)
            l = np.concatenate((l_l, l_r))
            l0 = l0s[i]
            x.append((l-l0)*np.cos(phi) + l0)
            y.append(phi)

        phi_r = np.linspace(0, -np.pi/2)
        phi_l = np.linspace(0, -np.pi/2)[::-1]
        phi = np.concatenate((phi_r, phi_l))
        for i in range(n):
            l_l = np.ones_like(phi_l)*(l0s[i]-np.pi/n)
            l_r = np.ones_like(phi_l)*(l0s[i]+np.pi/n)
            l = np.concatenate((l_r, l_l))
            l0 = l0s[i]
            x.append((l-l0)*np.cos(phi) + l0)
            y.append(phi)

        x = np.array(x).flatten()/(2*np.pi) + 0.5
        y = np.array(y).flatten()/np.pi + 0.5

        return Polygon(np.array([x,y]).T)

    def _gen_axes_spines(self):
        x = np.array([0,1,1,0,0])
        y = np.array([0,0,1,1,0])
        polygon = Polygon(np.array([x,y]).T)

        path = polygon.get_path()
        spine = mspines.Spine(axes=self, spine_type='circle', path=path)
        spine.set_transform(self.transAxes)
        return {'geo': spine}

register_projection(SinusoidalAxes)

class PolyconicAxes(GeoAxes):
    name = 'polyconic'

    class PolyconicTransform(Transform):
        input_dims = output_dims = 2

        def __init__(self, resolution):
            Transform.__init__(self)
            self._resolution = resolution

        def transform_non_affine(self, ll):
            l, phi = ll.T

            l0 = 0
            phi0 = 0

            ind = (phi != 0)
            x = np.zeros_like(phi)
            y = np.zeros_like(phi)

            #x[ind] = np.sin((l[ind]-l0)*np.sin(phi[ind]))/np.tan(phi[ind])
            #y[ind] = phi[ind] - phi0 + (1-np.cos((l[ind]-l0)*np.sin(phi[ind])))/np.tan(phi[ind])

            #x[~ind] = l[~ind] - l0
            #y[~ind] = -phi0


            n = 12
            l0s = np.arange(-np.pi, np.pi, 2*np.pi/n) + np.pi/n
            for i in range(n):
                l0 = l0s[i]
                ind = (l >= (l0-np.pi/n)) & (l <= (l0+np.pi/n)) & (phi != 0)
                x[ind] = np.sin((l[ind]-l0)*np.sin(phi[ind]))/np.tan(phi[ind]) + l0
                y[ind] = phi[ind] - phi0 + (1-np.cos((l[ind]-l0)*np.sin(phi[ind])))/np.tan(phi[ind])
                ind = (l >= (l0-np.pi/n)) & (l <= (l0+np.pi/n)) & (phi == 0)
                x[ind] = l[ind]
                y[ind] = -phi0

            badinds = (x == 0) & (y == 0)

            # Nasty stuff happens at the boundaries
            inds = (l == -np.pi)
            x[inds] = -(l[inds] - l0s[0]) + l0s[0]
            y[inds] = phi[inds]

            inds = (l == np.pi)
            x[inds] = (l[inds] - l0s[-1]) + l0s[-1]
            y[inds] = phi[inds]

            inds = (l < -np.pi)
            x[inds] = -np.pi
            y[inds] = phi[inds]

            inds = (l > np.pi)
            x[inds] = np.pi
            y[inds] = phi[inds]


            return np.column_stack([x, y])

        def transform_path_non_affine(self, path):
            # vertices = path.vertices
            ipath = path.interpolated(self._resolution)
            return Path(self.transform(ipath.vertices), ipath.codes)

        def inverted(self):
            return PolyconicAxes.InvertedPolyconicTransform(self._resolution)

    class InvertedPolyconicTransform(Transform):
        input_dims = output_dims = 2

        def __init__(self, resolution):
            Transform.__init__(self)
            self._resolution = resolution

        def transform_non_affine(self, xy):
            x, y = xy.T
            R = 1
            phi_0 = 0
            phi_1 = 0
            l_0 = 0

            l = x/(R*np.cos(phi_1)) + l_0
            phi = y/R + phi_0

            return np.column_stack([l, phi])

        def inverted(self):
            return PolyconicAxes.PolyconicTransform(self._resolution)

    def __init__(self, *args, **kwargs):
        self._longitude_cap = np.pi / 2.0
        super().__init__(*args, **kwargs)
        self.set_aspect(0.5, adjustable='box', anchor='C')
        self.cla()

    def _get_core_transform(self, resolution):
        return self.PolyconicTransform(resolution)

    def _gen_axes_patch(self):
        """
        Override this method to define the shape that is used for the
        background of the plot.  It should be a subclass of Patch.

        In this case, it is a Circle (that may be warped by the axes
        transform into an ellipse).  Any data and gridlines will be
        clipped to this shape.
        """
        x = []
        y = []

        phi_l = np.linspace(0, np.pi/2)
        phi_r = np.linspace(0, np.pi/2)[::-1]
        phi = np.concatenate((phi_l, phi_r))

        n = 12
        l0s = np.arange(-np.pi, np.pi, 2*np.pi/n) + np.pi/n
        for i in range(n):
            l_l = np.ones_like(phi_l)*(l0s[i]-np.pi/n)
            l_r = np.ones_like(phi_l)*(l0s[i]+np.pi/n)
            l = np.concatenate((l_l, l_r))
            l0 = l0s[i]
            xi = np.sin((l-l0)*np.sin(phi))/np.tan(phi) + l0
            yi = phi + (1-np.cos((l-l0)*np.sin(phi)))/np.tan(phi)
            xi[phi == 0] = l[phi==0]
            yi[phi == 0] = 0
            x.append(xi)
            y.append(yi)

        phi_r = np.linspace(0, -np.pi/2)
        phi_l = np.linspace(0, -np.pi/2)[::-1]
        phi = np.concatenate((phi_r, phi_l))
        for i in range(n):
            l_l = np.ones_like(phi_l)*(l0s[i]-np.pi/n)
            l_r = np.ones_like(phi_l)*(l0s[i]+np.pi/n)
            l = np.concatenate((l_r, l_l))
            l0 = l0s[i]
            xi = np.sin((l-l0)*np.sin(phi))/np.tan(phi) + l0
            yi = phi + (1-np.cos((l-l0)*np.sin(phi)))/np.tan(phi)
            xi[phi == 0] = l[phi==0]
            yi[phi == 0] = 0
            x.append(xi)
            y.append(yi)

        x = np.array(x).flatten()/(2*np.pi) + 0.5
        y = np.array(y).flatten()/np.pi + 0.5
        #x = np.array([0,1,1,0,0])
        #y = np.array([0,0,1,1,0])

        return Polygon(np.array([x,y]).T)

    def _gen_axes_spines(self):
        x = np.array([0,1,1,0,0])
        y = np.array([0,0,1,1,0])
        polygon = Polygon(np.array([x,y]).T)

        path = polygon.get_path()
        spine = mspines.Spine(axes=self, spine_type='circle', path=path)
        spine.set_transform(self.transAxes)
        return {'geo': spine}

register_projection(PolyconicAxes)

class CassiniAxes(GeoAxes):
    name = 'cassini'

    class CassiniTransform(Transform):
        input_dims = output_dims = 2

        def __init__(self, resolution):
            Transform.__init__(self)
            self._resolution = resolution

        def transform_non_affine(self, ll):
            l, phi = ll.T

            x = np.arcsin(np.cos(phi)*np.sin(l))
            y = np.arctan2(np.sin(phi), np.cos(phi)*np.cos(l))

            x = np.zeros_like(phi)
            y = np.zeros_like(phi)
            n = 12
            l0s = np.arange(-np.pi, np.pi, 2*np.pi/n) + np.pi/n
            for i in range(n):
                l0 = l0s[i]
                ind = (l >= (l0-np.pi/n)) & (l <= (l0+np.pi/n))
                phii = phi[ind]
                li = l[ind]
                x[ind] = np.arcsin(np.cos(phii)*np.sin(li-l0)) + l0
                y[ind] = np.arctan2(np.sin(phii), np.cos(phii)*np.cos(li-l0))

            # Nasty stuff happens at the boundaries
            inds = (l == -np.pi)
            x[inds] = -(l[inds] - l0s[0]) + l0s[0]
            y[inds] = phi[inds]

            inds = (l == np.pi)
            x[inds] = (l[inds] - l0s[-1]) + l0s[-1]
            y[inds] = phi[inds]

            inds = (l < -np.pi)
            x[inds] = -np.pi
            y[inds] = phi[inds]

            inds = (l > np.pi)
            x[inds] = np.pi
            y[inds] = phi[inds]

            return np.column_stack([x, y])

        def transform_path_non_affine(self, path):
            # vertices = path.vertices
            ipath = path.interpolated(self._resolution)
            return Path(self.transform(ipath.vertices), ipath.codes)

        def inverted(self):
            return CassiniAxes.InvertedCassiniTransform(self._resolution)

    class InvertedCassiniTransform(Transform):
        input_dims = output_dims = 2

        def __init__(self, resolution):
            Transform.__init__(self)
            self._resolution = resolution

        def transform_non_affine(self, xy):
            x, y = xy.T

            phi = np.arcsin(np.sin(y)*np.cos(x))
            l = np.arctan2(np.tan(x), np.cos(y))

            return np.column_stack([l, phi])

        def inverted(self):
            return CassiniAxes.CassiniTransform(self._resolution)

    def __init__(self, *args, **kwargs):
        self._longitude_cap = np.pi / 2.0
        super().__init__(*args, **kwargs)
        self.set_aspect(0.5, adjustable='box', anchor='C')
        self.cla()

    def _get_core_transform(self, resolution):
        return self.CassiniTransform(resolution)

    def _gen_axes_patch(self):
        """
        Override this method to define the shape that is used for the
        background of the plot.  It should be a subclass of Patch.

        In this case, it is a Circle (that may be warped by the axes
        transform into an ellipse).  Any data and gridlines will be
        clipped to this shape.
        """
        x = []
        y = []

        phi_l = np.linspace(0, np.pi/2)
        phi_r = np.linspace(0, np.pi/2)[::-1]
        phi = np.concatenate((phi_l, phi_r))

        n = 12
        l0s = np.arange(-np.pi, np.pi, 2*np.pi/n) + np.pi/n
        for i in range(n):
            l_l = np.ones_like(phi_l)*(l0s[i]-np.pi/n)
            l_r = np.ones_like(phi_l)*(l0s[i]+np.pi/n)
            l = np.concatenate((l_l, l_r))
            l0 = l0s[i]
            x.append(np.arcsin(np.cos(phi)*np.sin(l-l0)) + l0)
            y.append(np.arctan2(np.sin(phi), np.cos(phi)*np.cos(l-l0)))

        phi_r = np.linspace(0, -np.pi/2)
        phi_l = np.linspace(0, -np.pi/2)[::-1]
        phi = np.concatenate((phi_r, phi_l))
        for i in range(n):
            l_l = np.ones_like(phi_l)*(l0s[i]-np.pi/n)
            l_r = np.ones_like(phi_l)*(l0s[i]+np.pi/n)
            l = np.concatenate((l_r, l_l))
            l0 = l0s[i]
            x.append(np.arcsin(np.cos(phi)*np.sin(l-l0)) + l0)
            y.append(np.arctan2(np.sin(phi), np.cos(phi)*np.cos(l-l0)))

        x = np.array(x).flatten()/(2*np.pi) + 0.5
        y = np.array(y).flatten()/np.pi + 0.5

        return Polygon(np.array([x,y]).T)

    def _gen_axes_spines(self):
        #x = []
        #y = []
        #n = 12
        #l0s = np.arange(-np.pi, np.pi, 2*np.pi/n) + np.pi/n

        #phi_l = np.linspace(0, np.pi/2)
        #phi_r = np.linspace(0, np.pi/2)[::-1]
        #phi = np.concatenate((phi_l, phi_r))

        #for i in range(n):
        #    l_l = np.ones_like(phi_l)*(l0s[i]-np.pi/n)
        #    l_r = np.ones_like(phi_l)*(l0s[i]+np.pi/n)
        #    l = np.concatenate((l_l, l_r))
        #    l0 = l0s[i]
        #    x.append((l-l0)*np.cos(phi) + l0)
        #    y.append(phi)

        #phi_r = np.linspace(0, -np.pi/2)
        #phi_l = np.linspace(0, -np.pi/2)[::-1]
        #phi = np.concatenate((phi_r, phi_l))
        #l0s = l0s[::-1]
        #for i in range(n):
        #    l_l = np.ones_like(phi_l)*(l0s[i]-np.pi/n)
        #    l_r = np.ones_like(phi_l)*(l0s[i]+np.pi/n)
        #    l = np.concatenate((l_r, l_l))
        #    l0 = l0s[i]
        #    x.append((l-l0)*np.cos(phi) + l0)
        #    y.append(phi)

        #x = np.array(x).flatten()/(2*np.pi) + 0.5
        #y = np.array(y).flatten()/np.pi + 0.5
        #polygon = Polygon(np.array([x,y]).T)

        #path = polygon.get_path()
        #spine = mspines.Spine(axes=self, spine_type='circle', path=path)
        #spine.set_transform(self.transAxes)
        #return {'geo': spine}
        x = np.array([0,1,1,0,0])
        y = np.array([0,0,1,1,0])
        polygon = Polygon(np.array([x,y]).T)

        path = polygon.get_path()
        spine = mspines.Spine(axes=self, spine_type='circle', path=path)
        spine.set_transform(self.transAxes)
        return {'geo': spine}

register_projection(CassiniAxes)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from healpy.newvisufunc import projview, newprojplot
    import healpy as hp

    lmax = 256
    ell = np.arange(2, lmax)
    Cl = np.zeros(lmax)
    Cl[2:] = 1./ell**2
    np.random.seed(0)
    m1 = hp.synfast(Cl, lmax)


    m2 = np.arange(12)

    ms = [m1, m2]
    ms = [m1]


    for i, m in enumerate(ms):
        #projview(m, projection_type='mollweide', graticule=True,
            #graticule_labels=True)
        #projview(m, projection_type='hammer', graticule=True, graticule_labels=True)
        #projview(m, projection_type='custom_hammer', graticule=True, graticule_labels=True)
        #projview(m, projection_type='healpix', graticule=True, graticule_labels=True)
        #projview(m, projection_type='equirectangular', graticule=False,
        #    graticule_labels=False, cbar=False)
        #projview(m, projection_type='polyconic', graticule=True,
        #    graticule_labels=False, cbar=False)
        projview(m, projection_type='sinusoidal', graticule=False,
            graticule_labels=False, cbar=False)
        projview(m, projection_type='custom_hammer', graticule=True,
            graticule_labels=False, cbar=False)
        #projview(m, projection_type='mollweide', graticule=True,
        #    graticule_labels=False, cbar=False)
        #projview(m, projection_type='cart', graticule=False,
        #    graticule_labels=False, cbar=False)
        # For some reason, using plt.savefig really messes with the healpix
        # projection
        plt.show()

