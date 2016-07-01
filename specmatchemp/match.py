"""
@filename match.py

Defines the Match class
"""
import pandas as pd
import numpy as np
import lmfit
from scipy.interpolate import UnivariateSpline
import scipy.ndimage as nd

import specmatchemp.kernels

class Match:
    def __init__(self, wav, s_targ, serr_targ, s_ref, serr_ref):
        """
        The Match class used for matching two spectra

        Args:
            wav (np.ndarray): Common wavelength scale
            s_targ (np.ndarray): Target spectrum
            serr_targ (np.ndarray): Uncertainty in target spectrum
            s_ref (np.ndarray): Reference spectrum
            serr_ref (np.ndarray): Uncertainty in reference spectrum
        """
        self.w = wav
        self.s_targ = s_targ
        self.serr_targ = serr_targ
        self.s_ref = s_ref
        self.serr_ref = serr_ref
        self.best_params = lmfit.Parameters()
        self.best_chisq = np.NaN

    def create_model(self, params):
        """
        Creates a tweaked model based on the parameters passed,
        based on the reference spectrum.
        Stores the tweaked model in spectra.s_mod and serr_mod.
        """
        # Create a spline
        x = []
        y = []
        for i in range(params['num_knots'].value):
            p = 'knot_{0:d}'.format(i)
            x.append(params[p+'_x'].value)
            y.append(params[p+'_y'].value)
        s = UnivariateSpline(x, y, s=0)
        spl = s(self.w)

        self.s_mod = spl*self.s_ref
        self.serr_mod = spl*self.serr_ref

        # Apply broadening kernel
        SPEED_OF_LIGHT = 2.99792e5
        dv = (self.w[1]-self.w[0])/self.w[0]*SPEED_OF_LIGHT
        n = 151 # fixed number of points in the kernel
        vsini = params['vsini'].value
        varr, kernel = specmatchemp.kernels.rot(n, dv, vsini)
        self.s_mod = nd.convolve1d(self.s_mod, kernel)
        self.serr_mod = nd.convolve1d(self.serr_mod, kernel)

    def residual(self, params):
        """
        Objective function evaluating goodness of fit given the passed parameters

        Args:
            params
        Returns:
            Reduced chi-squared value between the target spectra and the 
            model spectrum generated by the parameters
        """
        self.create_model(params)

        # Calculate residuals
        diff = np.abs(self.s_targ-self.s_mod)
        # variance = np.sqrt(self.serr_targ**2+self.serr_mod**2)

        # return diffsq/variance

        return diff

    def best_fit(self):
        """
        Calculates the best fit model by minimizing over the parameters:
        - spline fitting to the continuum
        - rotational broadening
        """
        # Create a spline with 5 knots
        params = lmfit.Parameters()
        num_knots = 5
        params.add('num_knots', value=num_knots, vary=False)
        interval = int(len(self.w)/(num_knots+1))

        # Add spline positions
        for i in range(num_knots):
            p = 'knot_{0:d}'.format(i)
            params.add(p+'_x', value=self.w[interval*i], vary=False)
            params.add(p+'_y', value=self.s_targ[interval*i], min=0.5, max=1.5)

        # Rotational broadening
        params.add('vsini', value=10.0, min=0.0, max=15.0)

        # Minimize chi-squared
        out = lmfit.minimize(self.residual, params)

        # Save best fit parameters
        self.best_params = out.params
        self.best_chisq = out.redchi
        self.create_model(self.best_params)

        return self.best_chisq

