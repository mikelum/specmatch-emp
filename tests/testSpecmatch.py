#! /usr/bin/env python
# 
# Program: testSpecMatch.py
#
# Author: Mike Lum
#
# Usage: ./testSpecMatch.py [-vh]
#
# Description: Runs through some basic tests with sample spectra in the Spectra
#           directory.
#
# Revision History:
#    Date        Vers.    Author        Description
#    11/13/16    N/A      Lum        First checked in
#
# To Do:
#    
#
# For parsing the command line call
import os
from collections import deque

import astropy.io.fits as pyfits
import pylab as pl
import numpy as np

from specmatchemp.spectrum import Spectrum
from specmatchemp.specmatch import SpecMatch
import specmatchemp.library
#import specmatchemp.plots as smplot


# Star Parameters:
# List of test spectra.
# Each entry is of the form:
#       [
#        '<spectra path/filename>',
#        'Spectra Name', 
#        {stellar parm dict. For keys, see above}
#        unit multiplier (float) # Axis assumes angstroms (use mult=1.), 
#               other units use conversion to Angstroms (ie: nanometer: mult=10.)
#       ]
testSpectra = [['Spectra/HARPS_Sun.fits', 'Sun', {'Teff':5778, 'LogG':4.44, 'feh':0.00, 'Vmicro':1.06, 'Vmacro':4.00, 'Vsini':1.6}, 10.],\
['Spectra/NARVAL_HD22879.fits', 'HD22879', {'Teff':5868, 'LogG':4.27, 'feh':-0.88, 'Vmicro':1.09, 'Vmacro':5.08, 'Vsini':4.4}, 10.],\
['Spectra/HARPS_alfCenB.fits', 'Alpha Cen. B', {'Teff':5231, 'LogG':4.53, 'feh':0.22, 'Vmicro':0.92, 'Vmacro':2.80, 'Vsini':1.0}, 10.],\
['Spectra/UVES.POP_Procyon.fits', 'Procyon', {'Teff':6554, 'LogG':4.00, 'feh':-0.04, 'Vmicro':1.48, 'Vmacro':9.57, 'Vsini':2.8}, 10.],\
['Spectra/ESPaDOnS_61CygB.fits', '61BCyg', {'Teff':4044, 'LogG':4.67, 'feh':-0.38, 'Vmicro':0.83, 'Vmacro':4.99, 'Vsini':1.7}, 10.],\
['Spectra/Sophie_Vega.fits', 'Vega', {'Teff':9602, 'LogG':3.95, 'feh':-0.50}, 1.],\
]

#ngc752dir = '/home/mikelum/Dropbox/CodeCloud/MyTools/ClusterAnalysis/SpectraProcessing/Fluxes/'
#ngc752Spectra = [\
#[ngc752dir+'-PLA-0300/dc_45853_0.fits', 'PLA-300', {'Teff':5447, 'LogG':4.54, 'feh':-0.09, 'Vmicro':0.76, 'Vmacro':4.00, 'Vsini':1.6}, 1.],\
#[ngc752dir+'-PLA-0475/dc_42480_0.fits', 'PLA-475', {'Teff':5918, 'LogG':4.50, 'feh':-0.09, 'Vmicro':1.18, 'Vmacro':4.00, 'Vsini':1.6}, 1.],\
#[ngc752dir+'-PLA-0300/dc_45853_0.fits', 'PLA-300', {'Teff':5447, 'LogG':4.54, 'feh':-0.09, 'Vmicro':0.76, 'Vmacro':4.00, 'Vsini':1.6}, 1.],\
#]


# Giant Stars
#['Spectra/arcturus_NOAO.fits', 'Arcturus', {'Teff':4286, 'LogG':1.66, 'feh':-0.52, 'Vmicro':1.74}, 1.],\
#['Spectra/HARPS.GBOG_psiPhe.fits', 'Psi Phoenix', {'Teff':3472, 'LogG':0.51, 'feh':-1.23, 'Vmicro':1.55, 'Vmacro':6.14, 'Vsini':3.0}, 10.],\

# Fitting range:
rangeMin = 5050.
rangeMax = 5140.

# Utility functions:
def is_number(testString):
    # A little trick for nan: (nan == nan) = False, but we're not using that here...
    # even though it's probably faster than two attempted float conversions...
    
# Note: "real" nan values are actually floats, and won't be passed as strings.
# Basically, this means that EVERY call to this function (with a string parameter)
# will hit the following exception.  
    try:
        if np.isnan(testString):
            return False
    except TypeError:
        pass
 
    try:
        float(testString)
    except ValueError:
        return False
  
    if np.isnan(float(testString)):
        return False
        
    return True
    
def plotBestMatch(match2Plot, starParms={}):
    fig = pl.figure(figsize=(12,8))
    match2Plot.plot_chi_squared_surface()
    ax = fig.axes
    if 'Teff' in starParms.keys():
        ax[0].axvline(starParms['Teff'], color='k')
    if 'LogG' in starParms.keys():
        ax[1].axvline(starParms['LogG'], color='k')
    if 'feh' in starParms.keys():
        ax[2].axvline(starParms['feh'], color='k')
    pl.show()
    return
    
def plotMatches(match2Plot):

    fig = pl.figure(figsize=(10,5))
    match2Plot.target_unshifted.plot(normalize=True, plt_kw={'color':'forestgreen'}, text='Target (unshifted)')
    match2Plot.target.plot(offset=0.5, plt_kw={'color':'royalblue'}, text='Target (shifted):{0}'.format(match2Plot.target.name))
    match2Plot.shift_ref.plot(offset=1, plt_kw={'color':'firebrick'}, text='Reference: '+match2Plot.shift_ref.name)
    pl.xlim(rangeMin+5,rangeMax-5)
    pl.ylim(0,2.2)
    
    pl.show()
    return

def plotSpec(theSpec):
    fig = pl.figure(figsize=(10,5))
    theSpec.plot(plt_kw={'color':'royalblue'}, text=theSpec.name)
    pl.xlim(rangeMin-5,rangeMax+5)
    pl.ylim(0,1.3)
    
    pl.show()
    return

def testSpecMatch(specFileData):
# specFileData is of the form:
#       [
#        '<spectra path/filename>',
#        'Spectra Name', 
#        {stellar parm dict. For keys, see above}
#        unit multiplier (float) # Axis assumes angstroms (use mult=1.), 
#               other units use conversion to Angstroms (ie: nanometer: mult=10.)
#       ]
    # Hack to load spectra that weren't in a "happy" format
    myHDU = pyfits.open(specFileData[0])
    print myHDU[0].header.keys()
    wlStart = float(myHDU[0].header['CRVAL1'])*specFileData[3]
    wlDisp = float(myHDU[0].header['CDELT1'])*specFileData[3]
    numPoints = int(myHDU[0].header['NAXIS1'])

    wlVals = np.linspace(wlStart, np.round(wlStart+wlDisp*numPoints, 2), numPoints)
    print wlVals
    flxVals = np.array([flx if is_number(flx) else 1.0 for flx in myHDU[0].data])

    fullSpectrum=Spectrum(wlVals, flxVals, name=specFileData[1])

#    plotSpec(fullSpectrum)

    mySpectrum = fullSpectrum.cut(rangeMin-10, rangeMax+10)
    lib = specmatchemp.library.read_hdf(wavlim=[rangeMin-10, rangeMax+10])

    myMatches = SpecMatch(mySpectrum, lib=lib, wavlim=(rangeMin-10, rangeMax+10))
    
    myMatches.shift()

    plotMatches(myMatches)

    # Note: if the spectra cut size is smaller than the default wavstep, bad things
    # happen.  
    myMatches.match(wavstep=20.)

    plotBestMatch(myMatches, starParms=specFileData[2])

    # Generate best-fit parms
    myMatches.lincomb()
    
    print('Derived Parameters: ')
    print('Teff:{0:.0f}+/-{3:.0f}, LogG:{1:.2f}+/-{4:.2f}, [Fe/H]:{2:.2f}+/-{5:.2f}, Age:{6:f}+/-{7:f}'.format(
        myMatches.results['Teff'], myMatches.results['logg'], myMatches.results['feh'], myMatches.results['u_Teff'], myMatches.results['u_logg'], myMatches.results['u_feh'], myMatches.results['age'], myMatches.results['u_age']))
    return


def printHelpText():
    print('Program: testSpecmatch.py\n')
    print('Run a series of functional tests on sample spectra.\n')
    print('Usage: ./testSpecmatch.py [-hvi]\n')
    print('Options:')
    print('-h: Print this help text.')
    print('-v: Use verbose progress and error messages (Currently unused)')
    print('-i <index>: Run tests on spectra index #<index>. Default=0 (Solar spectrum)')


if __name__ == '__main__':

    # Parse your command line call
    temp = os.sys.argv
    argv = deque(temp)
    index = 0
    
    while len(argv) > 0:
        flag = argv.popleft()
        if flag == '-h':
            printHelpText()
            exit()
        elif flag == '-v':
            verboseMode = True
            print('Verbose Mode enabled.')
        elif flag == '-i':
            try:
                index = int(argv.popleft())
            except ValueError:
                print ("Invalid index following \'-i\'. Continuing")
        else:
        # Ignore everything else, or not...
            pass
    
    # Call your program here
    testSpecMatch(testSpectra[index])
    exit()
