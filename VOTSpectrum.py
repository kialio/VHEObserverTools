#!/usr/bin/env python

import numpy as np
from scipy import interpolate
from glob import glob

from VOTUtils import initLogger,checkParams

class VOTSpectrum:

    '''This class takes a spectral shape and returns the spectral shape
    along with the absorbed spectral shape.  You can either pass an
    array of energies in GeV and an array of differential flux points
    in photon/cm^2/GeV/s or use one of the provided spectral shapes.
    You can then access the energy array and the original spectral
    shape along with the absorbed spectral shape.'''

    def __init__(self,EBins = [], dNdE = [], Emin=1, Emax=10000, Nbins=100,
                 eblModel="Dominguez",redshift=0.5,spectralModel="PowerLaw", 
                 **spectralPars):


        self.logger = initLogger('VOTSpectrum')

        eblModels = {"Null": self.loadNullTau,
                     "Simple": self.loadSimpleTau,
                     "Dominguez": self.loadDominguezTau,
                     "Finke": self.loadFinkeTau,
                     "Kneiske": self.loadKneiskeTau}

        if(not EBins):
            self.EBins = self.generateEnergyBins(Emin,Emax,Nbins,"GeV")
        else:
            self.EBins = EBins
        if(not dNdE):
            self.dNdE = self.Spectrum(spectralModel,self.EBins,**spectralPars)
        else:
            self.dNdE = dNdE

        try:
            z_array,e_array,t_array = eblModels[eblModel]()
        except KeyError:
            self.logger.critical("Unsupported EBL model.Exiting")
            return

        self.interpolated_taus = self.interpolateTaus(redshift, self.EBins, z_array, e_array, t_array)
        
        self.dNdE_absorbed = self.absorbSpectrum(self.dNdE,self.interpolated_taus[0:,0])

    def helpEBL(self):
        
        '''                             - EBL Models - 
        
        These are the currently available EBL models which properly
        account for attenuation of the gamma-ray signal due to
        absorpotion off of the IR background.  The individual tau
        values are stored in the Tau_Data directory and are
        interpolated linerarly in energy and redshift with a bivarite
        spline function.

        'Null': this is a null EBL model that returns tau values of 0
        for all redshifts and energies.  This is an appropriate
        function for nearby objects like the Crab.  Note that the
        other functions will not return 0 even for nearby objects but
        will return the lowest value in the Tau table.

        'Simple': this is a very simple EBL table used for testing.
        It only has 4 energies and 3 redshifts and the tau values are
        completely made up.

        'Dominguez': this is a model from "Extragalactic background
        light inferred from AEGIS galaxy-SED-type fractions",
        A. Dominguez et al., 2011, MNRAS, 410, 2556.  It contains
        values for 39 redshifts from 0.01 to 2.0 and a wide range of
        energies.

        'Finke': this is a model from "Modeling the Extragalactic
        Background Light from Stars and Dust", Finke, Razzaque, &
        Dermer, 2010, ApJ, 712, 238.  It contains values for 500
        redshifts from 0.0 to 4.99 and a wide range of energies.

        'Kneiske': this is a model from "A strict lower-limit EBL
        Applications on gamma-ray absorption", Kneiske & Dole,
        arXiv:0810.1612.  It contains values for 200 redshifts from
        0.01055458 to 4.990822 and a wide range of energies.

        '''
        pass

    def generateEnergyBins(self,Emin=100,Emax=100000,Nbins=100,units="GeV"):

        '''Units can be MeV, GeV, or TeV.  All internal calculations are done
        in GeV so need to take care of this now.'''
            
        eConv = {"MeV": lambda x: x*0.001,
                 "GeV": lambda x: x,
                 "TeV": lambda x: x*1000.}

        return np.logspace(np.log10(eConv[units](Emin)),np.log10(eConv[units](Emax)),Nbins)
            
    def Spectrum(self,spectralModel, EnergyBins, **pars):

        '''                       - Spectral Functions - 
        
        The VOT code uses a spectrum (dN/dE) given an input array of
        energy bins (in GeV).  There are several available types
        avaialble and these are the only ones you can use to create a
        'custom' source.  The parameters given must match the
        parameters and units shown below.  Most of these functions are
        based on the functions in the Fermi LAT catalog except as
        noted.

        PowerLaw: a simple power law with normalization ('N0', cm^-2
        s^-1 GeV^-1), index ('index', negative), pivot energy ("E0",
        GeV).

        PowerLaw2: same as the PowerLaw but defined by the integral
        flux and not the differential flux.  Parameters are the
        integral flux ('N', cm^-2 s^-1 GeV^-1), index ('index',
        negative), lower energy limit ('E1', GeV) and upper energy
        limit ('E2', GeV).

        BrokenPowerLaw: two component power law with normalization
        ('N0', cm^-2 s^-1 GeV^-1), low energy spectral index
        ('index1', negative), high energy spectral index ('index2',
        negative) and break energy ('Eb', GeV).

        LogParabola: curving function used to model blazars with
        parameters Normalization ('N0', cm^-2 s^-1 GeV^-1), spectral
        index ('alpha', negative), index ('beta, negative) and break
        energy ('Eb', GeV).

        HESSExpCutoff: power law with a cutoff used by the HESS
        collaboration to model the Crab Nebula's spectrum.  The
        parameters are the normalization ('N0', cm^-2 s^-1 GeV^-1),
        index ('index', negative), pivot energy ('E0', GeV) and cutoff
        energy ('EC', GeV).

        Band: basic band function used to model GRBs.  The parameters
        are the normalization ('N0', cm^-2 s^-1 GeV^-1), alpha
        ('alpha', negative), beta ('beta', negative) and the peak
        energy ('Ep', GeV).

        '''

        spectralParameterTemplates = {"PowerLaw": {"N0":1e-9,"index":-2,"E0":700},
                                      "PowerLaw2": {"N":1e-6,"index":-2,"E1":1,"E2":100},
                                      "BrokenPowerLaw": {"N0":1e-9,"index1":-2,"index2":-1.5,"Eb":700},
                                      "LogParabola": {"N0":1e-9,"alpha":-1.0,"beta":-2.0,"Eb":700},
                                      "HESSExpCutoff": {"N0":1e-9, "index":-2, "E0":1000, "EC":14000},
                                      "Band": {"N0":1e-9, "alpha":-1.5,"beta":-2.5,"Ep":0.001},
                                      }
    
        if (spectralModel == "PowerLaw2"):
            '''Assumes a 1GeV Decorrelation energy and energies in GeV.'''
            pars["E0"] = 1.0
            X1 = pars["E1"]/pars["E0"]
            X2 = pars["E2"]/pars["E0"]
            pars["N0"] = (pars["N"]*(1 + pars["index"])/pars["E0"])/(X2**(1 + pars["index"]) - X1**(1 + pars["index"]))
            spectralModel = "PowerLaw"
            self.logger.info("Calculated normalization: " + str(pars["N0"]) + " s^-1 cm^-2 GeV^-1")

        spectralFunction = {"PowerLaw": lambda x: pars["N0"]*((x/pars["E0"])**pars["index"]),
                            "PowerLaw2": lambda x: ((pars["index"] + 1)*pars["N"]*x**pars["index"])/\
                                (pars["E2"]**(pars["index"]+1) - pars["E1"]**(pars["index"]+1)),
                            "BrokenPowerLaw": lambda x: x < pars["Eb"] and pars["N0"]*((x/pars["Eb"])**pars["index1"])\
                                or pars["N0"]*((x/pars["Eb"])**pars["index2"]),
                            "LogParabola": lambda x: pars["N0"]*((x/pars["Eb"])**(pars["alpha"]+pars["beta"]*np.log(x/pars["Eb"]))),
                            "HESSExpCutoff":lambda x: pars["N0"]*((x/pars["E0"])**pars["index"])*np.exp(-x/pars["EC"]),
                            "Band": lambda x: x < pars["Ep"]*(pars["alpha"]-pars["beta"])/(pars["alpha"]+2)\
                                and (x/0.1)**pars["alpha"]*np.exp(-(x/pars["Ep"])*(pars["alpha"]+2))\
                                or (x/0.1)**pars["beta"]*((pars["Ep"]/0.1)*(pars["alpha"]-pars["beta"])/(pars["alpha"]+2))**(pars["alpha"]-pars["beta"])*np.exp(pars["beta"]-pars["alpha"]),
                            "Band1": lambda x: (x/0.1)**pars["beta"]*((pars["Ep"]/0.1)*(pars["alpha"]-pars["beta"])/(pars["alpha"]+2))**(pars["alpha"]-pars["beta"])*np.exp(pars["beta"]-pars["alpha"]),
                            }
    
        try:
            checkParams(spectralParameterTemplates[spectralModel],pars, self.logger)
        except KeyError:
            return

        dNdE = []

        try:
            for energy in EnergyBins:
                dNdE.append(spectralFunction[spectralModel](energy))
            return dNdE
        except KeyError:
            self.logger.critical("No such model.")
        
    def loadFinkeTau(self, folder="Tau_Data/finke"):

        '''Returns an array of redshifts, an array of energies [GeV] and an
        array of taus where the rows are for the individual redshifts
        listed in the redshift array and the columns are for different
        energies.'''

        filenames = [filename for filename in glob(folder+"/*.dat")]
        redshifts = [float(filename[-8:-4]) for filename in filenames]

        init_data = np.genfromtxt(filenames[0])
        energies = init_data[0:,0] * 1000.
        taus = init_data[0:,1]

        for filename in filenames[1:]:
            this_data = np.genfromtxt(filename)
            taus = np.vstack((taus,this_data[0:,1]))

        return np.array(redshifts),energies,taus.transpose()

    def loadKneiskeTau(self, filename="Tau_Data/kneiske/tau_lower_limit.dat"):

        '''Returns an array of redshifts, an array of energies [GeV] and an
        array of taus where the rows are for the individual redshifts
        listed in the redshift array and the columns are for different
        energies.'''
        
        f = open(filename)
        redshift_string = f.readline()
        redshift_string = f.readline()
        redshift_string = f.readline()
        redshift_string = f.readline()
        redshift_string = f.readline()
        redshift_string = f.readline()
        redshifts = np.fromstring(redshift_string[14:-2], dtype=float, sep="  ")
        f.close()
        
        all_data = np.genfromtxt(filename)
        energies = all_data[0:,0]
        taus = all_data[0:,1:]

        return redshifts, 10**energies, taus


    def loadDominguezTau(self,filename="Tau_Data/dominguez/tau_dominguez11.out"):

        '''Returns an array of redshifts, an array of energies [GeV] and an
        array of taus where the rows are for the individual redshifts
        listed in the redshift array and the columns are for different
        energies.'''

        f = open(filename)
        redshift_string = f.readline()
        redshift_string = f.readline()
        redshift_string = redshift_string[redshift_string.find('[')+1:redshift_string.find(']')]
        f.close()
        redshifts = np.fromstring(redshift_string,dtype=float,sep=",")
    
        all_data = np.genfromtxt(filename)

        energies = all_data[0:,0] * 1000.
        taus = all_data[0:,1:]

        return redshifts,energies,taus

    def loadNullTau(self):

        '''Returns an array of zeros useful for galactic sources'''

        redshifts = np.arange(0,10,0.1)
        energies = np.arange(1,100000, 1000)
        taus = np.zeros( (energies.size, redshifts.size) )

        return redshifts, energies, taus

    def loadSimpleTau(self,filename="Tau_Data/simple/simple.dat"):
    
        '''Returns an array of redshifts, an array of energies [GeV] and an
        array of taus where the rows are for the individual redshifts
        listed in the redshift array and the columns are for different
        energies.'''

        f = open(filename)
        redshift_string = f.readline()
        redshift_string = f.readline()
        redshift_string = redshift_string[redshift_string.find('[')+1:redshift_string.find(']')]
        f.close()
        redshifts = np.fromstring(redshift_string,dtype=float,sep=",")
    
        all_data = np.genfromtxt(filename)

        energies = all_data[0:,0]
        taus = all_data[0:,1:]

        return redshifts,energies,taus

    def interpolateTaus(self,redshift, E, z_array, e_array, tau_array):

        '''Calculates a tau value for specific energies [GeV] and redshift
        given the energy, redshift and tau tables loaded.'''

        if(redshift <= z_array[0]):
            self.logger.warn( "Using minimum redshift in EBL table.")
        if(redshift >= z_array[-1]):
            self.logger.warn( "Using maximum redshift in EBL table.")
        if(E[0] <= e_array[0]):
            self.logger.warn( "Spectrum starts below minimum energy in EBL table.")
        if(E[-1] >= e_array[-1]):
            self.logger.warn("Spectrum continues beyond maximum energy in EBL table.")
    
        sp = interpolate.RectBivariateSpline(e_array,z_array,tau_array, kx=1,ky=1, s=0)
            
        return sp(E,redshift)

    def absorbSpectrum(self,dNdE,taus):

        return (dNdE*np.exp(-1.0*taus))

