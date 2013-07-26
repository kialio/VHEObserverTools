#!/usr/bin/env python

import numpy as np

from VOTUtils import initLogger, checkParams

'''This reads in a VHE IRF and calculates the response to a inputted
spectrum.  This can be from the extrapolatedSpectrum class or you can
input a file with energy and dNdE as colums.  Energy is in GeV and
dNdE is in photons*cm^-2*GeV^-1.'''

class VOTResponse:

    def __init__(self,instrument="VERITAS", **pars):
        
        self.logger = initLogger('VOTSpectrum')
        self.loadEA(instrument, **pars)
        
        
    def loadEA(self, instrument = "VERITAS", **pars):

        '''                    - Instruments - 
        
        These are the currently avaialble instruments.

        'VERITAS': this is an array of IACTs located in Southern
        Arizona and sensitive from about 100 GeV to tens of TeV.  You
        can choose a variety of zenith angles (0 is pointing straight
        up) and noise levels (ie. night sky background), and expected
        source spectrum.  The default noise level is adequate for most
        fields and the default expected source spectrum (-4) is
        adequate for most sources at VHE.

        'HESS': This is an array of IACTs located in Nambia and
        sensitive from about 100 GeV to tens of TeV.  You can choose
        three zenith angles (20, 45 and 60).  The effective areas are
        from A&A 457, 899 (2006), Figure 13.  The threshold is set to
        150 GeV.

        'HAWC': This is a water Cherenkov experiment located in
        Mexico.  The effective areas are from APh 35, 6412 (2012),
        Figure 2.  You can choose from a high trigger rate (nHit > 70)
        or a low trigger rate (nHit > 30).  You can also choose from
        four zenith angle rages.

        'CTA': this is a future IACT observatory that will be an order
        of magnitude more sensitive then the current generation.  The
        Effective Area curve is from "Monte Carlo design studies for
        the Cherenkov Telescope Array", Bernlohr et
        al. arXiv:1210.3503, Figure 15 and assume the baseline/MPIK
        curve.  Note that this is for demonstration purposes only and
        there are no modifiers for zenith angle, noise level or
        anything else.  The safe energy range is hardcoded from 100
        GeV to 100 TeV and the threshold is set to 100 GeV.

        '''


        EAParamTemplates = {"VERITAS" : {"zenith":20, "azimuth":0, "noise":4.07},
                            "CTA" : {"zenith":0, "azimuth":0, "noise":0},
                            "HESS" : {"zenith":20, "azimuth":0, "noise":0},
                            "HESS2" : {"zenith":0, "azimuth":0, "noise":0},
                            "HAWC" : {"zenith":20}}
        EffectiveAreas = {"VERITAS": self.loadVERITASEA,
                          "CTA": self.loadCTAEA,
                          "HESS": self.loadHESSEA,
                          "HESS2": self.loadHESS2EA,
                          "HAWC": self.loadHAWCEA}

        try:
            checkParams(EAParamTemplates[instrument],pars, self.logger)
        except KeyError:
            self.logger.critical("Unsupported instrument.")
            return

        try:
            self.EASummary,self.EACurve,self.SensCurve,self.EAFile,self.EATable,self.crabRate = EffectiveAreas[instrument](**pars)
        except KeyError:
            self.logger.critical("Unsupported instrument.")
            return
    
    def interpolateEA(self, E, EA):
        
        '''Calculates EA for specific energies [GeV]'''

        return np.interp(E, 10**EA[0:,0], EA[0:,1])

    def convolveSpectrum(self, EnergyBins, dNdE, EA, minE, maxE):
        
        '''Integrates a spectrum over an effective area.  dNdE and EA must
        have the same energy axis (ie. the same x axis points). minE
        and maxE are assumed to be inside the limits of E.  '''

        if(EnergyBins[0] > minE or EnergyBins[-1] <maxE):
            self.logger.warn("Not integrating over full VHE response")

        counts_per_second = 0
        it = np.nditer(EnergyBins, flags=['f_index'])
        while not it.finished:
            if(it[0] >= minE and it[0] <= maxE):
                counts_per_second = counts_per_second + (EnergyBins[it.index] - EnergyBins[it.index-1])*dNdE[it.index]*EA[it.index]
            it.iternext()

        return counts_per_second*60.

    def loadSensitivity(self, instrument='VERITAS'):

        '''This loads a sensitivity curve for a specific instrument.  Note
        that I don't have a sensitivity curve for CTA yet.'''

        if instrument == 'CTA':
            self.logger.warning("Assuming 10 times worse sensitivity than HESS for CTA!")
            filename = "Effective_Areas/HESS/sensitivity.csv"
            sensitivity = np.genfromtxt(filename,delimiter=",")
            sensitivity[:,1] = 0.1*sensitivity[:,1]
            return sensitivity
        elif instrument == 'HAWC':
            self.logger.warning("Assuming 1000 times worse sensitivity than HESS for HAWC!")
            filename = "Effective_Areas/HESS/sensitivity.csv"
            sensitivity = np.genfromtxt(filename,delimiter=",")
            sensitivity[:,1] = 1000.*sensitivity[:,1]
            return sensitivity
        elif instrument == 'HESS2':
            self.logger.warning("Assuming HESS sensitivity but using HESS2 effective areas!")
            filename = "Effective_Areas/HESS/sensitivity.csv"
            return np.genfromtxt(filename,delimiter=",")
        else:
            self.logger.warning("Time to detection assumes a Crab Nebula Spectrum.")
            filename = "Effective_Areas/"+instrument+"/sensitivity.csv"
            return np.genfromtxt(filename,delimiter=",")

    def loadVERITASEA(self, cuts = 'soft', **pars):

        '''This loads in the effective areas for VERITAS.  All units returned
        in GeV and cm^2.'''

        name = {'soft': "Effective_Areas/VERITAS/ea_Nov2010_na_ATM21_vegasv240rc1_7sam_050off_soft-1",
                'medium': "Effective_Areas/VERITAS/ea_Nov2010_na_ATM21_vegasv240rc1_7sam_050off_med-1",
                'hard': "Effective_Areas/VERITAS/ea_Nov2010_na_ATM21_vegasv240rc1_7sam_050off_hard-1"}


        crabRate = 659.0
        self.logger.warning("Using {} counts/hr as the rate from the Crab Nebula.".format(crabRate))

        filename = name[cuts] + ".summary.csv"
        summary_data  = np.genfromtxt(filename, delimiter=",", unpack=True,
                                      dtype=[('eanames', 'S50'),
                                             ('minSafeE', 'float'),
                                             ('maxSafeE', 'float'),
                                             ('peakArea', 'float'),
                                             ('threshold', 'float')])

        EAName = "EffectiveArea_Azimuth_" + str(pars["azimuth"]) + "_Zenith_" + str(pars["zenith"]) + "_Noise_" + str(pars["noise"])
        array_mask = summary_data['eanames'] == EAName

        try:
            EASummary = {'eaname': EAName,
                         'minSafeE': ((summary_data['minSafeE'])[array_mask])[0] + 3,
                         'maxSafeE': ((summary_data['maxSafeE'])[array_mask])[0] + 3,
                         'peakArea': ((summary_data['peakArea'])[array_mask])[0] * 10000.,
                         'threshold':((summary_data['threshold'])[array_mask])[0] * 1000.}
        except IndexError:
            self.logger.critical('Could not find that EA curve.')
            return 0,0
                
        EACurveFileName = name[cuts]+"/"+EAName+".csv"
        EACurve_data = np.genfromtxt(EACurveFileName,delimiter=",")
        
        EACurve_data = EACurve_data + [3.,0.]
        EACurve_data = EACurve_data * [1.,10000.]

        Sensitivity_data = self.loadSensitivity('VERITAS')

        return EASummary, EACurve_data, Sensitivity_data, filename, EAName, crabRate


    def loadHESSEA(self, zenith = 20, **pars):
        
        '''This loads in the effective areas for HESS.  All units returned in
        GeV and cm^2.  There are three zenith angles (20, 45 and 60)
        which each have their own safe energy range.'''
        
        crabRate = 1040.0
        self.logger.warning("Using {} counts/hr as the rate from the Crab Nebula.".format(crabRate))

        minEnergies = {"20": 0.174399764328,
                       "45": 0.395019338585,
                       "60": 1.14280205981}

        maxEnergies = {"20": 25.0,
                       "45": 45.0,
                       "60": 70.0}


        try:
            minE = minEnergies[str(zenith)]
            maxE = maxEnergies[str(zenith)]
        except IndexError:
            self.logger.critical('Could not find that EA curve.')
            return 0,0
        
        EAName = "EA_AA457_Fig13_True_Zenith_"+str(zenith)
        
        EASummary = {'eaname': EAName,
                     'minSafeE': np.log10(minE*1000.),
                     'maxSafeE': np.log10(maxE*1000.),
                     'peakArea': 0,
                     'threshold': 150.}

        EACurveFileName = "Effective_Areas/HESS/" + EAName + ".csv"
        EACurve_data = np.genfromtxt(EACurveFileName,delimiter=",")
        
        EASummary['peakArea'] = np.max(EACurve_data[:,0]*10000.)


        EACurve_data[:,0] = np.log10(EACurve_data[:,0]*1000.)
        EACurve_data = EACurve_data * [1.,10000.]

        Sensitivity_data = self.loadSensitivity('HESS')

        return EASummary, EACurve_data, Sensitivity_data, EACurveFileName, EAName, crabRate

    def loadHESS2EA(self, analysisChain=2,**pars):
        
        '''This loads in the effective areas for HESS2 (mono).  All units returned in
        GeV and cm^2.  There is only one (unknown) zenith.'''
        
        crabRate = 2150.0
        self.logger.warning('This EA is for demonstration purposes only.  All paramters (zenith etc.) are ignored.')
        self.logger.warning("Using {} counts/hr as the rate from the Crab Nebula.".format(crabRate))

        if(analysisChain == 2):
            EAName = 'EA_1307.6003v1_Fig2_AC2'
        else:
            EAName = 'EA_1307.6003v1_Fig2_AC1'

        EACurveFileName="Effective_Areas/HESS2/"+EAName+".csv"

        EASummary = {'eaname': EAName,
                     'minSafeE': np.log10(50.),
                     'maxSafeE': np.log10(990.),
                     'peakArea': 92000*10000.,
                     'threshold': 50.}
        
        EACurve_data = np.genfromtxt(EACurveFileName,delimiter=",")
        
        EACurve_data[:,0] = EACurve_data[:,0] + 3.0
        EACurve_data = EACurve_data * [1.,10000.]

        Sensitivity_data = self.loadSensitivity('HESS2')

        return EASummary, EACurve_data, Sensitivity_data, EACurveFileName, "EA_1307.6003v1_Fig2_AC2", crabRate


    def loadCTAEA(self, EACurveFileName="Effective_Areas/CTA/EA_1210.3503_Fig15_MPIK.csv", **pars):

        '''This loads in the effective areas for CTA.  All units returned in
        GeV and cm^2.  Note that this is for demonstration purposes
        only and there are no modifiers for zenith angle, noise level
        or anything else.  The safe energy range is hardcoded from 100
        GeV to 100 TeV and the threshold is set to 100 GeV.'''

        crabRate = 1760.0

        self.logger.warning('This EA is for demonstration purposes only.  All paramters (zenith etc.) are ignored.')
        self.logger.warning("Using {} counts/hr as the rate from the Crab Nebula.".format(crabRate))

        EASummary = {'eaname': 'EA_1210.3503_Fig15_MPIK',
                     'minSafeE': 2.0,
                     'maxSafeE': 5.0,
                     'peakArea': 3.37550925e10,
                     'threshold': 100.}

        EACurve_data = np.genfromtxt(EACurveFileName,delimiter=",")
        
        EACurve_data[:,0] = np.log10(EACurve_data[:,0]*1000.)
        EACurve_data = EACurve_data * [1.,10000.]

        Sensitivity_data = self.loadSensitivity('CTA')

        return EASummary, EACurve_data, Sensitivity_data, EACurveFileName, "EA_1210.3503_Fig15_MPIK", crabRate
        

    def loadHAWCEA(self, zenith = 20, trigger_rate = "high", **pars):

        '''This loads the effective area for HAWC.  All units returned in GeV
        and cm^2.'''

        crabRate = 19.

        theta_bins = ['0607','0708','0809','0910']
        theta_bounds = [0.6,0.7,0.8,0.9,1.0]

        if trigger_rate == 'high':
            nHit = '70'
            minEnergies = {"0607": 200.,
                           "0708": 100.,
                           "0809": 63.1,
                           "0910": 31.6}
        else:
            nHit = '30'
            minEnergies = {"0607": 100.,
                           "0708": 50.,
                           "0809": 25.,
                           "0910": 15.}

        cos_zenith = np.cos(zenith*np.pi/180)

        if cos_zenith <= 0.6: 
            theta_bin = theta_bins[0]
        else:
            theta_bin = theta_bins[np.searchsorted(theta_bounds,cos_zenith)-2]
        
        EAName = "EA_APh35_Fig2_nHit_"+nHit+"_cosTh_"+theta_bin

        EACurveFileName = "Effective_Areas/HAWC/" + EAName + ".csv"
        
        EASummary = {'eaname': EAName,
                     'minSafeE': np.log10(minEnergies[theta_bin]),
                     'maxSafeE': np.log10(99000.),
                     'peakArea': 1.0,
                     'threshold': 100.}

        EACurve_data = np.genfromtxt(EACurveFileName,delimiter=",")
        
        #EACurve_data[:,0] = np.log10(EACurve_data[:,0])+3.0
        EACurve_data = EACurve_data * [1.,10000.]


        Sensitivity_data = self.loadSensitivity('HAWC')

        return EASummary, EACurve_data, Sensitivity_data, EACurveFileName, EAName, crabRate
