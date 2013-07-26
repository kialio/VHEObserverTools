#!/usr/bin/env python

import numpy as np
import json

from VOTUtils import initLogger
from VOTSpectrum import VOTSpectrum
from VOTResponse import VOTResponse

class VOT:
    
    def __init__(self, source = "custom", input = "Plain", output = "Plain", 
                 jsonString = '[]', dataOut = False, sparseData = False, **pars):

        '''
        There are several ways to run this code.  You can provide
        execute it by providing it one of the built in sources to see
        what an example output looks like.  Availalble built-ins
        include 'PG1553', 'Crab', 'PKS2233' and '4C5517' (you must
        give those exact strings.  There are many options surrounding
        each of the custom sources (for example, you can pass a
        different redshift or average zenith angle).  You can also
        define a custom source by passing 'custom' to the source
        variable.  You'll then need to define the full spectral shape.
        See '--help custom' for more details.

        '''

        self.logger = initLogger('VOT')

        if (input == "JSON"):
            jsonObject = json.loads(jsonString)
            pars = jsonObject[0]
            source = pars['source']
        
        sources = {"PG1553" : self.calcPG1553,
                   "Crab" : self.calcCrab,
                   "PKS2233" : self.calcPKS2233,
                   "4C5517" : self.calc4C5517,
                   "custom" : self.calculateRateAndTime}
        
        sources[source](**pars)
        self.Print(output, dataOut, sparseData)
        
    def calculateRateAndTime(self, eMin = 0.1, eMax = 100000, Nbins = 10000, 
                             redshift = 0.1, eblModel = "Dominguez", 
                             instrument = "VERITAS", zenith = 20,
                             spectralModel="PowerLaw", **spectralPars):

        '''                        - custom source -

        This is the main way to calculates the rate in a VHE
        instrument based on an input spectral function.

        Parameter Definitions

        * eMin: minimum energy for the spectral calculations (GeV)
        * eMax: maximum energy for the spectral calculations (GeV) 
        * Nbins: number of bins in the spectral calculations
        * redshift: redshift of the object
        * eblModel: [Dominguez, Simple, NULL].  Use 'NULL' if you 
                    don't want any EBL absorption.  Try '--help EBL' 
                    for more details.
        * instrument: [VERITAS].  Try '--help instrument' for more 
                      details.
        * zenith: average zenith angle of the observations (degrees)
        * spectralModel: spectral shape of the source.  See below 
                         for more details.
        * spectralPars: parameters of the spectral shape.  See 
                        below for more details

        Details              

        eMin/eMax: Note that the underlying algorithms use the safe
        energies of the experiments for the actual rate and time
        calculation and not these values.  These are just for the
        underlying spectral calculations.

        spectralModel: can choose between many of the 2FGL spectral
        models along with some others.  Current options inlclude
        [PowerLaw, PowerLaw2, BrokenPowerLaw, LogParabola,
        HESSExpCutoff].  Try '--help spectrum' for more details.

        spectralPars: the parameters of the model.  In general the
        variable names match those as in the 2FGL models found on the
        FSSC website.  See the underlying function VOTSpectrum.VOT for
        more details.  All energy units should be in 'GeV' and
        differential fluxes in 's^-1 cm^-2 GeV^-1'.  Try '--help
        spectrum' for more details.  If running from the command line
        you'll need to supply all of the spectral parameters direclty
        there.

        '''

        self.VS = VOTSpectrum([],[],eMin, eMax, Nbins, eblModel,redshift,spectralModel,**spectralPars)

        self.VR = VOTResponse(instrument,zenith=zenith, azimuth=0, noise=4.07)
        if(not self.VR.EASummary):
            return
        self.EACurve_interpolated = self.VR.interpolateEA(self.VS.EBins, self.VR.EACurve)
        self.rate = self.VR.convolveSpectrum(self.VS.EBins, 
                                             self.VS.dNdE_absorbed, 
                                             self.EACurve_interpolated,
                                             10**self.VR.EASummary['minSafeE'], 
                                             10**self.VR.EASummary['maxSafeE'])
         

    def calcPG1553(self, eMin = 0.1, eMax = 100000, Nbins = 10000,
                   eblModel="Dominguez", instrument = "VERITAS", zenith=20, **pars):

        '''This should return about 100 gammas/hour if you use the
        defaults.'''

        self.calculateRateAndTime(eMin, eMax, Nbins, 0.5, eblModel, instrument, zenith, 
                                  spectralModel="PowerLaw", N0 = 2.6e-9, 
                                  index=-1.66533, E0 = 2.4)



    def calcCrab(self, eMin = 0.1, eMax = 100000, Nbins = 10000,
                   eblModel="Null", instrument = "VERITAS", zenith=20, **pars):

        '''This should give out on the order of 660 gammas/hour if you use the
        defaults.'''

        self.calculateRateAndTime(eMin, eMax, Nbins,0.0, eblModel, instrument, zenith, 
                                  spectralModel="HESSExpCutoff", N0 = 3.76e-14, 
                                  index=-2.39, E0 = 1000., EC = 14000.)


    def calcPKS2233(self, eMin = 0.1, eMax = 100000, Nbins=10000,
                    eblModel="Dominguez", instrument = "VERITAS", zenith=40, **pars):

        '''This should give out on the order of 0.98 gammas/hour if you use
        the defaults.'''

        self.calculateRateAndTime(eMin, eMax, Nbins, 0.325, eblModel, instrument, zenith, 
                                  spectralModel="PowerLaw2", N = 3.8e-9, 
                                  index=-2.23955, E1 = 1.0, E2 = 100.)

    def calc4C5517(self, eMin = 0.1, eMax = 100000, Nbins=10000,
                   eblModel="Dominguez", instrument = "VERITAS", zenith=20, **pars):

        self.calculateRateAndTime(eMin, eMax, Nbins, 0.8955, eblModel, instrument, 
                                  zenith, spectralModel="LogParabola", N0 = 1.359e-11, 
                                  alpha=-1.8772, beta=-0.067012, Eb = 0.9085)
        
    def Print(self, style = "Plain", dataOut = False, sparseData = False):
        
        Emin =  10**self.VR.EASummary['minSafeE']
        Emax =  10**self.VR.EASummary['maxSafeE']
        crabFlux = 100*self.rate*60./self.VR.crabRate
        detTime = np.interp([crabFlux*0.01], self.VR.SensCurve[:,0], self.VR.SensCurve[:,1])

        if(style == "Plain"):

            self.logger.info("Using " + self.VR.EAFile)
            self.logger.info("Using " + self.VR.EATable)

            self.logger.info("Safe energy range: {:,.2f} to {:,.2f} GeV".format(Emin, Emax))

            self.logger.info("dNdE at 1 GeV: {:,.2e} s^-1 cm^-2 GeV^-1".format(np.interp(1., 
                                                                                         self.VS.EBins, 
                                                                                         self.VS.dNdE)))
            self.logger.info("dNdE at 400 GeV: {:,.2e} s^-1 cm^-2 GeV^-1".format(np.interp(400., 
                                                                                           self.VS.EBins, 
                                                                                           self.VS.dNdE)))
            self.logger.info("dNdE at 1 TeV: {:,.2e} s^-1 cm^-2 GeV^-1".format(np.interp(1000., 
                                                                                         self.VS.EBins, 
                                                                                         self.VS.dNdE)))
            self.logger.info("tau at min safe E: {:,.2f}".format(np.interp(Emin, 
                                                                           self.VS.EBins, 
                                                                           self.VS.interpolated_taus[0:,0])))
            self.logger.info("tau at max safe E: {:,.2f}".format(np.interp(Emax, 
                                                                           self.VS.EBins, 
                                                                           self.VS.interpolated_taus[0:,0])))
            
            self.logger.info("Predicted counts/hour: {:,.2e}".format(self.rate*60.))
    

            self.logger.info("This is approximately {:,.4f}% of the Crab Nebula's Flux".format(crabFlux))


            self.logger.info("This will take approximately {:,.4f} hours to detect at a 5 sigma level".format(detTime[0]))


        if(style == "JSON"):
            
            if dataOut:
            
                spectrum = np.dstack((self.VS.EBins, self.VS.dNdE))[0].tolist()
                spectrum_abs = np.dstack((self.VS.EBins, self.VS.dNdE_absorbed))[0].tolist()
                tau = np.dstack((self.VS.EBins, self.VS.interpolated_taus[0:,0]))[0].tolist() 
                ea = np.dstack((self.VS.EBins, self.EACurve_interpolated))[0].tolist()
                
                if sparseData:
                    span = 10
                else:
                    span = 1

                print json.dumps([{"EAFile" : { "unit": "file name", "value": self.VR.EAFile},
                                   "EATable": {"unit": "table name", "value": self.VR.EATable}, 
                                   "Emin": {"unit": "GeV", "value": Emin}, 
                                   "Emax": {"unit": "GeV", "value": Emax}, 
                                   "Rate": {"unit": "counts/hour",  "value": self.rate*60.},
                                   "Crab": {"unit": "% Crab", "value": crabFlux},
                                   "DetTime": {"unit": "Hours", "value":detTime}},
                                  {"name" : "Spectrum",
                                   "xaxis" : "E (GeV)",
                                   "yaxis" : "dNdE (s^-1 cm^-2 GeV^-1)", 
                                   "data": spectrum[::span]},
                                  {"name" : "Absorbed Spectrum",
                                   "xaxis" : "E (GeV)",
                                   "yaxis" : "dNdE (s^-1 cm^-2 GeV^-1)", 
                                   "data": spectrum_abs[::span]},
                                  {"name" : "Tau",
                                   "xaxis" : "E (GeV)",
                                   "yaxis" : "Tau (Arb.)", 
                                   "data": tau[::span]},
                                  {"name" : "Effective Area",
                                   "xaxis" : "E (GeV)",
                                   "yaxis" : "Area (cm^2)", 
                                   "data": ea[::span]},


                                  ])

            else:
                print json.dumps([{"EAFile" : { "unit": "file name", "value": self.VR.EAFile},
                                   "EATable": {"unit": "table name", "value": self.VR.EATable}, 
                                   "Emin": {"unit": "GeV", "value": Emin}, 
                                   "Emax": {"unit": "GeV", "value": Emax}, 
                                   "Rate": {"unit": "counts/hour",  "value": self.rate*60.},
                                   "Crab": {"unit": "% Crab", "value": crabFlux},
                                   "DetTime": {"unit": "Hours", "value":detTime[0]}},
                                  ])


def printCLIHelp(**opts):

    import os
    import sys

    """This function prints out the help for the CLI."""

    if(opts):
        if(opts["subhelp"]):
            if(opts["subhelp"] == "spectrum"):
                print VOTSpectrum.Spectrum.__doc__
            elif(opts["subhelp"] == "EBL"):
                print VOTSpectrum.helpEBL.__doc__
            elif(opts["subhelp"] == "instrument"):
                print VOTResponse.loadEA.__doc__
            elif(opts["subhelp"] == "custom"):
                print VOT.calculateRateAndTime.__doc__
            elif(opts["subhelp"] == "sources"):
                print VOT.__init__.__doc__
            else:
                print "Unknown help option"
                printCLIHelp()

    else:
        cmd = os.path.basename(sys.argv[0])
        print """                                   
                            - VHEObserversTools - 

Calculate rates in a VHE detector given an input source function, a
given EBL model and redshift and VHE effective areas.

%s (-h|--help) ... This help text.  You can also get help for specific
 things by typing '--help spectrum', '--help EBL', '--help
 instrument', '--help sources' or '--help custom'.

%s (--source = <source>) ... <source> can be a built-in source
 ('Crab', 'PG1153', '4C5517' or 'PKS2233') or custom (user-inputed
 spectrum).  See '--help sources' or '--help custom' for more details.

%s (--input = <input>) ... <input> can be 'Plain' (default) or 'JSON'.
 If the input is 'JSON' you must supply a json string via the
 'jsonInput' option.

%s (--output = <output>) ... <output> can be 'Plain' (default) or
 'JSON'.  Prints the result out with plain descriptive text or as a
 json string.  The json string also includes all of the data arrays.

%s (--jsonInput = <jsonString>) ... <jsonString> is a well formatted
 json string used for json input.  As an example, this is a json
 string for a custom source which could be used as input:

 '[{"source":"custom", "eMin":0.1, "eMax":100000, "Nbins":10000,
 "redshift":0.8955, "eblModel":"Dominguez","instrument":"VERITAS",
 "zenith":20, "spectralModel":"LogParabola", "N0":1.359e-11,
 "alpha":-1.8772, "beta":-0.067012, "Eb":0.9085}]'

%s (--dataOut) ... If this is present, the various arrays used for
 calculations are outputted in the json string as three arrays.  One
 is the spectrum, one is the absorbed spectrum, one are the tau values
 and the last is the effective area curve.  Each json element contains
 a 'name' element (the name of the array), an 'xaxis' element (the
 name and units of the x axis variable), a 'yaxis' element (the name
 and units of the y axis variabl) and a 'data' element (the data as
 x,y pairs).

%s (--sparseData) ... If this is present, the outputted data arrays
  will be sparse (every 10th element).  

""" %(cmd, cmd, cmd, cmd, cmd, cmd, cmd )


def cli():

    import getopt
    import sys

    """Command line interface.  Call this without any options for usage
    notes."""

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'h', ['help=',
                                                       'source=',
                                                       'input=',
                                                       'output=',
                                                       'jsonInput=',
                                                       'dataOut',
                                                       'sparseData', 
                                                       ])
        source = "custom"
        input = "Plain"
        output = "Plain"
        jsonString = '[]'
        dataOut = False
        sparseData = False
        
        for opt, val in opts:
            if opt in ('-h'):
                printCLIHelp()
            if opt in ('--help'):
                printCLIHelp(subhelp=val)
                return
            if opt in ('--source'):
                source = val
            if opt in ('--input'):
                input = val
            if opt in ('--output'):
                output = val
            if opt in ('--jsonInput'):
                jsonString = val
            if opt in ('--dataOut'):
                dataOut = True
            if opt in ('--sparseData'):
                sparseData = True

        if not opts: raise getopt.GetoptError("Must specify an option, printing help.")

        VOT(source, input, output, jsonString, dataOut, sparseData)

    
    except getopt.error as e:
        print "Command Line Error: " + e.msg
        printCLIHelp()
            

if __name__ == '__main__': cli()



