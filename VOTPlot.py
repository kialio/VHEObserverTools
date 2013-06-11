#!/usr/bin/env python

from matplotlib import pyplot

def PlotSpectrum(spectrum, ESquared = True):
    
    EBins = spectrum.EBins
    dNdE = spectrum.dNdE

    fig1 = pyplot.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_xlabel('Energy [GeV]')
    if(ESquared):
        ax1.set_ylabel(r'E$^{2}$ dN/dE [GeV cm$^{-2}$ s$^{-1}$]')
        ax1.loglog(EBins, EBins**2*dNdE)
    else:
        ax1.set_ylabel(r'dN/dE [cm$^{-2}$ s$^{-1}$ GeV$^{-1}$]')
        ax1.loglog(EBins, dNdE)

def Plot(spectrum, response, plotTau = True):

    EBins = spectrum.EBins 
    dNdE = spectrum.dNdE
    dNdE_absorbed = spectrum.dNdE_absorbed
    interpolated_taus = spectrum.interpolated_taus   
    EASummary = response.EASummary
    EACurve = response.EACurve
   

    fig1 = pyplot.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_ylabel(r'Differential Flux [cm$^{-2}$ s$^{-1}$ GeV$^{-1}$]')
    ax1.set_xlabel('Energy [GeV]')
    ax1.loglog(EBins, dNdE)
    ax1.set_ylim([dNdE[-1]*0.1,dNdE[0]*1.1])
    ax1.loglog(EBins, dNdE_absorbed)
    ax2 = ax1.twinx()
    if(plotTau):
        ax2.loglog(EBins, interpolated_taus[0:,0], 'r')
    ax2.set_ylabel('tau')
        
    fig2 = pyplot.figure()
    fig2ax1 = fig2.add_subplot(111)
    fig2ax1.set_ylabel(r'Differential Flux [cm$^{-2}$ s$^{-1}$ GeV$^{-1}$]')
    fig2ax1.set_xlabel('Energy [GeV]')
    fig2ax1.loglog(EBins, dNdE)
    fig2ax1.loglog(EBins, dNdE_absorbed)
    fig2ax1.set_ylim([dNdE[-1]*0.1,dNdE[0]*1.1])
    fig2ax1.set_xlim([1,1000])
    fig2ax2 = fig2ax1.twinx()
    fig2ax2.loglog(10**EACurve[0:,0],EACurve[0:,1], 'r')
    fig2ax2.set_ylabel(r'Effective Area [cm$^2$]')
    fig2ax2.axvline(10**EASummary['minSafeE'])
    fig2ax2.axvline(10**EASummary['maxSafeE'])
