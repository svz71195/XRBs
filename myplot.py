
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mclr
from pynbody.filt import *


def extract_photon_data(filename,Emin,Emax,cr=0.,cd=0.,zv=1.,POS=False,SKY=False,restframe=False,redshift=0.):
    from astropy.io import fits
    
    print("...    Extract photon data")
    with fits.open(filename) as hdul:
        if ( ("PHLIST" in [hdul[i].name for i in range(len(hdul))]) or ("PHOTON_LIST" in [hdul[i].name for i in range(len(hdul))]) or ('' == hdul[1].name) )== False:
            print("No photon list found. Check manually\n")
            hdul.info()
            exit()
        for i in range(len(hdul)):
            if ("PHLIST" in hdul[i].name) or ("PHOTON_LIST" in hdul[i].name) or ('' == hdul[i].name):
                for f in hdul[i].columns.names:
                    #print(i, f, hdul[i].columns.names)
                    if ("ENERGY" in f) or ("PHE" in f):
                        _phE = hdul[i].data[f]
                        pre_band = (_phE >= Emin) & (_phE <= Emax)
                        if restframe:
                            try:
                                _phE = _phE*(1+hdul[0].header['REDSHIFT'])
                            except:
                                #print("No 'REDSHIFT' card in header of 'PrimaryHDU'")
                                _phE = _phE*(1+redshift)
                    if (("RA" or "RIGHTASCENSION") in f) and SKY:
                        _ra = hdul[i].data[f]
                        pre_sel1 = ( _ra >= (cr-zv/2.) ) & ( _ra <= (cr+zv/2.) )
                    if (("DEC" or "DECLINATION") in f) and SKY:
                        _dec = hdul[i].data[f]
                        pre_sel2 = (_dec >= (cd-zv/2.) ) & (_dec <= (cd+zv/2.) )
                    if ( ("XPOS" in f) or ('POS_X' in f) ) and POS:
                        _xpos = hdul[i].data[f]
                        # convert to RA with ang.distance D_A(z)
                    if ( ("YPOS" in f) or ('POS_X' in f) ) and POS:
                        _ypos = hdul[i].data[f]
                        # Convert to DEC
                    #if "ZPOS" in f and POS:
                    #    _zpos = hdul[i].data[f]
                try:
                    _phE
                except:
                    print(f,"No columns containing energy found. Check manually\n")
                    print(repr(hdul[i].header))
                    exit()

    if SKY:
        select = pre_band & pre_sel1 & pre_sel2
        #return np.array(_phE[select], dtype=float), np.array(_ra[select],dtype=float), np.array(_dec[select],dtype=float)
        return _phE[select].astype(float), _ra[select].astype(float), _dec[select].astype(float)
    elif POS:
        select = pre_band
        return np.array(_phE[select],dtype=float), np.array(_xpos[select],dtype=float), np.array(_ypos[select],dtype=float) #, np.array(zpos,dtype=float)
    else:
        select = pre_band
        return _phE[select].astype(float)

def cuml_hist(alist,norm=False):
    cuml = np.array([])
    for i,el in enumerate(alist):
        cuml = np.append(cuml,np.sum(alist[:i+1]))
    if norm:
        return cuml / cuml[-1]
    else:
        return cuml
    
def calc_bin_centers(edges):
    return edges[:-1] + np.diff(edges)/2.

def calc_err_margin(arr):
    mean = np.mean(arr)
    temp_u = []
    temp_l = []
    k=0
    j=0
    while(len(temp_u)/len(arr) < 0.34):
        for a in arr:
            if a < (mean*(1 + k*0.01)):
                temp_u.append(a)
        k += 1
    while(len(temp_l)/len(arr) < 0.34):
        for a in arr:
            if a < (mean*(1 - j*0.01)):
                temp_u.append(a)
        j += 1
    r1 = (mean*(1 + k*0.01))
    r2 = (mean*(1 - j*0.01))
    
    return r1, r2 

def median_abs_dev(arr):
    return np.median( np.abs( arr - np.median(arr) ) )

def err_shade(func,val1,val2,err1,err2):
    """
    uses err on val and produces shaded area of functional dependence
    """
    pass

def plot_MLR_Torres2010():
    # mass in Msol
    # radius in Rsol
    # Luminosity in Lsol
    
    with open("Torres2010_MLR.dat") as f:
        Lines = f.readlines()
        
    M       = np.zeros(len(Lines))
    dM      = np.zeros(len(Lines))
    
    R       = np.zeros(len(Lines))
    dR      = np.zeros(len(Lines))
    
    logL    = np.zeros(len(Lines))
    dlogL   = np.zeros(len(Lines))
        
    for i,l in enumerate(Lines):
        M[i]        = l.split()[3]
        dM[i]       = l.split()[4]
        R[i]        = l.split()[5]
        dR[i]       = l.split()[6]
        logL[i]     = l.split()[11]
        dlogL[i]    = l.split()[12]
    
    logM    = np.log10(M)
    logR    = np.log10(R)
    
    mM = (M < 1.)
    MM = (M > 1.)
    
    # log(R/Rsun) = alpha * log(M/Msun) + log(beta)
    aR,covR   = np.polyfit(logM,logR,1,cov=True)
    aRm,covRm = np.polyfit(logM[mM],logR[mM],1,cov=True)
    aRM,covRM = np.polyfit(logM[MM],logR[MM],1,cov=True)
    aL,covL   = np.polyfit(logM,logL,1,cov=True)
    aLm,covLm = np.polyfit(logM[mM],logL[mM],1,cov=True)
    aLM,covLM = np.polyfit(logM[MM],logL[MM],1,cov=True)
    print("M < 1: R ~ M^({:.2f}+-{:.2f})".format(aRm[0],np.sqrt(covRm[0][0])))
    print("M < 1: L ~ M^({:.2f}+-{:.2f})".format(aLm[0],np.sqrt(covLm[0][0])))
    print("M > 1: R ~ M^({:.2f}+-{:.2f})".format(aRM[0],np.sqrt(covRM[0][0])))
    print("M > 1: L ~ M^({:.2f}+-{:.2f})\n".format(aLM[0],np.sqrt(covLM[0][0])))
    plt.plot(logM,logR,lw=0.,marker='o',markersize=2.)
    plt.show()
    plt.plot(logM,logL,lw=0.,marker='o',markersize=2.)
    plt.show()
    return aR, aRm, aRM, aL, aLm, aLM
    
    
def plot_MLR_Eker2015():
    # mass in Msol
    # radius in Rsol
    # Luminosity in Lsol
    
    with open("Eker2015_MLR.dat") as f:
        Lines = f.readlines()
        
    M       = np.zeros(len(Lines))
    dM      = np.zeros(len(Lines))
    
    R       = np.zeros(len(Lines))
    dR      = np.zeros(len(Lines))
    
    logL    = np.zeros(len(Lines))
    dlogL   = np.zeros(len(Lines))
        
    for i,l in enumerate(Lines):
        M[i]        = l.split()[1]
        dM[i]       = l.split()[2]
        R[i]        = l.split()[3]
        dR[i]       = l.split()[4]
        logL[i]     = l.split()[7]
        dlogL[i]    = l.split()[8]
    
    logM    = np.log10(M)
    logR    = np.log10(R)
    
    mM = (M < 1.) & (M>0.1)
    MM = (M > 1.)
    
    # log(R/Rsun) = alpha * log(M/Msun) + log(beta)
    aR,covR   = np.polyfit(logM,logR,1,cov=True)
    aRm,covRm = np.polyfit(logM[mM],logR[mM],1,cov=True)
    aRM,covRM = np.polyfit(logM[MM],logR[MM],1,cov=True)
    aL,covL   = np.polyfit(logM,logL,1,cov=True)
    aLm,covLm = np.polyfit(logM[mM],logL[mM],1,cov=True)
    aLM,covLM = np.polyfit(logM[MM],logL[MM],1,cov=True)
    print("M < 1: R ~ M^({:.2f}+-{:.2f})".format(aRm[0],np.sqrt(covRm[0][0])))
    print("M < 1: L ~ M^({:.2f}+-{:.2f})".format(aLm[0],np.sqrt(covLm[0][0])))
    print("M > 1: R ~ M^({:.2f}+-{:.2f})".format(aRM[0],np.sqrt(covRM[0][0])))
    print("M > 1: L ~ M^({:.2f}+-{:.2f})\n".format(aLM[0],np.sqrt(covLM[0][0])))
    plt.plot(logM,logR,lw=0.,marker='o',markersize=2.)
    plt.show()
    plt.plot(logM,logL,lw=0.,marker='o',markersize=2.)
    plt.show()
    return aR, aRm, aRM, aL, aLm, aLM

def plot_stellar_age():
    
    import g3read as gr
    import cosmo
    groupbase = '/home/lcladm/Studium/Masterarbeit/test/dorc/uhr_test/groups_136/sub_136.0'
    snapbase  = '/home/lcladm/Studium/Masterarbeit/test/dorc/uhr_test/snapdir_136/snap_136'
    
    halo_positions = gr.read_new(groupbase,'GPOS',0)
    halo_r25K      = gr.read_new(groupbase,'R25K',0)
    
    ages = np.array([])
    for ihal,hpos in enumerate(halo_positions[:500]):
        scale = gr.read_particles_in_box(snapbase,hpos,halo_r25K[ihal],"AGE ",4)
        #mass  = gr.read_particles_in_box(snapbase,hpos,halo_r25K[ihal],"MASS",4)
        #imass = gr.read_particles_in_box(snapbase,hpos,halo_r25K[ihal],"iM  ",4)
        #plt.plot(cosmo.age_part(scale),mass/imass,lw=0.,markersize=6.,marker='+',c='k')
        
        ages  = np.append(ages,cosmo.age_part(scale))
        #print(ihal)
    #plt.show()
    
    age_hist, age_edge = np.histogram(ages,np.logspace(0,np.log10(13000),100))
    age_cent = age_edge[:-1] + np.diff(age_edge) / 2.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis([1.,13000.,5,2*np.amax(age_hist)])
    ax.loglog()
    #ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    #ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    #ax.set_xticks([10,100])
    #ax.set_yticks([1,10,50,100])
    #ax.axvline(x=3.,ymin=0.,ymax=10,c='k',lw=1.5,ls='--')
    ax.axvline(x=5.,ymin=0.,ymax=1e6,c='k',lw=.95,ls='--')
    ax.axvline(x=100.,ymin=0.,ymax=1e6,c='k',lw=.95,ls='--')
    ax.axvline(x=1000.,ymin=0.,ymax=1e6,c='k',lw=.95,ls='--')
    #ax.axvline(x=40.,ymin=0.,ymax=10,c='k',lw=1.5,ls='--')
    for j in range(len(age_cent)):
        if age_cent[j] > 5:
            j1 = j
            break
    print(j1)
    for j in range(len(age_cent)):
        if age_cent[j] > 100:
            j2 = j
            break
    print(j2)
    for j in range(len(age_cent)):
        if age_cent[j] > 1000:
            j3 = j
            break
    print(j3)
    ax.step(age_cent,age_hist,where='pre',lw=.95,c='k')
    #ax.step(age_cent[j1:j2+1],age_hist[j1:j2+1],where='pre',lw=.95,c='k')
    ax.step([4.,9.,20.,50.,130.,200.],[0.,10.,50.,65,15,0.],where='pre',lw=.95,c='darkgreen')
    ax.fill_between(age_cent[:j1],age_hist[:j1],0,step='pre',facecolor='r',alpha=0.6)#,label=r'T $\leq 5$Myr') #if filling step function remember to put step kword
    ax.fill_between(age_cent[j1-1:j2+1],age_hist[j1-1:j2+1],0,step='pre',facecolor='b',alpha=0.6)#,label=r'$t_H$')
    ax.fill_between(age_cent[j2:j3+1],age_hist[j2:j3+1],0,step='pre',facecolor='r',alpha=0.6,label=r'ineligible')
    ax.fill_between(age_cent[j3:],age_hist[j3:],0,step='pre',facecolor='b',alpha=0.6,label=r'eligible')
    ax.fill_between([4.,9.,20.,50.,130.,200.],[0.,10.,50.,65,15,0.],0,step='pre',facecolor='darkgreen',alpha=0.7,label=r'$\eta_{\mathrm{HMXB}}\cdot 10^6\, [M_{\odot}^{-1}]$')
    ax.set_xlabel(r'$T$ [Myr]',fontsize=13)
    ax.set_ylabel(r'$N_{*}$',fontsize=13)
    print("number of stars: ", np.sum(age_hist))
    print("fraction in t_H: ", np.sum(age_hist[j1-1:j2+1])/np.sum(age_hist))
    print("fraction in t_L: ", np.sum(age_hist[j3:])/np.sum(age_hist))
    print("fraction in inel.: ", (np.sum(age_hist[:j1])+np.sum(age_hist[j2:j3+1]))/np.sum(age_hist))
    print("fraction < 5 Myr in < 100 Myr: ", np.sum(age_hist[:j1])/np.sum(age_hist[:j2+1]))
    ax.text(15,1.e4,'HMXB',ha='center',va='center',bbox=dict(facecolor='black',alpha=0.5))
    ax.text(3000,1.e6,'LMXB',ha='center',va='center',bbox=dict(facecolor='black',alpha=0.5))
    #ax.text(2.,25.,r'$N={:g}$'.format(np.sum(age_hist[:j+1])),ha='center',va='center',bbox=dict(facecolor='red',alpha=0.5))
    #ax.text(15.,50.,r'$N={:g}$'.format(np.sum(age_hist[j+1:])),ha='center',va='center',bbox=dict(facecolor='blue',alpha=0.5))
    #ax.text(3.,1.,'first BH',rotation=90.,rotation_mode='anchor',bbox=dict(facecolor='grey',alpha=0.5))
    #ax.text(40.,1.,'last NS',rotation=90.,rotation_mode='anchor',bbox=dict(facecolor='grey',alpha=0.5))
    ax.legend(loc='upper left', fontsize=12)#, handlelength=2)
    plt.show()
    
def plot_pyn_gal(ihal,j,u=None):
    
    import g3read as gr
    import cosmo
    import pynbody as pb
    import pynbody
    import pynbody.plot
    import pynbody.plot.sph as sph
    from scipy.stats import gaussian_kde
    from astropy.io import fits
    
    groupbase = '/home/lcladm/Studium/Masterarbeit/test/dorc/uhr_test/groups_136/sub_136.0'
    snapbase  = '/home/lcladm/Studium/Masterarbeit/test/dorc/uhr_test/snapdir_136/snap_136'
    fitsbase  = '/home/lcladm/Studium/Masterarbeit/R333/fits/'
    filenameH = fitsbase+'gal'+str(ihal)+'HXRB.fits'
    filenameL = fitsbase+'gal'+str(ihal)+'LXRB.fits'
    
    halo_positions = gr.read_new(groupbase,'GPOS',0)
    halo_r25K      = gr.read_new(groupbase,'R25K',0)
    halo_rvir      = gr.read_new(groupbase,'RVIR',0)
    halo_FSUB      = gr.read_new(groupbase,'FSUB',0)
    halo_SFR       = gr.read_new(groupbase,'SSFR',1)
    halo_mass_str       = gr.read_new(groupbase,'MSTR',0) * 1e10 / cosmo.h
    mass25K  = halo_mass_str[:,5]
    
    f = pb.load(snapbase)
    #s = pb.load(groupbase)
    #h = s.halos()
    f.physical_units()
    f.properties['boxsize'] = f.properties['boxsize'] / cosmo.h * cosmo.aa_c
    print(f.properties)
    print(f.loadable_keys())
    print(f.gas.loadable_keys())
    print(f.stars.loadable_keys())
    
    with fits.open(filenameL) as fL:
        phEL = fL[1].data['PHOTON_ENERGY']
        xL   = fL[1].data['POS_X']
        yL   = fL[1].data['POS_Y']
        
        m = (phEL < 8.) & (phEL > .5)
        
        phEL = phEL[m]
        xL = xL[m]
        yL = yL[m]
    
    with fits.open(filenameH) as fH:
        phEH = fH[1].data['PHOTON_ENERGY']
        xH   = fH[1].data['POS_X']
        yH   = fH[1].data['POS_Y']
        
        m = (phEH < 8.) & (phEH > .5)
        
        phEH = phEH[m]
        xH = xH[m]
        yH = yH[m]
    
    center = halo_positions[ihal] / cosmo.h * cosmo.aa_c
    radius = halo_r25K[ihal] / cosmo.h * cosmo.aa_c
    print(radius)
    Rvir   = halo_rvir[ihal] / cosmo.h * cosmo.aa_c
    print(center,radius)
    
    print("...   Collecting halo data with pynb...")
    halo_part = f[pynbody.filt.Sphere(str(radius)+' kpc',cen=center)] #load gas from sphere with radius r around center
    halo_part['pos'] = halo_part['pos'] - center
    print(halo_part['pos'][0])
    
    print("...   Start KDE...")
    Nbins = 512
    xedges=np.linspace(-radius/2.,radius/2.,Nbins)
    E, yedges, xedges = np.histogram2d(yH,xH,bins=[xedges,xedges])
    F, yedges, xedges = np.histogram2d(yL,xL,bins=[xedges,xedges])
    #plt.imshow(E,extent=[np.amin(xedges),np.amax(xedges),np.amin(xedges),np.amax(xedges)],origin='lower',norm=mclr.LogNorm(),cmap='Greens')
    #plt.imshow(F,extent=[np.amin(xedges),np.amax(xedges),np.amin(xedges),np.amax(xedges)],origin='lower',norm=mclr.LogNorm(),cmap='Reds')
    #plt.show()
    plt.figure(figsize=(6,6))
    plt.contourf(E,extent=[np.amin(xedges),np.amax(xedges),np.amin(xedges),np.amax(xedges)],origin='lower',cmap='Greens',levels=5,norm=mclr.LogNorm(),vmin=.02,vmax=1000.,aspect='equal')
    plt.contourf(F,extent=[np.amin(xedges),np.amax(xedges),np.amin(xedges),np.amax(xedges)],origin='lower',cmap='Reds',levels=5,norm=mclr.LogNorm(),vmin=.02,vmax=1000.,aspect='equal')
    #plt.colorbar()
    print("FSUB = ", halo_FSUB[ihal], halo_SFR[halo_FSUB[ihal]], mass25K[ihal])
    plt.show()
    #exit()
    #kdeL = gaussian_kde(np.vstack([xL,yL]))
    #kdeH = gaussian_kde(np.vstack([xH,yH]))
    #Xgrid, Ygrid = np.mgrid[min(xpos):max(xpos):Nbins*1j,min(ypos):max(ypos):Nbins*1j]
    #Xgrid, Ygrid = np.mgrid[-radius:radius:Nbins*1j,-radius:radius:Nbins*1j]
    #ZL = kdeL.evaluate(np.vstack([Xgrid.flatten(), Ygrid.flatten()]))
    #ZH = kdeH.evaluate(np.vstack([Xgrid.flatten(), Ygrid.flatten()]))
    
    print("...   Start image process")
    #im = sph.image(halo_part.g,qty=j,units=u,width=str(radius)+' kpc',cmap="Blues",resolution=Nbins,denoise=False,approximate_fast=False,threaded=False)
    im = sph.image(halo_part.s,qty="age",width=str(radius)+' kpc',cmap="cividis",resolution=2*Nbins,denoise=False,approximate_fast=False)
    plt.contour(E,extent=[np.amin(xedges),np.amax(xedges),np.amin(xedges),np.amax(xedges)],origin='lower',cmap='Greens',levels=[15,70,140,300],norm=mclr.LogNorm(),vmin=.02,vmax=1000.)
    plt.contour(F,extent=[np.amin(xedges),np.amax(xedges),np.amin(xedges),np.amax(xedges)],origin='lower',cmap='Reds',levels=[15,70,140,300],norm=mclr.LogNorm(),vmin=.02,vmax=1000.)
    #circle1 = plt.Circle((0, 0), radius, color='k',fill=False,ls='--',lw=1.)
    #plt.gca().add_patch(circle1)
    #plt.contour(Xgrid,Ygrid,ZL.reshape(Xgrid.shape),15,origin='lower',colors='darkred',linewidths=.9)
    #plt.contour(Xgrid,Ygrid,ZH.reshape(Xgrid.shape),15,origin='lower',colors='darkgreen',linewidths=.9)
    #im2 = sph.image(halo_part.s,qty="mass",width=str(radius)+' kpc',cmap="Reds",resolution=500,denoise=True,approximate_fast=False)
    plt.show()
    
    
    
def plot_spec(centers,spec,label="",xl=r'$E_{ph}$ [keV]',yl='norm. counts',use_log=True):
    ce_w = centers[1] - centers[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    if use_log:
        ax.axis([np.amin(centers)-ce_w,np.amax(centers)+ce_w,1.e-4,2*np.amax(spec)])
        ax.loglog()
    else:
        ax.axis([np.amin(centers)-ce_w,np.amax(centers)+ce_w,0.,1.1*np.amax(spec)])
        
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.step(centers,spec,where="mid",lw=0.9,c="k",label=label)
    ax.legend()
    ax.set_xlabel(xl,fontsize=14)
    ax.set_ylabel(yl,fontsize=14)
    plt.tight_layout(pad=0.1)
    plt.show()
    
    
def plot_spec_contr(spec_bins,spec1,spec2=None,spec3=None,l1="",l2="",l3="",pl1=False,pl2=False):
    centers = calc_bin_centers(spec_bins)
    if (type(spec1) != np.ndarray):
        print("... 'myplot.plot_spec_contr()': expected ndarray as spec1 input")
        exit()
    if type(spec2) == np.ndarray:
        if type(spec3) == np.ndarray:
            spec = spec1 + spec2 + spec3
            contr1 = spec1 / spec
            contr2 = spec2 / spec
            contr3 = spec3 / spec
        else:
            spec = spec1 + spec2
            contr1 = spec1 / spec
            contr2 = spec2 / spec
            contr3 = np.zeros(len(spec))
    else:
        spec = spec1
        contr1 = np.ones(len(spec))
        contr2 = np.zeros(len(spec))
        contr3 = np.zeros(len(spec))
        
    ce_w = centers[1] - centers[0]
    
    f = plt.figure()
    axs = []
    gs = f.add_gridspec(nrows=2,ncols=1,hspace=0.01)
    axs.append( f.add_subplot(gs[0,0]) )
    axs.append( f.add_subplot(gs[1,0]) )
    for ax in axs:
        if pl1 or pl2:
            ax.axis([0.3,10.,0.0011,2*np.amax(spec)])
            ax.loglog()
            ax.grid()
        else:
            ax.axis([spec_bins[0],spec_bins[-1],0.0001,2*np.amax(spec)])
            ax.loglog()
            ax.grid()
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        #ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        ax.set_xticks([0.1,0.5,1.0,5.0,10.,20.,50.])
    axs[0].tick_params(labelbottom=False)
    axs[0].step(centers,spec1,where="mid",lw=0.9,c="b",alpha=0.6)#,label=l1)
    axs[0].step(centers,spec2,where="mid",lw=0.9,c="r",alpha=0.6)#,label=l2)
    axs[0].step(centers,spec3,where="mid",lw=0.9,c="g",alpha=0.6)#,label=l3)
    axs[0].step(centers,spec,where="mid",lw=0.9,c="k",label="comb.")
    if pl1:
        
        Capp31 = 1.37*centers**(-1.25)
        Capp1 = [(1.37+0.3)*centers**(-1.25),(1.37-0.3)*centers**(-1.25),(1.37)*centers**(-1.25+0.35),(1.37)*centers**(-1.25-0.35),(1.37+0.3)*centers**(-1.25+0.35),(1.37+0.3)*centers**(-1.25-0.35),(1.37-0.3)*centers**(-1.25+0.35),(1.37-0.3)*centers**(-1.25-0.35)]
        Capp3 = 4.18*centers**(-1.57)
        Capp = [(4.18+0.26)*centers**(-1.57),(4.18-0.26)*centers**(-1.57),(4.18)*centers**(-1.57+0.1),(4.18)*centers**(-1.57-0.1),(4.18+0.26)*centers**(-1.57+0.1),(4.18+0.26)*centers**(-1.57-0.1),(4.18-0.26)*centers**(-1.57+0.1),(4.18-0.26)*centers**(-1.57-0.1)]
        CappU1 = []
        CappU = []
        CappL1 = []
        CappL = []
        for l in range(len(Capp[0])):
            CappU.append( np.amax( [z[l] for z in Capp] ) )
            CappL.append( np.amin( [z[l] for z in Capp] ) )
            CappU1.append( np.amax( [z[l] for z in Capp1] ) )
            CappL1.append( np.amin( [z[l] for z in Capp1] ) )
        axs[0].plot(centers,Capp31,lw=1.,ls='--',c='m',alpha=0.7,label="-1.25, removed faint galaxies")
        axs[0].fill_between(centers,CappU1,CappL1,facecolor='m',alpha=0.2,interpolate=True)
        axs[0].plot(centers,Capp3,lw=1.,ls='--',c='c',alpha=0.7,label="-1.57, removed faint galaxies")
        axs[0].fill_between(centers,CappU,CappL,facecolor='c',alpha=0.2,interpolate=True)
        rel = (spec-Capp3)/( np.array(CappU)-np.array(CappL) )
        rel1 = (spec-Capp31)/( np.array(CappU1)-np.array(CappL1) )
        ta = axs[1].twinx()
        ta.set_ylabel('(data-model)/error',color='m')
        ta.tick_params(axis='y',labelcolor='m')
        ta.plot(centers,rel,lw=1.,ls='--',c='c',alpha=0.7,label='(data-model)/error')
        ta.plot(centers,rel1,lw=1.,ls='--',c='m',alpha=0.7,label='(data-model)/error')
    if pl2:
        Capp31 = 1.37*centers**(-1.25+2)
        Capp1 = [(1.37+0.3)*centers**(-1.25+2),(1.37-0.3)*centers**(-1.25+2),(1.37)*centers**(-1.25+0.35+2),(1.37)*centers**(-1.25-0.35+2),(1.37+0.3)*centers**(-1.25+0.35+2),(1.37+0.3)*centers**(-1.25-0.35+2),(1.37-0.3)*centers**(-1.25+0.35+2),(1.37-0.3)*centers**(-1.25-0.35+2)]
        Capp3 = 4.18*centers**(-1.57+2)
        Capp = [(4.18+0.26)*centers**(-1.57+2),(4.18-0.26)*centers**(-1.57+2),(4.18)*centers**(-1.57+0.1+2),(4.18)*centers**(-1.57-0.1+2),(4.18+0.26)*centers**(-1.57+0.1+2),(4.18+0.26)*centers**(-1.57-0.1+2),(4.18-0.26)*centers**(-1.57+0.1+2),(4.18-0.26)*centers**(-1.57-0.1+2)]
        CappU1 = []
        CappU = []
        CappL1 = []
        CappL = []
        for l in range(len(Capp[0])):
            CappU.append( np.amax( [z[l] for z in Capp] ) )
            CappL.append( np.amin( [z[l] for z in Capp] ) )
            CappU1.append( np.amax( [z[l] for z in Capp1] ) )
            CappL1.append( np.amin( [z[l] for z in Capp1] ) )
        axs[0].plot(centers,Capp31,lw=1.,ls='--',c='m',alpha=0.7,label="-1.25, removed faint galaxies")
        axs[0].fill_between(centers,CappU1,CappL1,facecolor='m',alpha=0.2,interpolate=True)
        axs[0].plot(centers,Capp3,lw=1.,ls='--',c='c',alpha=0.7,label="-1.57, removed known sources")
        axs[0].fill_between(centers,CappU,CappL,facecolor='c',alpha=0.2,interpolate=True)
        rel = (spec-Capp3)/( np.array(CappU)-np.array(CappL) )
        rel1 = (spec-Capp31)/( np.array(CappU1)-np.array(CappL1) )
        ta = axs[1].twinx()
        ta.set_ylabel('(data-model)/error',color='m')
        ta.tick_params(axis='y',labelcolor='m')
        ta.plot(centers,rel,lw=1.,ls='--',c='c',alpha=0.7,label='(data-model)/error')
        ta.plot(centers,rel1,lw=1.,ls='--',c='m',alpha=0.7,label='(data-model)/error')
        
    axs[1].plot(centers,contr1,ls=":",lw=0.9,c="b",label=l1)
    axs[1].plot(centers,contr2,ls=":",lw=0.9,c="r",label=l2)
    axs[1].plot(centers,contr3,ls="-",lw=0.9,c="g",label=l3)
    axs[0].legend(loc='lower left')
    if pl1 or pl2:
        axs[1].legend(loc='upper right')
    axs[1].set_xlabel(r'$E_{ph}$ [keV]',fontsize=14)
    if pl1:
        axs[0].set_ylabel(r'$f_E\quad\mathrm{[cm^{-2}\,s^{-1}\,keV^{-1}\,sr^{-1}]}$',fontsize=12)
    elif pl2:
        axs[0].set_ylabel(r'$E^2\,f_E\quad\mathrm{[keV^2\,cm^{-2}\,s^{-1}\,keV^{-1}\,sr^{-1}]}$',fontsize=12)
    else:
        axs[0].set_ylabel(r'norm. counts',fontsize=12)
    axs[1].set_ylabel(r'$\frac{c_i}{c_{tot}}$',fontsize=16)
    
    plt.show()
    
    
def plot_zslice_contr(bins,spec1,spec2,spec3,zslice,l3=''):
    """
    spec is a list of spectra
    zslice is array of redshifts for lightcone slices
    If this is matplotlib you can get rid of the white lines in the scale with cbar.solids.set_rasterized(True)
    
    """
    if len(spec3) != len(zslice):
        print("Length of zslice doesn't match len specs!")
        return
    zsclice = zslice/np.amax(zslice)
    contr3_s = []
    
    from matplotlib.collections import LineCollection
    
    spec_ce = calc_bin_centers(bins)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.axis([0.3,10.,0.01,1.02])
    #ax.set_xscale('log')
    ax.loglog()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.set_yticks([0.1,0.2,0.4,0.6,0.8,1])
    ax.grid()
    ax.set_xlabel('$E_{ph}$ [keV]',fontsize=14)
    ax.set_ylabel(r'$\frac{c_{'+l3+'} }{c_{tot}}$',fontsize=16)
    
    for i,a in enumerate(spec3):
        contr3_s.append( np.array(a) / ( np.array(a)+np.array(spec1[i])+np.array(spec2[i]) ) )
       
    segments = [np.column_stack([spec_ce, y]) for y in contr3_s]
    lc = LineCollection(segments, lw=0.9, cmap='jet',alpha=0.75)
    lc.set_array(zslice)
    ax.add_collection(lc)
    axcb = fig.colorbar(lc)
    axcb.set_label(r"z")
    axcb.solids.set_rasterized(True)
    axcb.solids.set_edgecolor("face")
    plt.show()
        
    
def plot_hist(h1,h2,h3,bins,l1="",l2="",l3="",j=""):
    ce = calc_bin_centers(bins)
    xl = r'$\frac{c_{XRB}+c_{'+j+'}}{c_{'+j+'}}$'
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(211)
    ax3 = fig2.add_subplot(212,sharex=ax2)
    ax2.axis([bins[0],bins[-1],0.01,1])
    ax2.tick_params(labelbottom=False)
    ax2.loglog()
    #ax2.grid()
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax2.set_xticks([1.0,5.0,10.,20.,50.,100.])
    ax2.step(ce,h1,where="mid",lw=0.9,c="r",alpha=0.8,label=l1,marker='D',ms=5.,fillstyle='none')
    ax2.step(ce,h2,where="mid",lw=0.9,c="b",alpha=0.8,label=l2,marker='D',ms=5.,fillstyle='none')
    ax2.step(ce,h3,where="mid",lw=0.9,c="g",alpha=0.8,label=l3,marker='D',ms=5.,fillstyle='none')
    
    ax3.plot(ce,cuml_hist(h1),ls="-",lw=0.9,c="r",label="",marker='D',ms=5.,fillstyle='none')
    ax3.plot(ce,cuml_hist(h2),ls="-",lw=0.9,c="b",label="",marker='D',ms=5.,fillstyle='none')
    ax3.plot(ce,cuml_hist(h3),ls="-",lw=0.9,c="g",label="",marker='D',ms=5.,fillstyle='none')
    ax2.legend()
    ax3.grid()
    ax3.set_xlabel(xl,fontsize=16)
    ax2.set_ylabel('fraction',fontsize=14)
    ax3.set_ylabel(r'fraction',fontsize=14)
    fig2.subplots_adjust(hspace=0.)
    plt.tight_layout(pad=0.1)
    plt.show()
    
def plot_hist2(h1,h2,h3,h4,h5,h6,bins,l1="",l2="",l3="",j=""):
    ce = calc_bin_centers(bins)
    xl = r'$\frac{c_{XRB}+c_{'+j+'}}{c_{'+j+'}}$'
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(211)
    ax3 = fig2.add_subplot(212,sharex=ax2)
    ax2.axis([bins[0],bins[-1],0.01,1])
    ax2.tick_params(labelbottom=False)
    ax2.grid()
    ax2.loglog()
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax2.set_xticks([1.0,5.0,10.,20.,50.,100.])
    ax2.step(ce,h1,where="mid",lw=0.9,c="r",alpha=0.8,label=l1,marker='D',ms=5.,fillstyle='none')
    ax2.step(ce,h2,where="mid",lw=0.9,c="b",alpha=0.8,label=l2,marker='D',ms=5.,fillstyle='none')
    ax2.step(ce,h3,where="mid",lw=0.9,c="g",alpha=0.8,label=l3,marker='D',ms=5.,fillstyle='none')
    ax2.step(ce,h4,where="mid",lw=0.9,c="r",alpha=0.8,label="",marker='o',ms=5.,fillstyle='none',ls='--')
    ax2.step(ce,h5,where="mid",lw=0.9,c="b",alpha=0.8,label="",marker='o',ms=5.,fillstyle='none',ls='--')
    ax2.step(ce,h6,where="mid",lw=0.9,c="g",alpha=0.8,label="",marker='o',ms=5.,fillstyle='none',ls='--')
    
    ax3.plot(ce,cuml_hist(h1),ls="-",lw=0.9,c="r",label="",marker='D',ms=5.,fillstyle='none')
    ax3.plot(ce,cuml_hist(h2),ls="-",lw=0.9,c="b",label="",marker='D',ms=5.,fillstyle='none')
    ax3.plot(ce,cuml_hist(h3),ls="-",lw=0.9,c="g",label="",marker='D',ms=5.,fillstyle='none')
    ax3.plot(ce,cuml_hist(h4),ls="--",lw=0.9,c="r",label="",marker='o',ms=5.,fillstyle='none')
    ax3.plot(ce,cuml_hist(h5),ls="--",lw=0.9,c="b",label="",marker='o',ms=5.,fillstyle='none')
    ax3.plot(ce,cuml_hist(h6),ls="--",lw=0.9,c="g",label="",marker='o',ms=5.,fillstyle='none')
    ax2.legend()
    ax3.grid()
    ax3.set_xlabel(xl,fontsize=16)
    ax2.set_ylabel('fraction',fontsize=14)
    ax3.set_ylabel(r'fraction',fontsize=14)
    fig2.subplots_adjust(hspace=0.)
    plt.tight_layout(pad=0.1)
    plt.show()
    
def plot_cnt_ratio(counts_GAS_sx,counts_GAS_hx,counts_GAS_mx,counts_AGN_sx,counts_AGN_hx,counts_AGN_mx,counts_XRB_sx,counts_XRB_hx,counts_XRB_mx,use_AGN,use_GAS,cnt_lim):
    
    counts_add_sx   = counts_XRB_sx
    counts_add_hx   = counts_XRB_hx
    counts_add_mx   = counts_XRB_mx
    
    if use_AGN:
        if len(counts_AGN_sx) != len(counts_XRB_sx):
            print("... Somehow there are not equal number of files for AGN ...")
            counts_AGN_sx = np.zeros_like(counts_XRB_sx)
        counts_add_sx = counts_add_sx + counts_AGN_sx
        counts_add_hx = counts_add_hx + counts_AGN_hx
        counts_add_mx = counts_add_mx + counts_AGN_mx
    
    if use_GAS:
        if len(counts_GAS_sx) != len(counts_XRB_sx):
            print("... Somehow there are not equal number of files for ICM ...")
            counts_GAS = np.zeros_like(counts_XRB_sx)
        counts_add_sx = counts_add_sx + counts_GAS_sx
        counts_add_hx = counts_add_hx + counts_GAS_hx
        counts_add_mx = counts_add_mx + counts_GAS_mx
    
    
    if use_GAS:
        lim_mask_sx = (counts_GAS_sx >= cnt_lim)
        lim_mask_hx = (counts_GAS_hx >= cnt_lim)
        lim_mask_mx = (counts_GAS_mx >= cnt_lim)
        
        np.clip(counts_GAS_sx,.1,a_max=None,out=counts_GAS_sx)
        np.clip(counts_GAS_hx,.1,a_max=None,out=counts_GAS_hx)
        np.clip(counts_GAS_mx,.1,a_max=None,out=counts_GAS_mx)
        
        ratio_sx_g = counts_XRB_sx[lim_mask_sx] / counts_GAS_sx[lim_mask_sx] + 1
        ratio_hx_g = counts_XRB_hx[lim_mask_hx] / counts_GAS_hx[lim_mask_hx] + 1
        ratio_mx_g = counts_XRB_mx[lim_mask_mx] / counts_GAS_mx[lim_mask_mx] + 1
        
        average_sx_g = np.average(ratio_sx_g,weights=counts_GAS_sx[lim_mask_sx]/np.sum(counts_GAS_sx[lim_mask_sx]))
        average_hx_g = np.average(ratio_hx_g,weights=counts_GAS_hx[lim_mask_hx]/np.sum(counts_GAS_hx[lim_mask_hx]))
        average_mx_g = np.average(ratio_mx_g,weights=counts_GAS_mx[lim_mask_mx]/np.sum(counts_GAS_mx[lim_mask_mx]))
        
        Nbins = np.logspace(0,2,11)

        gal_hist_sx_g, edges = np.histogram(np.clip(ratio_sx_g,a_min=None,a_max=Nbins[-1]),Nbins)
        gal_hist_samp_sx_g = gal_hist_sx_g / len(ratio_sx_g) #normalized by sample size
        gal_hist_tot_sx_g  = gal_hist_sx_g / len(counts_XRB_sx) # normalized by total number

        gal_hist_hx_g, edges = np.histogram(np.clip(ratio_hx_g,a_min=None,a_max=Nbins[-1]),Nbins)
        gal_hist_samp_hx_g = gal_hist_hx_g / len(ratio_hx_g) #normalized by sample size
        gal_hist_tot_hx_g  = gal_hist_hx_g / len(counts_XRB_hx) # normalized by total number

        gal_hist_mx_g, edges = np.histogram(np.clip(ratio_mx_g,a_min=None,a_max=Nbins[-1]),Nbins)
        gal_hist_samp_mx_g = gal_hist_mx_g / len(ratio_mx_g) #normalized by sample size
        gal_hist_tot_mx_g  = gal_hist_mx_g / len(counts_XRB_mx) # normalized by total number
        
        #plot_hist2(gal_hist_samp_sx_g,gal_hist_samp_hx_g,gal_hist_samp_mx_g,gal_hist_tot_sx_g,gal_hist_tot_hx_g,gal_hist_tot_mx_g,Nbins,l1="0.5 - 2 keV",l2="2 - 10 keV",l3="0.5 - 10 keV",j="GAS")
        plot_hist(gal_hist_samp_sx_g,gal_hist_samp_hx_g,gal_hist_samp_mx_g,Nbins,l1="0.5 - 2 keV",l2="2 - 10 keV",l3="0.5 - 10 keV",j="GAS")
        
    if use_AGN:
        lim_mask_sx = (counts_AGN_sx >= cnt_lim)
        lim_mask_hx = (counts_AGN_hx >= cnt_lim)
        lim_mask_mx = (counts_AGN_mx >= cnt_lim)
        
        np.clip(counts_AGN_sx,.1,a_max=None,out=counts_AGN_sx)
        np.clip(counts_AGN_hx,.1,a_max=None,out=counts_AGN_hx)
        np.clip(counts_AGN_mx,.1,a_max=None,out=counts_AGN_mx)
        
        ratio_sx_a = counts_XRB_sx[lim_mask_sx] / counts_AGN_sx[lim_mask_sx] + 1
        ratio_hx_a = counts_XRB_hx[lim_mask_hx] / counts_AGN_hx[lim_mask_hx] + 1
        ratio_mx_a = counts_XRB_mx[lim_mask_mx] / counts_AGN_mx[lim_mask_mx] + 1
        
        average_sx_a = np.average(ratio_sx_a,weights=counts_AGN_sx[lim_mask_sx]/np.sum(counts_AGN_sx[lim_mask_sx]))
        average_hx_a = np.average(ratio_hx_a,weights=counts_AGN_hx[lim_mask_hx]/np.sum(counts_AGN_hx[lim_mask_hx]))
        average_mx_a = np.average(ratio_mx_a,weights=counts_AGN_mx[lim_mask_mx]/np.sum(counts_AGN_mx[lim_mask_mx]))
        
        gal_hist_sx_a, edges = np.histogram(np.clip(ratio_sx_a,a_min=None,a_max=Nbins[-1]),Nbins)
        gal_hist_samp_sx_a = gal_hist_sx_a / len(ratio_sx_a) #normalized by sample size
        gal_hist_tot_sx_a  = gal_hist_sx_a / len(counts_XRB_sx) # normalized by total number

        gal_hist_hx_a, edges = np.histogram(np.clip(ratio_hx_a,a_min=None,a_max=Nbins[-1]),Nbins)
        gal_hist_samp_hx_a = gal_hist_hx_a / len(ratio_hx_a) #normalized by sample size
        gal_hist_tot_hx_a  = gal_hist_hx_a / len(counts_XRB_hx) # normalized by total number

        gal_hist_mx_a, edges = np.histogram(np.clip(ratio_mx_a,a_min=None,a_max=Nbins[-1]),Nbins)
        gal_hist_samp_mx_a = gal_hist_mx_a / len(ratio_mx_a) #normalized by sample size
        gal_hist_tot_mx_a  = gal_hist_mx_a / len(counts_XRB_mx) # normalized by total number
        
        #plot_hist2(gal_hist_samp_sx_a,gal_hist_samp_hx_a,gal_hist_samp_mx_a,gal_hist_tot_sx_a,gal_hist_tot_hx_a,gal_hist_tot_mx_a,Nbins,l1="0.5 - 2 keV",l2="2 - 10 keV",l3="0.5 - 10 keV",j="AGN")
        plot_hist(gal_hist_samp_sx_a,gal_hist_samp_hx_a,gal_hist_samp_mx_a,Nbins,l1="0.5 - 2 keV",l2="2 - 10 keV",l3="0.5 - 10 keV",j="AGN")
        
        
def plot_stat_bin(phE1,phE2,phE3,stat,stat_edges,spec_bins,l1="",l2="",l3="",statl=""):
    
    Nbins = spec_bins
    centers = calc_bin_centers(spec_bins)
    if len(stat_edges) != 5:
        print("myplot.py 'plot_stat_bin()': only 4 bins supported")
        exit()
    if len(phE1) != len(phE2) or len(phE1) != len(phE3) or len(phE2) != len(phE3):
        print("myplot.py 'plot_stat_bin()': input energies must same length list of arrays")
        exit()
    #Ninf = len(stat)/4.
    #stat_hist,stat_edges = np.histogram(stat,stat_edges)
    #w = (stat_edges[1]-stat_edges[0])
    #print(stat_hist,Ninf,np.sum(stat_hist)/4)
    
    stat_bin_1 = [ihal for ihal,st in enumerate(stat) if st >= stat_edges[0] and st < stat_edges[1]]
    stat_bin_2 = [ihal for ihal,st in enumerate(stat) if st >= stat_edges[1] and st < stat_edges[2]]
    stat_bin_3 = [ihal for ihal,st in enumerate(stat) if st >= stat_edges[2] and st < stat_edges[3]]
    stat_bin_4 = [ihal for ihal,st in enumerate(stat) if st >= stat_edges[3] and st < stat_edges[4]]
    
    if statl == "SFR":
        statl_1 = r"${:.0f}<".format(stat_edges[0])+statl+"<{:.3f}$, $N_g={:d}$".format(stat_edges[1],len(stat_bin_1))
        statl_2 = r"${:.3f}<".format(stat_edges[1])+statl+"<{:.1f}$, $N_g={:d}$".format(stat_edges[2],len(stat_bin_2))
        statl_3 = r"${:.1f}<".format(stat_edges[2])+statl+"<{:.1f}$, $N_g={:d}$".format(stat_edges[3],len(stat_bin_3))
        statl_4 = r"${:.1f}<".format(stat_edges[3])+statl+"<{:.1f}$, $N_g={:d}$".format(stat_edges[4],len(stat_bin_4))
    elif statl == "sSFR":
        statl_1 = r"$\log("+statl+")<{:.1f}$, $N_g={:d}$".format(np.log10(stat_edges[1]),len(stat_bin_1))
        statl_2 = r"{:.1f}<$\log(".format(np.log10(stat_edges[1]))+statl+")<{:.1f}$, $N_g={:d}$".format(np.log10(stat_edges[2]),len(stat_bin_2))
        statl_3 = r"{:.1f}<$\log(".format(np.log10(stat_edges[2]))+statl+")<{:.1f}$, $N_g={:d}$".format(np.log10(stat_edges[3]),len(stat_bin_3))
        statl_4 = r"{:.1f}<$\log(".format(np.log10(stat_edges[3]))+statl+")<{:.1f}$, $N_g={:d}$".format(np.log10(stat_edges[4]),len(stat_bin_4))
    else:
        statl_1 = r"${:.1f}<\log(".format(np.log10(stat_edges[0]))+statl+")<{:.1f}$, $N_g={:d}$".format(np.log10(stat_edges[1]),len(stat_bin_1))
        statl_2 = r"${:.1f}<\log(".format(np.log10(stat_edges[1]))+statl+")<{:.1f}$, $N_g={:d}$".format(np.log10(stat_edges[2]),len(stat_bin_2))
        statl_3 = r"${:.1f}<\log(".format(np.log10(stat_edges[2]))+statl+")<{:.1f}$, $N_g={:d}$".format(np.log10(stat_edges[3]),len(stat_bin_3))
        statl_4 = r"${:.1f}<\log(".format(np.log10(stat_edges[3]))+statl+")<{:.1f}$, $N_g={:d}$".format(np.log10(stat_edges[4]),len(stat_bin_4))
    
    ph1_hist_arr = []
    ph2_hist_arr = []
    ph3_hist_arr = []
    phT_hist_arr = []
    contr1_arr   = []
    contr2_arr   = []
    contr3_arr   = []
    
    
    for p in range(len(phE1)):
        nph = len(phE1[p]) + len(phE2[p]) + len(phE3[p])
        ph1_hist, tot_edges = np.histogram(phE1[p],Nbins)#,edgecolor='g',histtype='step',label="XRB")
        ph2_hist, tot_edges = np.histogram(phE2[p],Nbins)#,edgecolor='r',histtype='step',label="AGN")
        ph3_hist, tot_edges = np.histogram(phE3[p],Nbins)#,edgecolor='b',histtype='step',label="GAS")
        if nph > 0:
            ph1_hist_arr.append(ph1_hist / nph)
            ph2_hist_arr.append(ph2_hist / nph)
            ph3_hist_arr.append(ph3_hist / nph)
            phT_hist_arr.append( (ph1_hist + ph2_hist + ph3_hist) / nph )
        else:
            ph1_hist_arr.append( np.zeros(len(Nbins)-1) )
            ph2_hist_arr.append( np.zeros(len(Nbins)-1) )
            ph3_hist_arr.append( np.zeros(len(Nbins)-1) )
            phT_hist_arr.append( np.zeros(len(Nbins)-1) )
        
    phT_hist_avg_1 = []; ph1_hist_avg_1 = []; ph2_hist_avg_1 = []; ph3_hist_avg_1 = []
    phT_hist_avg_2 = []; ph1_hist_avg_2 = []; ph2_hist_avg_2 = []; ph3_hist_avg_2 = []
    phT_hist_avg_3 = []; ph1_hist_avg_3 = []; ph2_hist_avg_3 = []; ph3_hist_avg_3 = []
    phT_hist_avg_4 = []; ph1_hist_avg_4 = []; ph2_hist_avg_4 = []; ph3_hist_avg_4 = []
    
    phT_hist_sig_1 = []; ph1_hist_sig_1 = []; ph2_hist_sig_1 = []; ph3_hist_sig_1 = []
    phT_hist_sig_2 = []; ph1_hist_sig_2 = []; ph2_hist_sig_2 = []; ph3_hist_sig_2 = []
    phT_hist_sig_3 = []; ph1_hist_sig_3 = []; ph2_hist_sig_3 = []; ph3_hist_sig_3 = []
    phT_hist_sig_4 = []; ph1_hist_sig_4 = []; ph2_hist_sig_4 = []; ph3_hist_sig_4 = []
    
    phT_hist_med_1 = []; ph1_hist_med_1 = []; ph2_hist_med_1 = []; ph3_hist_med_1 = []    
    phT_hist_med_2 = []; ph1_hist_med_2 = []; ph2_hist_med_2 = []; ph3_hist_med_2 = []    
    phT_hist_med_3 = []; ph1_hist_med_3 = []; ph2_hist_med_3 = []; ph3_hist_med_3 = []    
    phT_hist_med_4 = []; ph1_hist_med_4 = []; ph2_hist_med_4 = []; ph3_hist_med_4 = []
    
    phT_hist_mad_1 = []; ph1_hist_mad_1 = []; ph2_hist_mad_1 = []; ph3_hist_mad_1 = []    
    phT_hist_mad_2 = []; ph1_hist_mad_2 = []; ph2_hist_mad_2 = []; ph3_hist_mad_2 = []    
    phT_hist_mad_3 = []; ph1_hist_mad_3 = []; ph2_hist_mad_3 = []; ph3_hist_mad_3 = []    
    phT_hist_mad_4 = []; ph1_hist_mad_4 = []; ph2_hist_mad_4 = []; ph3_hist_mad_4 = []
    
    for l in range(len(Nbins)-1):
        """
        Make list of specs in stat_bin
        take elements 'alist' from the first list
        take element l of element 'alist'
        """
        temp_arr = [alist[l] for alist in [phT_hist_arr[x] for x in stat_bin_1] ] 
        #phT_hist_med_1.append(np.median(temp_arr))
        #phT_hist_mad_1.append(median_abs_dev(temp_arr)) #median absolute deviation of the median
        phT_hist_avg_1.append(np.mean(temp_arr))
        phT_hist_sig_1.append(np.std(temp_arr))
        
        temp_arr = [alist[l] for alist in [phT_hist_arr[x] for x in stat_bin_2] ]
        #phT_hist_med_2.append(np.median(temp_arr))
        #phT_hist_mad_2.append(median_abs_dev(temp_arr)) #median absolute deviation of the median
        phT_hist_avg_2.append(np.mean(temp_arr))
        phT_hist_sig_2.append(np.std(temp_arr))
        
        temp_arr = [alist[l] for alist in [phT_hist_arr[x] for x in stat_bin_3] ]
        #phT_hist_med_3.append(np.median(temp_arr))
        #phT_hist_mad_3.append(median_abs_dev(temp_arr)) #median absolute deviation of the median
        phT_hist_avg_3.append(np.mean(temp_arr))
        phT_hist_sig_3.append(np.std(temp_arr))
        
        temp_arr = [alist[l] for alist in [phT_hist_arr[x] for x in stat_bin_4] ]
        #phT_hist_med_4.append(np.median(temp_arr))
        #phT_hist_mad_4.append(median_abs_dev(temp_arr)) #median absolute deviation of the median
        phT_hist_avg_4.append(np.mean(temp_arr))
        phT_hist_sig_4.append(np.std(temp_arr))
        
        temp_arr = [alist[l] for alist in [ph1_hist_arr[x] for x in stat_bin_1] ]
        #ph1_hist_med_1.append(np.median(temp_arr))
        #ph1_hist_mad_1.append(median_abs_dev(temp_arr)) #median absolute deviation of the median
        ph1_hist_avg_1.append(np.mean(temp_arr))
        ph1_hist_sig_1.append(np.std(temp_arr))
        
        temp_arr = [alist[l] for alist in [ph1_hist_arr[x] for x in stat_bin_2] ]
        #ph1_hist_med_2.append(np.median(temp_arr))
        #ph1_hist_mad_2.append(median_abs_dev(temp_arr)) #median absolute deviation of the median
        ph1_hist_avg_2.append(np.mean(temp_arr))
        ph1_hist_sig_2.append(np.std(temp_arr))
        
        temp_arr = [alist[l] for alist in [ph1_hist_arr[x] for x in stat_bin_3] ]
        #ph1_hist_med_3.append(np.median(temp_arr))
        #ph1_hist_mad_3.append(median_abs_dev(temp_arr)) #median absolute deviation of the median
        ph1_hist_avg_3.append(np.mean(temp_arr))
        ph1_hist_sig_3.append(np.std(temp_arr))
        
        temp_arr = [alist[l] for alist in [ph1_hist_arr[x] for x in stat_bin_4] ]
        #ph1_hist_med_4.append(np.median(temp_arr))
        #ph1_hist_mad_4.append(median_abs_dev(temp_arr)) #median absolute deviation of the median
        ph1_hist_avg_4.append(np.mean(temp_arr))
        ph1_hist_sig_4.append(np.std(temp_arr))
        
        temp_arr = [alist[l] for alist in [ph2_hist_arr[x] for x in stat_bin_1] ]
        #ph2_hist_med_1.append(np.median(temp_arr))
        #ph2_hist_mad_1.append(median_abs_dev(temp_arr)) #median absolute deviation of the median
        ph2_hist_avg_1.append(np.mean(temp_arr))
        ph2_hist_sig_1.append(np.std(temp_arr))
        
        temp_arr = [alist[l] for alist in [ph2_hist_arr[x] for x in stat_bin_2] ]
        #ph2_hist_med_2.append(np.median(temp_arr))
        #ph2_hist_mad_2.append(median_abs_dev(temp_arr)) #median absolute deviation of the median
        ph2_hist_avg_2.append(np.mean(temp_arr))
        ph2_hist_sig_2.append(np.std(temp_arr))
        
        temp_arr = [alist[l] for alist in [ph2_hist_arr[x] for x in stat_bin_3] ]
        #ph2_hist_med_3.append(np.median(temp_arr))
        #ph2_hist_mad_3.append(median_abs_dev(temp_arr)) #median absolute deviation of the median
        ph2_hist_avg_3.append(np.mean(temp_arr))
        ph2_hist_sig_3.append(np.std(temp_arr))
        
        temp_arr = [alist[l] for alist in [ph2_hist_arr[x] for x in stat_bin_4] ]
        #ph2_hist_med_4.append(np.median(temp_arr))
        #ph2_hist_mad_4.append(median_abs_dev(temp_arr)) #median absolute deviation of the median
        ph2_hist_avg_4.append(np.mean(temp_arr))
        ph2_hist_sig_4.append(np.std(temp_arr))
        
        temp_arr = [alist[l] for alist in [ph3_hist_arr[x] for x in stat_bin_1] ]
        #ph3_hist_med_1.append(np.median(temp_arr))
        #ph3_hist_mad_1.append(median_abs_dev(temp_arr)) #median absolute deviation of the median
        ph3_hist_avg_1.append(np.mean(temp_arr))
        ph3_hist_sig_1.append(np.std(temp_arr))
        
        temp_arr = [alist[l] for alist in [ph3_hist_arr[x] for x in stat_bin_2] ]
        #ph3_hist_med_2.append(np.median(temp_arr))
        #ph3_hist_mad_2.append(median_abs_dev(temp_arr)) #median absolute deviation of the median
        ph3_hist_avg_2.append(np.mean(temp_arr))
        ph3_hist_sig_2.append(np.std(temp_arr))
        
        temp_arr = [alist[l] for alist in [ph3_hist_arr[x] for x in stat_bin_3] ]
        #ph3_hist_med_3.append(np.median(temp_arr))
        #ph3_hist_mad_3.append(median_abs_dev(temp_arr)) #median absolute deviation of the median
        ph3_hist_avg_3.append(np.mean(temp_arr))
        ph3_hist_sig_3.append(np.std(temp_arr))
        
        temp_arr = [alist[l] for alist in [ph3_hist_arr[x] for x in stat_bin_4] ]
        #ph3_hist_med_4.append(np.median(temp_arr))
        #ph3_hist_mad_4.append(median_abs_dev(temp_arr)) #median absolute deviation of the median
        ph3_hist_avg_4.append(np.mean(temp_arr))
        ph3_hist_sig_4.append(np.std(temp_arr))
        
    contr1_arr_1 = [ ph1_hist_avg_1[i] / phT_hist_avg_1[i] for i in range(len(phT_hist_avg_1)) ]
    contr2_arr_1 = [ ph2_hist_avg_1[i] / phT_hist_avg_1[i] for i in range(len(phT_hist_avg_1)) ]
    contr3_arr_1 = [ ph3_hist_avg_1[i] / phT_hist_avg_1[i] for i in range(len(phT_hist_avg_1)) ]
    
    contr1_arr_2 = [ ph1_hist_avg_2[i] / phT_hist_avg_2[i] for i in range(len(phT_hist_avg_1)) ]
    contr2_arr_2 = [ ph2_hist_avg_2[i] / phT_hist_avg_2[i] for i in range(len(phT_hist_avg_1)) ]
    contr3_arr_2 = [ ph3_hist_avg_2[i] / phT_hist_avg_2[i] for i in range(len(phT_hist_avg_1)) ]
    
    contr1_arr_3 = [ ph1_hist_avg_3[i] / phT_hist_avg_3[i] for i in range(len(phT_hist_avg_1)) ]
    contr2_arr_3 = [ ph2_hist_avg_3[i] / phT_hist_avg_3[i] for i in range(len(phT_hist_avg_1)) ]
    contr3_arr_3 = [ ph3_hist_avg_3[i] / phT_hist_avg_3[i] for i in range(len(phT_hist_avg_1)) ]
    
    contr1_arr_4 = [ ph1_hist_avg_4[i] / phT_hist_avg_4[i] for i in range(len(phT_hist_avg_1)) ]
    contr2_arr_4 = [ ph2_hist_avg_4[i] / phT_hist_avg_4[i] for i in range(len(phT_hist_avg_1)) ]
    contr3_arr_4 = [ ph3_hist_avg_4[i] / phT_hist_avg_4[i] for i in range(len(phT_hist_avg_1)) ]
        
    #mean_arrs_1 = (phT_hist_avg_1,ph1_hist_avg_1,ph2_hist_avg_1,ph3_hist_avg_1)
    #mean_arrs_2 = (phT_hist_avg_2,ph1_hist_avg_2,ph2_hist_avg_2,ph3_hist_avg_2)
    #mean_arrs_3 = (phT_hist_avg_3,ph1_hist_avg_3,ph2_hist_avg_3,ph3_hist_avg_3)
    #mean_arrs_4 = (phT_hist_avg_4,ph1_hist_avg_4,ph2_hist_avg_4,ph3_hist_avg_4)
    
    #contr_arrs_1 = (contr1_arr_1,contr2_arr_1,contr3_arr_1)
    #contr_arrs_2 = (contr1_arr_2,contr2_arr_2,contr3_arr_2)
    #contr_arrs_3 = (contr1_arr_3,contr2_arr_3,contr3_arr_3)
        
    f = plt.figure(figsize=(10,10))
    gs3 = f.add_gridspec(nrows=2,ncols=1,left=0.08,right=0.47,top=0.48,bottom=0.05,hspace=0.01,height_ratios=[2,1])   
    gs4 = f.add_gridspec(nrows=2,ncols=1,left=0.58,right=0.97,top=0.48,bottom=0.05,hspace=0.01,height_ratios=[2,1])
    gs1 = f.add_gridspec(nrows=2,ncols=1,left=0.08,right=0.47,top=0.98,bottom=0.55,hspace=0.01,height_ratios=[2,1])
    gs2 = f.add_gridspec(nrows=2,ncols=1,left=0.58,right=0.97,top=0.98,bottom=0.55,hspace=0.01,height_ratios=[2,1])
    
    axs = []
    axs.append( f.add_subplot(gs1[0,0]) )
    axs.append( f.add_subplot(gs1[1,0]) )
    axs.append( f.add_subplot(gs2[0,0]) )
    axs.append( f.add_subplot(gs2[1,0]) )
    axs.append( f.add_subplot(gs3[0,0]) )
    axs.append( f.add_subplot(gs3[1,0]) )
    axs.append( f.add_subplot(gs4[0,0]) )
    axs.append( f.add_subplot(gs4[1,0]) )
    
    for ax in axs[::2]:
        ax.axis([Nbins[0],Nbins[-1],5e-5,5e-2])
        ax.loglog()
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0e}'.format(y)))
        ax.tick_params(labelbottom=False,bottom=False)
        ax.grid()
        ax.set_ylabel("norm. counts",fontsize=14)
    
    for ax in axs[1::2]:
        ax.axis([Nbins[0],Nbins[-1],-0.02,1.02])
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
        ax.grid()
        ax.set_xlabel('$E_{ph}$ [keV]',fontsize=14)
        ax.set_ylabel(r'$\frac{c_i}{c_{tot}}$',fontsize=16)
    
    axs[0].text(0.15,0.03,statl_1,fontsize=10,bbox=dict(facecolor='grey', alpha=0.5))
    axs[0].step(centers,phT_hist_avg_1,lw=0.9,c="k",where="mid",label="comb.")
    axs[0].step(centers,ph1_hist_avg_1,lw=0.9,c="b",where="mid",label=l1)
    axs[0].step(centers,ph2_hist_avg_1,lw=0.9,c="r",where="mid",label=l2)
    axs[0].step(centers,ph3_hist_avg_1,lw=0.9,c="g",where="mid",label=l3)
    axs[1].plot(centers,contr1_arr_1,lw=0.9,c="b",ls=":")
    axs[1].plot(centers,contr2_arr_1,lw=0.9,c="r",ls=":")
    axs[1].plot(centers,contr3_arr_1,lw=0.9,c="g",ls="-")
    #print(np.sum(phT_hist_avg_1))
    
    axs[2].text(0.15,0.03,statl_2,fontsize=10,bbox=dict(facecolor='grey', alpha=0.5))
    axs[2].step(centers,phT_hist_avg_2,lw=0.9,c="k",where="mid",label="comb.")
    axs[2].step(centers,ph1_hist_avg_2,lw=0.9,c="b",where="mid",label=l1)
    axs[2].step(centers,ph2_hist_avg_2,lw=0.9,c="r",where="mid",label=l2)
    axs[2].step(centers,ph3_hist_avg_2,lw=0.9,c="g",where="mid",label=l3)
    axs[3].plot(centers,contr1_arr_2,lw=0.9,c="b",ls=":")
    axs[3].plot(centers,contr2_arr_2,lw=0.9,c="r",ls=":")
    axs[3].plot(centers,contr3_arr_2,lw=0.9,c="g",ls="-")
    #print(np.sum(phT_hist_avg_2))
    
    axs[4].text(0.15,0.03,statl_3,fontsize=10,bbox=dict(facecolor='grey', alpha=0.5))
    axs[4].step(centers,phT_hist_avg_3,lw=0.9,c="k",where="mid",label="comb.")
    axs[4].step(centers,ph1_hist_avg_3,lw=0.9,c="b",where="mid",label=l1)
    axs[4].step(centers,ph2_hist_avg_3,lw=0.9,c="r",where="mid",label=l2)
    axs[4].step(centers,ph3_hist_avg_3,lw=0.9,c="g",where="mid",label=l3)
    axs[5].plot(centers,contr1_arr_3,lw=0.9,c="b",ls=":")
    axs[5].plot(centers,contr2_arr_3,lw=0.9,c="r",ls=":")
    axs[5].plot(centers,contr3_arr_3,lw=0.9,c="g",ls="-")
    #print(np.sum(phT_hist_avg_3))
    
    axs[6].text(0.15,0.03,statl_4,fontsize=10,bbox=dict(facecolor='grey', alpha=0.5))
    axs[6].step(centers,phT_hist_avg_4,lw=0.9,c="k",where="mid",label="comb.")
    axs[6].step(centers,ph1_hist_avg_4,lw=0.9,c="b",where="mid",label=l1)
    axs[6].step(centers,ph2_hist_avg_4,lw=0.9,c="r",where="mid",label=l2)
    axs[6].step(centers,ph3_hist_avg_4,lw=0.9,c="g",where="mid",label=l3)
    axs[7].plot(centers,contr1_arr_4,lw=0.9,c="b",ls=":")
    axs[7].plot(centers,contr2_arr_4,lw=0.9,c="r",ls=":")
    axs[7].plot(centers,contr3_arr_4,lw=0.9,c="g",ls="-")
    #print(np.sum(phT_hist_avg_4))
    
    plt.show()
