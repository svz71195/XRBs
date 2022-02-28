import numpy as np
from scipy.special import gammaincc, gamma

### UNIT CONVERSION ###
KEV_TO_ERG = 1.60218e-9
KPC_TO_CM = 3.0856e21
MPC_TO_CM = 3.0856e24

### COMPONENT PLOT KWARGS ###
clrAGN = 'crimson'
clrLXB = 'yellowgreen'
clrHXB = 'darkgreen'
clrGAS = 'royalblue'

def GammaIncc(a,x):
    """
    Incomplete upper Gamma function optimized to also work with a < 0
    (native scipy functions don't allow this) using recursion.
    See "http://en.wikipedia.org/wiki/Incomplete_gamma_function#Properties"
    Used for integration of HMXB LF model from Lehmer+21 of the form:

    exp(-x/b) * x**(-a)  >>>> integration >>>>  -b**(1-a)*Gamma(1-a,x/b) + C
    """
    
    if a >= 0.:
        res = gammaincc(a,x)*gamma(a)
    else:
        res = ( GammaIncc(a+1,x)-x**(a)*np.exp(-x) ) / a
    return res

def calc_bin_centers(edges: np.ndarray):
    """
    Returns bin centers for histograms
    """
    return edges[:-1] + np.diff(edges)/2.

def calc_lum_cgs(phE: np.ndarray, Aeff: float, Tobs: float, dLum: float,
             Emin: float = .5, Emax: float = 8):
    """
    Calculate luminosity in energy range [Emin,Emax] from photon
    energy array phE
    Returns luminosity in erg/s
    -----
    Aeff: effective area in cm^2
    Tobs: exposure time in s
    dLum: luminosity distance in cm
    Emin: in keV
    Emax: in keV
    """
    if not isinstance(phE, np.ndarray):
        phE = np.array(phE)
    dLum2 = 4*np.pi*(dLum)**2/Aeff/Tobs
    Emask = (phE >= Emin) & (phE <= Emax)

    return np.sum(phE[Emask])*dLum2*KEV_TO_ERG

def calc_flux_per_bin(bins: np.ndarray, hist: np.ndarray,
                             Aeff: float = 1., Tobs: float = 1.):
    """
    Calculate flux per energy bin given a raw photon count
    histogram from a spectrum
    -----
    bins: bin edges for energy range (dim=N+1)
    hist: photon counts per energy bin
    Aeff: effective area in 'cm^2'
    Tobs: exposure time in 's'
    """
    bin_w = np.diff(bins)
    bin_c = calc_bin_centers(bins)
    flux_hist = hist / bin_w / Aeff / Tobs * bin_c**2
    return flux_hist

def get_num_XRB( phE_h: list, Tobs: float = 1., Aeff: float = 1., Dlum: float = 1., Lc: float = -1.):
    """
    Returns tuple containing number of XRBs and array of luminosities
    """

    indx_pckg_end_h = np.where( np.diff(phE_h) < 0)[0]
    numHXB = len(indx_pckg_end_h)

    lumH = np.zeros(numHXB)
    
    for i in range(numHXB):
        if i == 0:
            lumH[i] = calc_lum_cgs( phE_h[0:indx_pckg_end_h[0]+1], Tobs, Aeff, Dlum )
        else:
            lumH[i] = calc_lum_cgs( phE_h[indx_pckg_end_h[i-1]+1:indx_pckg_end_h[i]+1], Tobs, Aeff, Dlum )

    if Lc < 0:
        return (numHXB, lumH)
    else:
        lumH = lumH[lumH>Lc]
        numH = len(lumH)
        return (numH, lumH)

class Integrate:

    @staticmethod
    def Riemann(func: np.ufunc, lim_l: float, lim_u: float, n: int, *arg: tuple) -> float:
        eval = np.linspace(lim_l,lim_u,n+1)
        delta = np.diff(eval)
        res = func(eval[:n]+delta/2,*arg)*delta
        return np.sum(res)
    
    @staticmethod
    def Riemann_log(func: np.ufunc, 
                    lim_l: float, lim_u: float, n: int, *arg: tuple) -> float:
        """
        Custom implementation of a mid-point Riemann sum approximation with logarithmic
        measure. This is faster than and equally precise as 'scipy.integrate.quad()'
        specifically when integrating 'self.diff_Nhxb_met()'. Might also be faster than
        """
        eval = np.logspace(np.log10(lim_l),np.log10(lim_u),n+1)
        delta = np.diff(eval)
        res = func(eval[:n]+delta/2,*arg)*delta
        return np.sum(res)

    @staticmethod
    def Riemann_log_log(func,lim_l,lim_u,n):
        eval = np.logspace(np.log10(lim_l),np.log10(lim_u),n)
        leval = np.log(eval)
        delta = np.diff(leval)
        res = func(eval[:n-1])*delta*eval[:n-1]
        return np.sum(res)

    @staticmethod
    def Simpson(func,lim_l,lim_u,n,*arg):
        eval = np.linspace(lim_l,lim_u,n+1)
        delta = np.diff(eval[::2])
        f = func(eval,*arg)
        S = 1./6. * np.sum((f[0:-1:2] + 4*f[1::2] + f[2::2])*delta)
        return S

    @staticmethod
    def Simpson_log(func,lim_l,lim_u,n,*arg):
        eval = np.logspace(np.log10(lim_l),np.log10(lim_u),n+1)
        delta = np.diff(eval[::2])
        f = func(eval,*arg)
        S = 1./6. * np.sum((f[0:-1:2] + 4*f[1::2] + f[2::2])*delta)
        return S

    @staticmethod
    def Trapez(func,lim_l,lim_u,n,*arg):
        eval = np.linspace(lim_l,lim_u,n+1)
        f = func(eval,*arg)
        delta = np.diff(eval)
        T = .5 * np.sum((f[1:]+f[:-1])*delta)
        return T
    
    @staticmethod
    def Trapez_log(func,lim_l,lim_u,n,*arg):
        eval = np.logspace(np.log10(lim_l),np.log10(lim_u),n+1)
        f = func(eval,*arg)
        delta = np.diff(eval)
        T = .5 * np.sum((f[1:]+f[:-1])*delta)
        return T

class QuickStats:

    @staticmethod
    def chi2(d: np.ndarray, m: np.ndarray, s = 1., dof = 2.):
        """
        Quick and dirty chisquared estimate.
        Returns statistic.
        -----
        d: data
        m: model to compare data to
        s: y-error of data. Can either be single number or array
           with same length as data
        """
        r = d - m
        chisq = np.sum((r/s)**2)
        return chisq/(len(d)-dof)
    
    @staticmethod
    def log_err(x,dx):
        """
        Converts std absolute error to
        relative log error
        -----
        0.43429 = 1/ln(10)
        x: data
        dx: absolute std error of data
        """
        return 0.43429*dx/x