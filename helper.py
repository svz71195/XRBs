from typing import Callable
import numpy as np
from scipy.special import gammaincc, gamma

class Integrate:

    def Riemann(self, func: Callable, lim_l: float, lim_u: float, n: int, *arg: tuple) -> float:
        eval = np.linspace(lim_l,lim_u,n)
        delta = np.diff(eval)
        res = func(eval[:n-1]+delta/2,*arg)*delta
        return np.sum(res)

    def Riemann_log(self, func: Callable, 
                    lim_l: float, lim_u: float, n: int, *arg: tuple) -> float:
        """
        Custom implementation of a mid-point Riemann sum approximation with logarithmic
        measure. This is faster than and equally precise as 'scipy.integrate.quad()'
        specifically when integrating 'self.diff_Nhxb_met()'. Might also be faster than
        """
        eval = np.logspace(np.log10(lim_l),np.log10(lim_u),n)
        delta = np.diff(eval)
        res = func(eval[:n-1]+delta/2,*arg)*delta
        return np.sum(res)

    def Riemann_log_log(self,func,lim_l,lim_u,n):
        eval = np.logspace(np.log10(lim_l),np.log10(lim_u),n)
        leval = np.log(eval)
        delta = np.diff(leval)
        res = func(eval[:n-1])*delta*eval[:n-1]
        return np.sum(res)

    def Simpson(self,func,lim_l,lim_u,n,*arg):
        eval = np.linspace(lim_l,lim_u,n+1)
        delta = np.diff(eval[::2])
        f = func(eval,*arg)
        S = 1./6. * np.sum((f[0:-1:2] + 4*f[1::2] + f[2::2])*delta)
        return S

    def Simpson_log(self,func,lim_l,lim_u,n,*arg):
        eval = np.logspace(np.log10(lim_l),np.log10(lim_u),n+1)
        delta = np.diff(eval[::2])
        f = func(eval,*arg)
        S = 1./6. * np.sum((f[0:-1:2] + 4*f[1::2] + f[2::2])*delta)
        return S

    def Trapez(self,func,lim_l,lim_u,n,*arg):
        eval = np.linspace(lim_l,lim_u,n+1)
        f = func(eval,*arg)
        delta = np.diff(eval)
        T = .5 * np.sum((f[1:]+f[:-1])*delta)
        return T
    
    def Trapez_log(self,func,lim_l,lim_u,n,*arg):
        eval = np.logspace(np.log10(lim_l),np.log10(lim_u),n+1)
        f = func(eval,*arg)
        delta = np.diff(eval)
        T = .5 * np.sum((f[1:]+f[:-1])*delta)
        return T

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


class experimental:
    def diff_Nhxb_met(self, lum_in: float,
                      A: float, Lb: float, logLc: float, logLc_logZ: float,
                      g1: float, g2: float, g2_logZ: float,
                      logOH12: float ) -> float:
        """
        Differential function dN/dL for metallicity enhanced HMXB LF in Lehmer+21\\
        Needs to be integrated numerically. See implementation
        of 'self.Riemann_log(func: Callable, l_min: float, l_max: float, *par: tuple)'.

        NOTE: authors were not clear about normalization A in the Lehmer+21. They refer to Lehmer+19
        for a non-metallicity model of HMXBs which is normalized to 1e38 erg/s
        -----
        lum_in      :   input luminosity in units of 1.e38 erg/s
        A           :   model normalization at L = 1.e38 erg/s
        Lb          :   Power-law break luminosity
        logLc       :   base 10 logarithm of solar reference cut-off luminosity
        logLc_logZ  :   expansion constant of first derivative of log10(Z) dependent cut-off luminosity
                        used to calculate LcZ
        g1          :   Power-law slope of low luminosity regime
        g2          :   solar Z reference Power-law slope of high luminosity regime
        g2_logZ     :   expansion constant of first derivative log10(Z) dependent Power-law slope
        logOH12     :   metallicity measured as 12+log(O/H)
        -----
        in function
        LcZ         :   metallicity dependent cut-off luminosity
        g2Z         :   metallicity dependent high L slope
        """

        # need to rescale to 1.e38 erg/s normalization
        LcZ = 10**( logLc + logLc_logZ * ( logOH12 - 8.69 ) ) / 1.e38
        Lb  = Lb/1.e38

        g2Z = g2 + g2_logZ * ( logOH12 - 8.69 )
        

        if lum_in < Lb:
            return A * np.exp(-lum_in/LcZ) * lum_in**(-g1)
        else:
            return A * np.exp(-lum_in/LcZ) * lum_in**(-g2Z) * Lb**(g2Z-g1)

    def model_Nhxb(self, case: str = '0', SFR: float = 1., bRand: bool = False ):
        """
        Vectorization of analytic solutions. Depending on value passed to 'case',\\
            different model parameters can be loaded
        -----
        SFR         :   Rescaling parameter in units of Msun/yr
        bRand       :   boolean switching between randomized parameters\\
                            according to their uncertainty
        case        :   Decides which model to use, by passing KeyWord strings\\
                        'Mi12s' -> Mineo+2012 single PL\\
                        'Mi12b' -> Mineo+2012 broken PL\\
                        'Le20'  -> Lehmer+2020 broken PL\\
                        'Le21'  -> Lehmer+2021 metallicity\\
                        ...
                        Can also be accessed by passing integers starting from 0
                        
        """

        vec_calc_Nhxb_SPL = np.vectorize(self.calc_Nhxb_SPL)
        vec_calc_Nhxb_BPL = np.vectorize(self.calc_Nhxb_BPL)
        vec_calc_Nhxb_met = np.vectorize(self.diff_Nhxb_met)


        try:
            par: tuple = self.modelsH[case](bRand)

        except KeyError:
            raise KeyError("Desired model '"+str(case)+"' not implemented! Available models are",
                           [key for key in self.modelsH.keys() if len(str(key))>2]
                        )
        
        if len(par) == 3:
            Nhx_arr = vec_calc_Nhxb_SPL(self.lumarr/1.e38, *par)
        elif len(par) == 5:
            Nhx_arr = vec_calc_Nhxb_BPL(self.lumarr/1.e38, *par)
        elif len(par) == 8:
            Nhx_arr = np.zeros_like(self.lumarr)
            
            # need integral of each input luminosity in self.lumarr normalized to 1.e38
            # -----
            # the step number 'n' is scaled by how far we are into lumarr
            # this is to save time at integration since the contribution of near LcZ is
            # marginal
            # for loop takes < 5 sec
            # -----
            # par are parameters passed to func
            end = self.lumarr[-1] / 1.e38
            N = len(self.lumarr)
            for i,lum in enumerate(self.lumarr/1.e38):
                if i <= 2000:
                    steps = N
                else: # i == 2001
                    steps = int( N/np.sqrt((i-2000)/10.) )
                
                Nhx_arr[i] = self.Riemann_log(func = vec_calc_Nhxb_met,
                                              lum_l = lum,
                                              lum_u = end,
                                              n = steps,
                                              *par)

        return Nhx_arr * SFR