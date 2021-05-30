from typing import Callable
import numpy as np
from scipy.special import gammaincc, gamma
import xrb_units as xu

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

class XRB:
    def __init__(self, nchan: int = 10000, Lmin: float = 34, Lmax: float = 41, Emin: float = 0.05, Emax: float = 50.1) -> None:
        self.nchan      = nchan
        self.Lmin       = Lmin
        self.Lmax       = Lmax

        if self.Lmax < self.Lmin:
            raise ValueError("Lmax can't be smaller than Lmin!")

        if self.Lmin < 0:
            raise ValueError("Lmin can't be smaller than 0!")
        
        self.lumarr     = np.logspace(Lmin, Lmax, self.nchan)

        self.Emin       = Emin
        self.Emax       = Emax

        if self.Emax < self.Emin:
            raise ValueError("Emax can't be smaller than Emin!")

        if self.Emin < 0:
            raise ValueError("Emin can't be smaller than 0!")

        # THIS WORKS !!! Sadly, too convoluted. Simpler is to instanciate subclasses of 
        # XRB with model functions returning the complete model array
        # 
        # self.modelsL = {
        #     "Zh12"  : self.Zhang12,
        #     "0"     : self.Zhang12,
        #     0       : self.Zhang12
        # }
        
        # self.modelsH = {
        #     "Mi12S" : self.Mineo12S,
        #     "0"     : self.Mineo12S,
        #     0       : self.Mineo12S,
        #     "Mi12B" : self.Mineo12B,
        #     "1"     : self.Mineo12B,
        #     1       : self.Mineo12B,
        #     "Le21"  : self.Lehmer21,
        #     "2"     : self.Lehmer21,
        #     2       : self.Lehmer21,
        # }
    
    def calc_Zhang12(self, 
                  lum_in: float, K1: float, 
                  Lb1: float, Lb2: float, Lcut: float,
                  alpha1: float, alpha2: float, alpha3: float) -> float:
        """
        Analytic solution of broken Power-Law LMXB luminosity function (Zhang+12)\\
        Used for vectorization in model_Nlxb()
        -----
        lum_in      :   input luminosity in units of 1.e36 erg/s
        K1,K2,K3    :   Normalization in units of 1.e11 solar masses
        Lb1         :   First luminosity break in units of 1e36 erg/s
        Lb2         :   Second luminosity break in 1e36 erg/s
        Lcut        :   Luminosity cut-off in 1.e36 erg/s
        alpha1      :   Power-Law slope up to first break
        alpha2      :   Power-Law slope from first to second break
        alpha3      :   Power-Law slope from second break to cut-off
        """

        K2: float = self.Knorm(K1,Lb1,Lb2,alpha2)
        K3: float = self.Knorm(K2,Lb2,Lcut,alpha3)

        if lum_in < Lb1:
            return( K1*Lb1**alpha1 * ( lum_in**(1.-alpha1) - Lb1**(1.-alpha1) ) / (alpha1-1.) 
                + K2*Lb2**alpha2 * ( Lb1**(1.-alpha2)-Lb2**(1.-alpha2) ) / (alpha2-1.)
                + K3*Lcut**alpha3 * ( Lb2**(1.-alpha3)-Lcut**(1.-alpha3) ) / (alpha3-1.) 
                )

        elif lum_in >= Lb1 and lum_in < Lb2:
            return( K2*Lb2**alpha2 * ( lum_in**(1.-alpha2) - Lb2**(1.-alpha2) ) / (alpha2-1.)
                + K3*Lcut**alpha3 * ( Lb2**(1.-alpha3) - Lcut**(1.-alpha3) ) / (alpha3-1.)
                )

        elif lum_in >= Lb2 and lum_in < Lcut:
            return K3*Lcut**alpha3 * ( lum_in**(1.-alpha3) - Lcut**(1.-alpha3) ) / (alpha3-1.)
        
        else:
            return 0

    def calc_SPL(self, lum_in: float, 
                      xi: float, Lcut: float, gamma: float ) -> float:
        """
        Analytic solution of single Power-Law integral for luminosity functions\\
        Used for vectorization in model_Nhxb()
        -----
        lum_in      :   input luminosity in units of 1.e38 erg/s
        xi          :   normalization constant
        Lcut        :   Luminosity cut-off in 1.e38 erg/s
        gamma       :   Power-Law slope
        SFR         :   Star-Formation-Rate in units of Msun/yr\\
                        Used for rescaling of normalization
        """
        
        return xi/(gamma-1.)*((lum_in)**(1.-gamma)-(Lcut)**(1.-gamma))

    def calc_BPL(self, lum_in: float,
                      xi: float, Lb1: float, Lcut: float,
                      gamma1: float, gamma2: float ) -> float:
        """
        Analytic solution of broken Power-Law integral for luminosity functions\\
        Used for vectorization in model_Nhxb(). 
        -----
        lum_in      :   input luminosity in units of 1.e38 erg/s
        xi          :   normalization constant
        Lb1         :   luminosity break in units of 1.e38 erg/s
        gamma1      :   Power-Law slope uo to first
        Lcut        :   Luminosity cut-off in 1.e38 erg/s
        """

        if (lum_in < Lb1):
            return( xi * ( ( lum_in**(1.-gamma1) - Lb1**(1.-gamma1) )/(gamma1-1.)
                + Lb1**(gamma2-gamma1)*( Lb1**(1.-gamma2) - Lcut**(1.-gamma2) ) / (gamma2-1.) )
                )

        elif (lum_in >= Lb1) and (lum_in < Lcut):
            return xi * Lb1**(gamma2-gamma1)*(lum_in**(1.-gamma2) - Lcut**(1.-gamma2)) / (gamma2-1.)

    def calc_Lehmer21(self, lum_in: float,
                      A: float, Lb: float, logLc: float, logLc_logZ: float,
                      g1: float, g2: float, g2_logZ: float,
                      logOH12: float ) -> float:
        """
        Analytic solution of 'self.diff_Nhxb_met()' for metallicity enhanced HMXB LF in Lehmer+21\\
        Makes use of a custom implementation of the incomplete upper Gamma function

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
        slope1,2    :   redefined slopes for integration
        pre1,2      :   prefactor for first/second integration half respectively
        end         :   upper integration limit ~infinity
        """

        LcZ = 10**( logLc + logLc_logZ * ( logOH12 - 8.69 ) - 38 )
        Lb  = Lb / 1.e38

        g2Z = g2 + g2_logZ * ( logOH12 - 8.69 )

        slope1  = 1.-g1
        slope2  = 1.-g2Z
        pre1    = (LcZ**slope1)
        pre2    = (LcZ**slope2)*Lb**(g2Z-g1) 
        end     = 10**(self.Lmax-38)

        if lum_in < Lb:
            return( A * ( pre1 * ( GammaIncc(slope1,lum_in/LcZ) - GammaIncc(slope1,Lb/LcZ) )
                    + pre2 * ( GammaIncc(slope2,Lb/LcZ) - GammaIncc(slope2,end/LcZ)) )
                )
        else:
            return A * pre2 * ( GammaIncc(slope2,lum_in/LcZ) - GammaIncc(slope2,end/LcZ))

    def Knorm(self, K: float, L1: float, L2: float, alpha: float):
        """
        Calculates normalization for changing slopes
        -----
        K           :   normalization of previous slope
        L1          :   lower luminosity limit for slope range
        L2          :   higher luminosity limit for slope range 
        alpha       :   slope of the desired range
        """
        return ( K * ( L1 / L2 )**alpha )

    def par_rand(self, mu, sigma, size=None):
        return np.random.normal(mu,sigma,size)

class LMXB(XRB):
    
    def __init__(self, nchan: int = 10000, Lmin: float = 34, Lmax: float = 41, Emin: float = 0.05, Emax: float = 50.1) -> None:
        """
        Additionally initializes vectorized functions of underlying models
        """
        super().__init__(nchan=nchan, Lmin=Lmin, Lmax=Lmax, Emin=Emin, Emax=Emax)
        self.vec_calc_SPL = np.vectorize(super().calc_SPL)
        self.vec_calc_BPL = np.vectorize(super().calc_BPL)
        self.vec_calc_Zhang12 = np.vectorize(super().calc_Zhang12)

    def Zhang12(self, Mstar: float = 1., bRand: bool = False, bLum: bool = False) -> np.ndarray:
        """
        Initializes model parameters for LMXB LFs of Zhang+12
        returns array of either number of LMXBs > L or total luminosity
        -----
        Mstar       :   model scaling, as host-galaxy's stellar mass in units of 1.e11
        bRand       :   boolean switching between randomized parameters\\
                            according to their uncertainty
        bLum        :   boolean switch between returning cumulative number function or total 
                        luminosity. Since these are negative power laws, we modify input slope
                        by -1
        -----
        in function
        norm1       :   Normalization in units of 1.e11 solar masses
        Lb1         :   First luminosity break in units of 1e36 erg/s
        Lb2         :   Second luminosity break in 1e36 erg/s
        Lcut        :   Luminosity cut-off in 1.e36 erg/s
        alpha1      :   Power-Law slope up to first break
        alpha2      :   Power-Law slope from first to second break
        alpha3      :   Power-Law slope from second break to cut-off
        par         :   tuple of parameters loaded from xrb_units.py
        arr         :   return array
        """
        if Mstar < 0.:
            raise ValueError("Mstar can not be smaller than zero")

        if not bRand:
            par = (xu.norm1, xu.Lb1, xu.Lb2, xu.Lcut_L, xu.alpha1, xu.alpha2, xu.alpha3)
        else:
            par = ( self.par_rand(xu.norm1,xu.sig_K1),
                    self.par_rand(xu.Lb1,xu.sig_Lb1), 
                    self.par_rand(xu.Lb2,xu.sig_Lb2), 
                    xu.Lcut_L, # has no uncertainty given in Zhang+12
                    self.par_rand(xu.alpha1,xu.sig_a1), 
                    self.par_rand(xu.alpha2,xu.sig_a2), 
                    self.par_rand(xu.alpha3,xu.sig_a3)
                  )

        if bLum:
            par[5] -= 1
            par[6] -= 1
            par[7] -= 1
        
        arr = self.vec_calc_Zhang12(self.lumarr/1.e36, *par)

        return arr * Mstar

class HMXB(XRB):
    def __init__(self, nchan: int = 10000, Lmin: float = 34, Lmax: float = 41, Emin: float = 0.05, Emax: float = 50.1) -> None:
        """
        Additionally initializes vectorized functions of underlying models
        """
        super().__init__(nchan=nchan, Lmin=Lmin, Lmax=Lmax, Emin=Emin, Emax=Emax)
        self.vec_calc_SPL = np.vectorize(super().calc_SPL)
        self.vec_calc_BPL = np.vectorize(super().calc_BPL)
        self.vec_calc_Lehmer21 = np.vectorize(super().calc_Lehmer21)

    def Mineo12S(self, SFR: float = 1., bRand: bool = False, bLum: bool = False) -> np.ndarray:
        """
        Initializes model parameters for HMXB LFs of Mineo+12 single PL
        returns array of either number of HMXBs > L or total luminosity of HMXBs > L
        -----
        SFR         :   model scaling, as host-galaxy's star formation rate
                        in units of Msun/yr
        bRand       :   boolean switching between randomized parameters\\
                            according to their uncertainty
        bLum        :   boolean switch between returning cumulative number function or total 
                        luminosity. Since these are negative power laws, we modify input slope
                        by -1
        """
        if not bRand:
            par = (xu.xi_s, xu.Lcut_Hs, xu.gamma_s)
        else:
            par = ( 10**self.rand(xu.log_xi_s, xu.log_sig_xi_s), 
                    xu.Lcut_Hs,
                    self.rand(xu.gamma_s,xu.sig_gam_s)
                  )
        
        if bLum:
            par[2] -= 1

        arr = self.vec_calc_SPL(self.lumarr/1.e38, *par)

        return arr * SFR

    def Mineo12B(self, SFR: float = 1., bRand: bool = False, bLum: bool = False) -> np.ndarray:
        """
        Initializes model parameters for HMXB LFs of Mineo+12 broken PL
        returns array of either number of HMXBs > L or total luminosity of HMXBs > L
        -----
        SFR         :   model scaling, as host-galaxy's star formation rate
                        in units of Msun/yr
        bRand       :   boolean switching between randomized parameters\\
                            according to their uncertainty
        bLum        :   boolean switch between returning cumulative number function or total 
                        luminosity. Since these are negative power laws, we modify input slope
                        by -1
        """
        if not bRand:
            par = (xu.xi2_b, xu.LbH ,xu.Lcut_Hb, xu.gamma1_b, xu.gamma2_b)
        else:
            par = ( self.rand(xu.xi2_b, xu.sig_xi2),
                    self.rand(xu.LbH, xu.sig_LbH),
                    xu.Lcut_Hb,
                    self.rand(xu.gamma1_b, xu.sig_g1),
                    self.rand(xu.gamma2_b, xu.sig_g2)
                  )

        if bLum:
            par[-1] -= 1
            par[-2] -= 1
        
        arr = self.vec_calc_BPL(self.lumarr/1.e38, *par)

        return arr * SFR

        
    
    def Lehmer21(self, logOH12: float = 8.69, SFR: float = 1., bRand: bool = False, bLum: bool = False) -> np.ndarray:
        """
        Initializes model parameters for HMXB LFs of Mineo+12 single PL
        returns array of either number of HMXBs > L or total luminosity of HMXBs > L
        -----
        logOH12     :   metallicity measure, used to scale model parameters
                        refers to convention '12 + log(O/H)'
        SFR         :   model scaling, as host-galaxy's star formation rate
                        in units of Msun/yr
        bLum        :   boolean switch between returning cumulative number function or total 
                        luminosity. Since these are negative power laws, we modify input slope
                        by -1
        bRand       :   boolean switching between randomized parameters\\
                        according to their uncertainty. Does not randomize 
                        logOH12 as it is an external scaling parameter
        """
        if not bRand:
            par = ( xu.A_h, 10**xu.logLb, xu.logLc, xu.logLc_logZ, xu.g1_h, xu.g2_h, xu.g2_logZ )
        else:
            par = ( self.rand(xu.A_h, xu.sig_Ah),
                    10**self.rand(xu.logLb, xu.sig_logLb),
                    self.rand(xu.logLc,xu.sig_logLc),
                    self.rand(xu.logLc_logZ,xu.sig_logLcZ),
                    self.rand(xu.g1_h,xu.sig_g1h),
                    self.rand(xu.g2_h,xu.sig_g2h),
                    self.rand(xu.g2_logZ,xu.sig_g2logZ)
                )

        if bLum:
            par[4] -= 1
            par[5] -= 1
        
        arr = self.vec_calc_Lehmer21(self.lumarr/1.e38,*par,logOH12)

        return arr * SFR



if __name__ == "__main__":
    # x = XRB()
    import matplotlib.pyplot as plt
    # plt.plot(x.lumarr,x.model_Nhxb(),c='k',label='True')
    # plt.plot(x.lumarr,x.model_Nhxb(0,1,True))
    # plt.plot(x.lumarr,x.model_Nhxb(0,1,True))
    # plt.plot(x.lumarr,x.model_Nhxb(0,1,True))
    # plt.legend()
    # plt.show()

    import time
    import helper
    
    hxb = HMXB(Lmin=36)
    OH = [x for x in np.arange(7,9,0.2)]
    # Nhx = xrb.model_Nhxb(2)
    for oh in OH:
        plt.plot( np.log10(hxb.lumarr),np.log10(hxb.Lehmer21(logOH12=oh)),label=f'{oh:.1f}' )
        print(f'{oh:.1f}, {np.log10(hxb.lumarr[6000]):.2f}, {hxb.Lehmer21(logOH12=oh)[6000]:.2f}')
    plt.plot( np.log10(hxb.lumarr),np.log10(hxb.Mineo12S()),c='k',label='Mineo+12' )
    plt.xlim([36,41])
    plt.ylim([-2.5,2.5])
    plt.legend()
    plt.show()
    
    # print(f'{Nhx[2000]}, {Nhx[4000]}, {Nhx[6000]}')
    
