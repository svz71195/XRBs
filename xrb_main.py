import numpy as np
from numba import njit
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
    """
    Class structure conatining base functions for model generation. Also contains functions for
    model sampling and basic analysis functions. Provides the intrinsic luminosity array and
    sampling accuracy
    """
    def __init__(self, nchan: int = 10000, Lmin: float = 35, Lmax: float = 41, Emin: float = 0.05, Emax: float = 50.1) -> None:
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
        if lum_in < Lcut:
            return xi/(gamma-1.)*((lum_in)**(1.-gamma)-(Lcut)**(1.-gamma))
        else:
            return 0

    def calc_BPL(self, lum_in: float,
                      xi: float, Lb1: float, Lcut: float,
                      gamma1: float, gamma2: float ) -> float:
        """
        Analytic solution of broken Power-Law integral for luminosity functions\\
        Used for vectorization.
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
        
        else:
            return 0

    def calc_expSPL(self, lum_in: float,
                          norm: float, gamma: float, cut: float ) -> float:
        """
        Analytic solution of single Power-Law integral for luminosity functions with exponential
        cutoff. Used for vectorization. 
        -----
        lum_in      :   input luminosity in units of 1.e38 erg/s
        norm        :   normalization constant
        gamma       :   Power-Law slope
        cut         :   Luminosity cut-off in 1.e38 erg/s
        """

        slope   = 1 - gamma
        pre     = cut**slope
        end     = 10**(self.Lmax-38)

        return norm * pre * ( GammaIncc(slope,lum_in/cut) - GammaIncc(slope,end/cut) )

    def calc_Lehmer21(self, lum_in: float,
                      A: float, logLb: float, logLc: float, logLc_logZ: float,
                      g1: float, g2: float, g2_logZ: float,
                      logOH12: float) -> float:
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

        LcZ = 10**( logLc + ( logLc_logZ * ( logOH12 - 8.69 ) ) - 38 )
        Lb  = 10**(logLb - 38)

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

    @staticmethod
    def Knorm(K: float, L1: float, L2: float, alpha: float) -> float:
        """
        Calculates normalization for changing slopes
        -----
        K           :   normalization of previous slope
        L1          :   lower luminosity limit for slope range
        L2          :   higher luminosity limit for slope range 
        alpha       :   slope of the desired range
        """
        return ( K * ( L1 / L2 )**alpha )

    @staticmethod
    def par_rand(self, mu, sigma, size=None) -> float:
        """
        Randomizes input parameters of models in subclasses LMXB and HMXB
        """
        return np.random.normal(mu,sigma,size)

    @staticmethod
    def calc_pCDF(inp: np.ndarray) -> np.ndarray:
        """
        Calculates pseudo CDF from model input
        -----
        inp         :   model input obtained from subclasses
        """
        if isinstance(inp,np.ndarray):
            return 1. - inp/inp[0]
        elif isinstance(inp,[]):
            pCDF = []
            for a in inp:
                pCDF.append( 1 - a/inp[0] )
            return pCDF

    def sample(self, NXRB: int) -> np.ndarray:
        if len(self.model()) != len(self.lumarr):
            # should not happen if inp is generated in the scope of XRB()
            raise IndexError("Input array is not the same length as intrinsic luminosity")

        inpCDF = self.calc_pCDF(self.model())
        
        return self._sample(LumArr=self.lumarr,CDF=inpCDF,N=NXRB)

    @staticmethod
    @njit
    def _sample(LumArr: np.ndarray, CDF: np.ndarray, N: int) -> np.ndarray:
        """
        Wrapper function for self.sample(). Associates luminosities to population of XRBs 
        according to a given CDF. LumArr and CDF should have the same length
        -----
        LumArr      :   Input luminosity np.ndarray, usually passes with self.lumarr
        CDF         :   Input CDF derived from a XLF model. Should have the same length as
                            LumArr
        N           :   Population/Sample size. Needs to be integer
        """
        
        jj = 1
        kk = 1
        lum = np.zeros(N)
        ranvec = np.sort( np.random.rand(N) )
        
        for ii in range(0,N):
            jj = kk     # restart from jj where it arrived previously
            if jj == len(CDF) - 1:
                break   # otherwise, loop produces error due to overcounting
            while ( CDF[jj] < ranvec[ii] ):
                jj +=1
                if jj == len(CDF) - 1:
                    break
            kk == jj
            lum[ii] = LumArr[jj-1]+(LumArr[jj]-LumArr[jj-1])*(ranvec[ii]-CDF[jj-1])/(CDF[jj]-CDF[jj-1])
        return lum

    @staticmethod
    def count(samp: np.ndarray, lim: float) -> int:
        """
        Counts the number of XRBs with a luminosity greater than a certain value 'lim' from
        the given sample
        -----
        samp        :   Sample of XRBs as an list/np.ndarray containing their luminosities
        lim         :   Luminosity limit above which XRBs from samp are counted     
        """
        m = (samp >= lim)
        return len(samp[m])
    
    @staticmethod
    def lum_sum(samp: np.ndarray, lim: float = 1.e35) -> float:
        """
        Sums luminosities of XRBs in sample which have individual luminosities greater than 'lim'
        -----
        samp        :   Sample of XRBs as an list/np.ndarray containing their luminosities
        lim         :   Luminosity limit above which XRBs from samp are counted  
        """
        m = (samp >= lim)
        return np.sum(samp[m])


class Gilfanov04(XRB):

    normG                   = 440.4
    Lb1G                    = .19
    Lb2G                    = 5.
    a1G                     = 1.
    a2G                     = 1.86
    a3G                     = 4.8
    LcutG                   = 500

    def __init__(self, 
                 nchan: int = 10000, Lmin: float = 35, Lmax: float = 41, 
                 Emin: float = 0.05, Emax: float = 50.1, 
                 Mstar: float = 1.) -> None:
        super().__init__(nchan=nchan, Lmin=Lmin, Lmax=Lmax, Emin=Emin, Emax=Emax)
        if Mstar < 0:
            raise ValueError("Mstar can't be smaller than 0")
        self.Mstar = Mstar
        self.vec_calc_BPL = np.vectorize(super().calc_Zhang12)

    def model(self, bRand: bool = False, bLum: bool = False) -> np.ndarray:
        """
        Initializes model parameters for LMXB LFs of Gilfanov04
        returns array of either number of LMXBs > L or total luminosity
        -----
        Mstar       :   model scaling, as host-galaxy's stellar mass in units of 1.e11
        bRand       :   boolean switching between randomized parameters\\
                            according to their uncertainty
        bLum        :   boolean switch between returning cumulative number function or total 
                        luminosity. Since these are negative power laws, we modify input slope
                        by -1
        """

        if not bRand:
            par = ( self.normG, self.Lb1G, self.Lb2G, self.LcutG, self.a1G, self.a2G, self.a3G )
        else:
            par = ( self.normG, self.Lb1G, self.Lb2G, self.LcutG, self.a1G, self.a2G, self.a3G )

        if bLum:
            par = list(par)
            par[5] -= 1
            par[6] -= 1
            par[7] -= 1
            par = tuple(par)
        
        arr = self.vec_calc_Zhang12(self.lumarr/1.e38, *par)

        return arr * self.Mstar

class Zhang12(XRB):

    #--- slopes with error ---#
    alpha1: float           = 1.02  #+0.07 -0.08
    alpha2: float           = 2.06  #+0.06 -0.05
    alpha3: float           = 3.63  #+0.67 -0.49
    sig_a1: float           = 0.075
    sig_a2: float           = 0.055
    sig_a3: float           = 0.58
    #--- luminosity breaks in units of 1.e36 erg/s---#
    Lb1: float              = 54.6  #+4.3 -3.7
    Lb2: float              = 599.  #+95 -67
    Lcut_L: float           = 5.e4
    sig_Lb1: float          = 4.e-2
    sig_Lb2: float          = (95+67)/2
    #--- normalization in units of 1.e11 Msol ---#
    norm1: float            = 1.01  #+-0.28; per 10^11 solar masses
    sig_K1: float           = 0.28

    def __init__(self, 
                 nchan: int = 10000, Lmin: float = 35, Lmax: float = 41, 
                 Emin: float = 0.05, Emax: float = 50.1, 
                 Mstar: float = 1.) -> None:
        super().__init__(nchan=nchan, Lmin=Lmin, Lmax=Lmax, Emin=Emin, Emax=Emax)
        if Mstar < 0:
            raise ValueError("Mstar can't be smaller than 0")
        self.Mstar = Mstar
        self.vec_calc_BPL = np.vectorize(super().calc_Zhang12)

    def model(self, bRand: bool = False, bLum: bool = False) -> np.ndarray:
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
    
        if not bRand:
            par = (self.norm1, self.Lb1, self.Lb2, self.Lcut_L, self.alpha1, self.alpha2, self.alpha3)
        else:
            par = ( self.par_rand(self.norm1, self.sig_K1),
                    self.par_rand(self.Lb1, self.sig_Lb1), 
                    self.par_rand(self.Lb2, self.sig_Lb2), 
                    self.Lcut_L, # has no uncertainty given in Zhang+12
                    self.par_rand(self.alpha1, self.sig_a1), 
                    self.par_rand(self.alpha2, self.sig_a2), 
                    self.par_rand(self.alpha3, self.sig_a3)
                  )

        if bLum:
            par = list(par)
            par[5] -= 1
            par[6] -= 1
            par[7] -= 1
            par = tuple(par)
        
        arr = self.vec_calc_Zhang12(self.lumarr/1.e36, *par)

        return arr * self.Mstar

class Lehmer19L(XRB):

    norm2: float            = 33.8
    alph1: float            = 1.28
    alph2: float            = 2.33
    bre: float              = 1.48
    cut: float              = 10**2.7 # (40.7 -38)

    sig_norm2: float        = 5.
    sig_alph1: float        = 0.06
    sig_alph2: float        = 0.24
    sig_bre: float          = 0.68
    sig_cut: float          = 0.3

    def __init__(self, 
                 nchan: int = 10000, Lmin: float = 35, Lmax: float = 41, 
                 Emin: float = 0.05, Emax: float = 50.1, 
                 Mstar: float = 1.) -> None:
        super().__init__(nchan=nchan, Lmin=Lmin, Lmax=Lmax, Emin=Emin, Emax=Emax)
        if Mstar < 0:
            raise ValueError("Mstar can't be smaller than 0")
        self.Mstar = Mstar
        self.vec_calc_BPL = np.vectorize(super().calc_BPL)
        
    def model(self, bRand: bool = False, bLum: bool = False) -> np.ndarray:
        """
        Initializes model parameters for Lehmer+19 LMXB LFs based on BPL
        returns array of either number of LMXBs > L or total luminosity of HMXBs > L
        -----
        Mstar       :   model scaling, as host-galaxy's stellar mass in units of 1.e11
        bRand       :   boolean switching between randomized parameters\\
                            according to their uncertainty
        bLum        :   boolean switch between returning cumulative number function or total 
                        luminosity. Since these are negative power laws, we modify input slope
                        by -1
        """
        if not bRand:
            par = ( self.norm2, self.bre, self.cut, self.alph1, self.alph2 )
        else:
            par = ( self.par_rand(self.norm2, self.sig_norm2), 
                    self.par_rand(self.bre, self.sig_bre), 
                    self.par_rand(self.cut, self.sig_cut), 
                    self.par_rand(self.alph1, self.sig_alph1), 
                    self.par_rand(self.alph2, self.sig_alph2) )
        
        if bLum:
            par = list(par)
            par[4] = par[4] - 1
            par[5] = par[5] - 1
            par = tuple(par)

        arr = self.vec_calc_BPL(self.lumarr/1.e38, *par)

        return arr * self.Mstar

class Lehmer20(XRB):

    # GC
    K_GC                    = 8.08
    gamma_GC                = 1.08
    cut_GC                  = 10**.61 #(38.61 - 38)

    # field (no priors)
    K_field                 = 42.4
    a1_field                = 0.98
    Lb_field                = 0.45
    a2_field                = 2.43
    cut_field               = 100

    K_seed                  = 5.
    gamma_seed              = 1.21
    cut_seed                = 10**.66 #(38.66-38)

    def __init__(self, 
                 nchan: int = 10000, Lmin: float = 35, Lmax: float = 41, 
                 Emin: float = 0.05, Emax: float = 50.1, 
                 Mstar: float = 1., Sn: float = 1.) -> None:
        super().__init__(nchan=nchan, Lmin=Lmin, Lmax=Lmax, Emin=Emin, Emax=Emax)
        if Mstar < 0:
            raise ValueError("Mstar can't be smaller than 0")
        self.Mstar = Mstar
        if Sn < 0:
            raise ValueError("Sn can't be smaller than 0")
        self.Sn = Sn
        self.vec_calc_BPL       = np.vectorize(super().calc_BPL)
        self.vec_calc_expSPL    = np.vectorize(super().calc_expSPL)

    def model(self, bRand: bool = False, bLum: bool = False) -> np.ndarray:
        """
        Initializes model parameters for Lehmer+20 LMXB LFs based on BPL and exponential GC seeding
        returns array of either number of HMXBs > L or total luminosity of HMXBs > L
        Combination of in-situ LMXBs and GC seeded (col 4 and 5 in table 8 of Lehmer+20)
        -----
        Mstar       :   model scaling, as host-galaxy's stellar mass in units of 1.e11
        Sn          :   Specific frequency of globular clusters in galaxy
        bRand       :   boolean switching between randomized parameters\\
                            according to their uncertainty
        bLum        :   boolean switch between returning cumulative number function or total 
                        luminosity. Since these are negative power laws, we modify input slope
                        by -1
        """
        if not bRand:
            par_field = (self.K_field, self.Lb_field, self.cut_field, self.a1_field, self.a2_field)
            par_seed  = (self.K_seed, self.gamma_seed, self.cut_seed)
            par_GC    = (self.K_GC, self.gamma_GC, self.cut_GC)
        else:
            par_field = (self.K_field, self.Lb_field, self.cut_field, self.a1_field, self.a2_field)
            par_seed  = (self.K_seed, self.gamma_seed, self.cut_seed)
            par_GC    = (self.K_GC, self.gamma_GC, self.cut_GC)

        if bLum:
            par_field = list(par_field)
            par_seed  = list(par_seed)
            par_GC    = list(par_GC)
            par_field[4] = par_field[4] - 1
            par_field[5] = par_field[5] - 1
            par_seed[1] = par_seed[1] - 1
            par_GC[1] = par_GC[1] - 1
            par_field = tuple(par_field)
            par_seed  = tuple(par_seed)
            par_GC    = tuple(par_GC)

        arr1 = self.vec_calc_BPL(self.lumarr/1.e38, *par_field) 
        arr2 = self.vec_calc_expSPL(self.lumarr/1.e38, *par_seed) 
        arr3 = self.vec_calc_expSPL(self.lumarr/1.e38, *par_GC)

        return self.Mstar * ( arr1 + self.Sn * (arr2 + arr3) )

class Grimm03(XRB):

    norm: float             = 3.3
    gamma: float            = 1.61
    Lcut: float             = 240

    def __init__(self, 
                 nchan: int = 10000, Lmin: float = 35, Lmax: float = 41, 
                 Emin: float = 0.05, Emax: float = 50.1, 
                 SFR: float = 1.) -> None:
        super().__init__(nchan=nchan, Lmin=Lmin, Lmax=Lmax, Emin=Emin, Emax=Emax)
        if SFR < 0:
            raise ValueError("SFR can't be smaller than 0")
        self.SFR = SFR
        self.vec_calc_SPL = np.vectorize(super().calc_SPL)

    def model(self, bRand: bool = False, bLum: bool = False) -> np.ndarray:
        """
        Initializes model parameters for HMXB LFs of Grimm+03 single PL
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
            par = (self.norm, self.Lcut, self.gamma)
        else:
            par = ( 10**self.par_rand(xu.log_xi_s, xu.log_sig_xi_s), 
                    xu.Lcut_Hs,
                    self.par_rand(xu.gamma_s,xu.sig_gam_s)
                  )
        
        if bLum:
            par = list(par)
            par[2] -= 1
            par = tuple(par)

        arr = self.vec_calc_SPL(self.lumarr/1.e38, *par)

        return arr * SFR

class Mineo12S(XRB):

    Lcut: float             = 1.e3  # in units of 1.e38 erg/s
    gamma: float            = 1.59  #+-0.25 (rms, Mineo 2012)
    xi: float               = 1.88  #*/ 10^(0.34) (rms=0.34 dex, Mineo 2012)

    log_xi: float           = 0.27415 #log10
    sig_gam: float          = 0.25
    log_sig_xi: float       = 0.34

    def __init__(self, 
                 nchan: int = 10000, Lmin: float = 35, Lmax: float = 41, 
                 Emin: float = 0.05, Emax: float = 50.1, 
                 SFR: float = 1.) -> None:
        super().__init__(nchan=nchan, Lmin=Lmin, Lmax=Lmax, Emin=Emin, Emax=Emax)
        if SFR < 0:
            raise ValueError("SFR can't be smaller than 0")
        self.SFR = SFR
        self.vec_calc_SPL = np.vectorize(super().calc_SPL)

    def model(self, bRand: bool = False, bLum: bool = False) -> np.ndarray:
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
            par = ( 10**self.par_rand(self.log_xi_s, self.log_sig_xi_s), 
                    self.Lcut_Hs,
                    self.par_rand(self.gamma_s, self.sig_gam_s)
                  )
        
        if bLum:
            par = list(par)
            par[2] -= 1
            par = tuple(par)

        arr = self.vec_calc_SPL(self.lumarr/1.e38, *par)

        return arr * self.SFR

class Mineo12B(XRB):

    Lcut: float             = 5.e3
    LbH: float              = 110.

    gamma1: float           = 1.58
    gamma2: float           = 2.73
    xi2: float              = 1.49

    sig_Lb: float           = (57+34)/2
    sig_g1: float           = 0.02
    sig_g2: float           = (1.58+0.54)/2
    sig_xi2: float          = 0.07

    def __init__(self, 
                 nchan: int = 10000, Lmin: float = 35, Lmax: float = 41, 
                 Emin: float = 0.05, Emax: float = 50.1, 
                 SFR: float = 1.) -> None:
        super().__init__(nchan=nchan, Lmin=Lmin, Lmax=Lmax, Emin=Emin, Emax=Emax)
        if SFR < 0:
            raise ValueError("SFR can't be smaller than 0")
        self.SFR = SFR
        self.vec_calc_BPL = np.vectorize(super().calc_BPL)

    def model(self, bRand: bool = False, bLum: bool = False) -> np.ndarray:
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
            par = (self.xi2, self.LbH, self.Lcut, self.gamma1, self.gamma2)
        else:
            par = ( self.par_rand(self.xi2, self.sig_xi2),
                    self.par_rand(self.LbH, self.sig_LbH),
                    self.Lcut,
                    self.par_rand(self.gamma1_b, self.sig_g1),
                    self.par_rand(self.gamma2_b, self.sig_g2)
                  )

        if bLum:
            par = list(par)
            par[-1] -= 1
            par[-2] -= 1
            par = tuple(par)
        
        arr = self.vec_calc_BPL(self.lumarr/1.e38, *par)

        return arr * self.SFR

class Lehmer19H(XRB):

    norm: float             = 1.96
    gam: float              = 1.65
    cut: float              = 10**2.7 # (40.7 -38)

    sig_norm: float         = 0.14
    sig_gam: float          = 0.025
    sig_cut: float          = 0.3

    def __init__(self, 
                 nchan: int = 10000, Lmin: float = 35, Lmax: float = 41, 
                 Emin: float = 0.05, Emax: float = 50.1, 
                 SFR: float = 1.) -> None:
        super().__init__(nchan=nchan, Lmin=Lmin, Lmax=Lmax, Emin=Emin, Emax=Emax)
        if SFR < 0:
            raise ValueError("SFR can't be smaller than 0")
        self.SFR = SFR
        self.vec_calc_SPL = np.vectorize(super().calc_SPL)

    def model(self, bRand: bool = False, bLum: bool = False) -> np.ndarray:
        """
        Initializes model parameters for HMXB LFs of Lehmer+19 single PL
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
            par = ( self.norm, self.cut, self.gam )
        else:
            par = ( self.par_rand(self.norm, self.sig_norm), 
                    self.par_rand(self.cut, self.sig_cut),
                    self.par_rand(self.gam, self.sig_gam) )

        if bLum:
            par = list(par)
            par[3] = par[3] - 1
            par = tuple(par)

        arr = self.vec_calc_SPL(self.lumarr/1.e38, *par)

        return arr * self.SFR

class Lehmer21(XRB):

    A_h: float              = 1.29
    g1_h: float             = 1.74
    g2_h: float             = 1.16
    g2_logZ: float          = 1.215 # actually 1.34, tweaked to better reflect skewness of parameter
    logLb: float            = 38.54
    logLc: float            = 39.98
    logLc_logZ: float       = 0.51 # actually 0.6, tweaked to better reflect skewness of parameter

    sig_Ah: float           = 0.185
    sig_g1h: float          = 0.04
    sig_g2h: float          = 0.17
    sig_g2logZ: float       = 0.5
    sig_logLb: float        = 0.2
    sig_logLc: float        = 0.24
    sig_logLcZ: float       = 0.3

    def __init__(self, 
                 nchan: int = 10000, Lmin: float = 35, Lmax: float = 41, 
                 Emin: float = 0.05, Emax: float = 50.1, 
                 logOH12: float = 8.69, SFR: float = 1.) -> None:
        super().__init__(nchan=nchan, Lmin=Lmin, Lmax=Lmax, Emin=Emin, Emax=Emax)
        self.logOH12 = logOH12
        if SFR < 0:
            raise ValueError("SFR can't be smaller than 0")
        self.SFR = SFR
        self.vec_calc_Lehmer21 = np.vectorize(super().calc_Lehmer21)

    def model(self, bRand: bool = False, bLum: bool = False) -> np.ndarray:
        """
        Initializes model parameters for HMXB LFs of Lehmer+21
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
            par = ( self.A_h, self.logLb, self.logLc, self.logLc_logZ, self.g1_h, self.g2_h, self.g2_logZ )
        else:
            par = ( self.par_rand(self.A_h, self.sig_Ah),
                    self.par_rand(self.logLb, self.sig_logLb),
                    self.par_rand(self.logLc, self.sig_logLc),
                    self.par_rand(self.logLc_logZ, self.sig_logLcZ),
                    self.par_rand(self.g1_h, self.sig_g1h),
                    self.par_rand(self.g2_h, self.sig_g2h),
                    self.par_rand(self.g2_logZ, self.sig_g2logZ)
                )

        if bLum:
            par = list(par)
            par[4] = par[4] - 1.
            par[5] = par[5] - 1.
            par = tuple(par)
        
        arr = self.vec_calc_Lehmer21( self.lumarr/1.e38, *par, self.logOH12 )

        return arr * self.SFR


import itertools

def model_err(mod: np.ufunc, LumArr: np.ndarray, args: tuple, logOH12: float) -> tuple:
    """
    input args must have the form ((a,aL,au),(b,bL,bU),...) in order to retain the parameter ordering
    for the underlying model
    """
    comb: tuple = itertools.product(*args) # tuple of tuples
    calc = []
    for combi in comb:
        try:
            calc.append( mod( LumArr,*combi, logOH12 ) )
        except TypeError:
            calc.append( mod(lum_in=LumArr,*combi) )
    
    print(len(calc))
    print(len(calc[0]))
    errU = np.zeros(len(calc[0]))
    errL = np.zeros(len(calc[0]))
    for i in range(len(calc[0])):
        errU[i] =  np.amax( [z[i] for z in calc] )
        errL[i] =  np.amin( [z[i] for z in calc] )
    return (errU, errL)




if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    import time
    import helper


    hxb = XRB(Lmin=35,Lmax=41,nchan=10000)
    Li = np.array([7,7.2,7.4,7.6,7.8,8.,8.2,8.4,8.6,8.8,9.,9.2])
    Lo = [4.29,4.27,3.97,3.47,2.84,2.2,1.62,1.15,0.8,0.54,0.37,0.25]
    Loerr = [[3.07,2.53,1.9,1.33,.88,.54,.31,.18,.15,.13,.11,.09],[7.63,5.09,3.26,2.01,1.19,0.68,0.37,0.21,0.17,0.17,0.16,0.15]]
    Loerr2 = [[1.45,1.42,1.17,.85,.55,.33,.28,.1,.08,.06,.05,.03],[5.64,3.76,2.38,1.42,.81,0.44,0.23,0.13,0.09,0.08,0.07,0.06]]
    Lo2 = [1.60,1.80,1.83,1.68,1.41,1.09,0.77,0.51,.32,.2,.11,.07]
    Lu = [40.21,40.25,40.25,40.22,40.16,40.06,39.94,39.8,39.64,39.49,39.34,39.21]
    Luerr = [[.66,.5,.38,.28,.2,.15,.11,.09,.1,.12,.13,.12],[.69,.53,.4,.29,.21,.15,.12,.1,.11,.13,.15,.16]]
    OH = np.linspace(7,9.2,12)
    SFR = [0.01,0.1,1,10,100]
    par = ( xu.A_h, xu.logLb, xu.logLc, xu.logLc_logZ-0.09, xu.g1_h, xu.g2_h, xu.g2_logZ-.125 )
    N39 = np.vectorize(hxb.calc_Lehmer21)

    import tqdm
    
    errU = np.zeros((len(OH),len(SFR)))
    errL = np.zeros((len(OH),len(SFR)))
    errm = np.zeros((len(OH),len(SFR)))
    # for k,oh in enumerate(tqdm.tqdm(OH)):
    #     print(oh, np.log10(hxb.Lehmer21(logOH12=oh,bLum=True)[0])+38)
    #     for j,sfr in enumerate(SFR):
    #         h = hxb.Lehmer21(logOH12=oh,SFR=sfr)
    #         L = np.zeros(1000)
    #         N = int(h[0])
    #         for i in range(1000):
    #             dist = hxb.sample(h,N)
    #             L[i] = hxb.lum_sum(dist) / sfr
    #         print(np.log10(np.median(L)),np.log10(np.percentile(L,84)),np.log10(np.percentile(L,16)))
    #         errU[k][j] = np.percentile(L,84)
    #         errL[k][j] = np.percentile(L,16)
    #         errm[k][j] = np.median(L)
    
    L39U = np.median(errU,axis=1)
    L39L = np.median(errL,axis=1)
    L39m = np.median(errm,axis=1) 

    fig, ax = plt.subplots(figsize=(12,10))
    line, = plt.plot(OH, N39(10, *par, logOH12=OH), label='N39, analytic integration' )
    line2, = plt.plot(OH, N39(31.62, *par, logOH12=OH), label='N39.5, analytic integration' )
    lineM = plt.errorbar(Li, Lo, yerr=Loerr, ms=8.,capsize=3.,capthick=1.,c='k',fmt='x',label='N39, Lehmer+21')
    lineM2 = plt.errorbar(Li, Lo2, yerr=Loerr2, ms=8.,capsize=3.,capthick=1.,c='r',fmt='x',label='N39.5, Lehmer+21')
    lineM2[-1][0].set_linestyle('--')
    # plt.plot(OH,ssss)
    ax.set_xlabel(r'$12+\log[O/H]$',fontsize=12)
    ax.set_ylabel(r'$N(L>L_{X})$',fontsize=12)

    plt.subplots_adjust(bottom=0.5,left=0.05,right=0.95,top=0.95)

    axAh = plt.axes([0.1, 0.42, 0.65, 0.02])
    axLb = plt.axes([0.1, 0.37, 0.65, 0.02])
    axLc = plt.axes([0.1, 0.32, 0.65, 0.02])
    axLz = plt.axes([0.1, 0.27, 0.65, 0.02])
    axg1 = plt.axes([0.1, 0.22, 0.65, 0.02])
    axg2 = plt.axes([0.1, 0.17, 0.65, 0.02])
    axgz = plt.axes([0.1, 0.12, 0.65, 0.02])
    axOH = plt.axes([0.1, 0.07, 0.65, 0.02])
    Ah_slider = Slider(ax=axAh, label=r'$A_{HMXB}$', valmin=0.1, valmax=6, valinit=par[0])
    Lb_slider = Slider(ax=axLb, label=r'$\log{L_b}$', valmin=37, valmax=40, valinit=par[1])
    Lc_slider = Slider(ax=axLc, label=r'$\log{L_c}$', valmin=39, valmax=41, valinit=par[2])
    Lz_slider = Slider(ax=axLz, label=r'$d\log{L_c}/d\log{Z}$', valmin=0.1, valmax=2, valinit=par[3])
    g1_slider = Slider(ax=axg1, label=r'$\gamma_1$', valmin=0.1, valmax=3, valinit=par[4])
    g2_slider = Slider(ax=axg2, label=r'$\gamma_2$', valmin=0.1, valmax=3, valinit=par[5])
    gz_slider = Slider(ax=axgz, label=r'$d\gamma_2/d\log{Z}$', valmin=0.1, valmax=3, valinit=par[6])
    OH_slider = Slider(ax=axOH, label=r'$OH$ offset', valmin=-1, valmax=1, valinit=0)

    def update(val):
        line.set_ydata(N39(10, Ah_slider.val, (Lb_slider.val), Lc_slider.val, Lz_slider.val, g1_slider.val, g2_slider.val, gz_slider.val, OH+OH_slider.val))
        line2.set_ydata(N39(31.62, Ah_slider.val, (Lb_slider.val), Lc_slider.val, Lz_slider.val, g1_slider.val, g2_slider.val, gz_slider.val, OH+OH_slider.val))
        fig.canvas.draw_idle()

    Ah_slider.on_changed(update)
    Lb_slider.on_changed(update)
    Lc_slider.on_changed(update)
    Lz_slider.on_changed(update)
    g1_slider.on_changed(update)
    g2_slider.on_changed(update)
    gz_slider.on_changed(update)
    OH_slider.on_changed(update)

    ax.legend(fontsize=11)

    plt.show()  

    par2 = ( xu.A_h, xu.logLb, xu.logLc, xu.logLc_logZ-.09, xu.g1_h-1, xu.g2_h-1, xu.g2_logZ-.125 )
    fig3, ax3 = plt.subplots(figsize=(12,10))
    linen, = plt.plot(OH, np.log10(N39(.001, *par2, logOH12=OH))+38, label='total L, analytic integration' )
    lineN = plt.errorbar(Li, Lu, yerr=Luerr, ms=8.,capsize=3.,capthick=1.,c='k',fmt='x',label='total Lum, Lehmer+21')
    ax3.set_xlabel(r'$12+\log[O/H]$',fontsize=12)
    ax3.set_ylabel(r'$\log(L_X / \mathrm{SFR})$',fontsize=12)

    plt.subplots_adjust(bottom=0.5,left=0.05,right=0.95,top=0.95)

    axAh = plt.axes([0.1, 0.42, 0.65, 0.02])
    axLb = plt.axes([0.1, 0.37, 0.65, 0.02])
    axLc = plt.axes([0.1, 0.32, 0.65, 0.02])
    axLz = plt.axes([0.1, 0.27, 0.65, 0.02])
    axg1 = plt.axes([0.1, 0.22, 0.65, 0.02])
    axg2 = plt.axes([0.1, 0.17, 0.65, 0.02])
    axgz = plt.axes([0.1, 0.12, 0.65, 0.02])
    axOH = plt.axes([0.1, 0.07, 0.65, 0.02])
    Ah_slider = Slider(ax=axAh, label=r'$A_{HMXB}$', valmin=0.1, valmax=6, valinit=par2[0])
    Lb_slider = Slider(ax=axLb, label=r'$\log{L_b}$', valmin=37, valmax=40, valinit=par2[1])
    Lc_slider = Slider(ax=axLc, label=r'$\log{L_c}$', valmin=39, valmax=41, valinit=par2[2])
    Lz_slider = Slider(ax=axLz, label=r'$d\log{L_c}/d\log{Z}$', valmin=0.1, valmax=2, valinit=par2[3])
    g1_slider = Slider(ax=axg1, label=r'$\gamma_1$', valmin=0.1, valmax=3, valinit=par2[4])
    g2_slider = Slider(ax=axg2, label=r'$\gamma_2$', valmin=0., valmax=.5, valinit=par2[5])
    gz_slider = Slider(ax=axgz, label=r'$d\gamma_2/d\log{Z}$', valmin=0.1, valmax=3, valinit=par2[6])
    OH_slider = Slider(ax=axOH, label=r'$OH$ offset', valmin=-1, valmax=1, valinit=0)

    def update3(val):
        linen.set_ydata(np.log10(N39(.001, Ah_slider.val, (Lb_slider.val), Lc_slider.val, Lz_slider.val, g1_slider.val, g2_slider.val, gz_slider.val, OH+OH_slider.val))+38)
        fig3.canvas.draw_idle()

    Ah_slider.on_changed(update3)
    Lb_slider.on_changed(update3)
    Lc_slider.on_changed(update3)
    Lz_slider.on_changed(update3)
    g1_slider.on_changed(update3)
    g2_slider.on_changed(update3)
    gz_slider.on_changed(update3)
    OH_slider.on_changed(update3)

    ax3.legend(fontsize=11)

    plt.show()  
