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

    def Knorm(self, K: float, L1: float, L2: float, alpha: float) -> float:
        """
        Calculates normalization for changing slopes
        -----
        K           :   normalization of previous slope
        L1          :   lower luminosity limit for slope range
        L2          :   higher luminosity limit for slope range 
        alpha       :   slope of the desired range
        """
        return ( K * ( L1 / L2 )**alpha )

    def par_rand(self, mu, sigma, size=None) -> float:
        """
        Randomizes input parameters of models in subclasses LMXB and HMXB
        """
        return np.random.normal(mu,sigma,size)

    def calc_pCDF(self, inp: np.ndarray) -> np.ndarray:
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

    def sample(self, inp: np.ndarray, NXRB: int) -> np.ndarray:
        if len(inp) != len(self.lumarr):
            # should not happen if inp is generated in the scope of XRB()
            raise IndexError("Input array is not the same length as intrinsic luminosity")

        inpCDF = self.calc_pCDF(inp)
        
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

    def count(self, samp: np.ndarray, lim: float) -> int:
        """
        Counts the number of XRBs with a luminosity greater than a certain value 'lim' from
        the given sample
        -----
        samp        :   Sample of XRBs as an list/np.ndarray containing their luminosities
        lim         :   Luminosity limit above which XRBs from samp are counted     
        """
        m = (samp >= lim)
        return len(samp[m])
    
    def lum_sum(self, samp: np.ndarray, lim: float = 1.e35) -> float:
        """
        Sums luminosities of XRBs in sample which have individual luminosities greater than 'lim'
        -----
        samp        :   Sample of XRBs as an list/np.ndarray containing their luminosities
        lim         :   Luminosity limit above which XRBs from samp are counted  
        """
        m = (samp >= lim)
        return np.sum(samp[m])

class LMXB(XRB):
    
    def __init__(self, nchan: int = 10000, Lmin: float = 34, Lmax: float = 41, Emin: float = 0.05, Emax: float = 50.1) -> None:
        """
        Additionally initializes vectorized functions of underlying models
        """
        super().__init__(nchan=nchan, Lmin=Lmin, Lmax=Lmax, Emin=Emin, Emax=Emax)
        self.vec_calc_SPL       = np.vectorize(super().calc_SPL)
        self.vec_calc_BPL       = np.vectorize(super().calc_BPL)
        self.vec_calc_expSPL    = np.vectorize(super().calc_expSPL)
        self.vec_calc_Zhang12   = np.vectorize(super().calc_Zhang12)

    def Gilfanov04(self, Mstar: float = 1., bRand: bool = False, bLum: bool = False) -> np.ndarray:
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

        if Mstar < 0.:
            raise ValueError("Mstar can not be smaller than zero")

        if not bRand:
            par = ( xu.normG, xu.Lb1G, xu.Lb2G, xu.LcutG, xu.a1G, xu.a2G, xu.a3G )
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
            par = list(par)
            par[5] -= 1
            par[6] -= 1
            par[7] -= 1
            par = tuple(par)
        
        arr = self.vec_calc_Zhang12(self.lumarr/1.e38, *par)

        return arr * Mstar


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
            par = list(par)
            par[5] -= 1
            par[6] -= 1
            par[7] -= 1
            par = tuple(par)
        
        arr = self.vec_calc_Zhang12(self.lumarr/1.e36, *par)

        return arr * Mstar

    def Lehmer19(self, Mstar: float = 1., bRand: bool = False, bLum: bool = False) -> np.ndarray:
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

        if Mstar < 0.:
            raise ValueError("Mstar can not be smaller than zero")

        if not bRand:
            par = ( xu.norm2, xu.bre, xu.cut, xu.alph1, xu.alph2 )
        else:
            par = ( self.par_rand(xu.norm2,xu.sig_norm2), 
                    self.par_rand(xu.bre, xu.sig_bre), 
                    self.par_rand(xu.cut, xu.sig_cut), 
                    self.par_rand(xu.alph1, xu.sig_alph1), 
                    self.par_rand(xu.alph2, xu.sig_alph2) )
        
        if bLum:
            par = list(par)
            par[4] = par[4] - 1
            par[5] = par[5] - 1
            par = tuple(par)

        arr = self.vec_calc_BPL(self.lumarr/1.e38, *par)

        return arr * Mstar

    def Lehmer20(self, Mstar: float = 1, Sn: float = 1., bRand: bool = False, bLum: bool = False) -> np.ndarray:
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

        if Mstar < 0.:
            raise ValueError("Mstar can not be smaller than zero")

        if not bRand:
            par_field = (xu.K_field, xu.Lb_field, xu.cut_field, xu.a1_field, xu.a2_field)
            par_seed  = (xu.K_seed, xu.gamma_seed, xu.cut_seed)
            par_GC    = (xu.K_GC, xu.gamma_GC, xu.cut_GC)
        else:
            par_field = (xu.K_field, xu.Lb_field, xu.cut_field, xu.a1_field, xu.a2_field)
            par_seed  = (xu.K_seed, xu.gamma_seed, xu.cut_seed)
            par_GC    = (xu.K_GC, xu.gamma_GC, xu.cut_GC)

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

        return Mstar * ( arr1 + Sn * (arr2 + arr3) )

class HMXB(XRB):
    def __init__(self, nchan: int = 10000, Lmin: float = 34, Lmax: float = 41, Emin: float = 0.05, Emax: float = 50.1) -> None:
        """
        Additionally initializes vectorized functions of underlying models
        """
        super().__init__(nchan=nchan, Lmin=Lmin, Lmax=Lmax, Emin=Emin, Emax=Emax)
        self.vec_calc_SPL = np.vectorize(super().calc_SPL)
        self.vec_calc_BPL = np.vectorize(super().calc_BPL)
        self.vec_calc_Lehmer21 = np.vectorize(super().calc_Lehmer21)

    def Grimm03(self, SFR: float = 1., bRand: bool = False, bLum: bool = False) -> np.ndarray:
        """
        Initializes model parameters for HMXB LFs of Grimm+12 single PL
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
        if SFR < 0.:
            raise ValueError("SFR can not be smaller than zero")

        if not bRand:
            par = (xu.norm_Gr, xu.Lcut_Gr, xu.gamma_Gr)
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

        if SFR < 0.:
            raise ValueError("SFR can not be smaller than zero")

        if not bRand:
            par = (xu.xi_s, xu.Lcut_Hs, xu.gamma_s)
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

        if SFR < 0.:
            raise ValueError("SFR can not be smaller than zero")

        if not bRand:
            par = (xu.xi2_b, xu.LbH ,xu.Lcut_Hb, xu.gamma1_b, xu.gamma2_b)
        else:
            par = ( self.par_rand(xu.xi2_b, xu.sig_xi2),
                    self.par_rand(xu.LbH, xu.sig_LbH),
                    xu.Lcut_Hb,
                    self.par_rand(xu.gamma1_b, xu.sig_g1),
                    self.par_rand(xu.gamma2_b, xu.sig_g2)
                  )

        if bLum:
            par = list(par)
            par[-1] -= 1
            par[-2] -= 1
            par = tuple(par)
        
        arr = self.vec_calc_BPL(self.lumarr/1.e38, *par)

        return arr * SFR

    def Lehmer19(self, SFR: float = 1., bRand: bool = False, bLum: bool = False) -> np.ndarray:
        """
        Single PL model parameters for HMXB LF of Lehmer+19
        returns array of either number of HMXBs > L or total luminosity of HMXBs > L
        -----
        SFR         :   model scaling, as host-galaxy's star formation rate
                        in units of Msun/yr
        bLum        :   boolean switch between returning cumulative number function or total 
                        luminosity. Since these are negative power laws, we modify input slope
                        by -1
        bRand       :   boolean switching between randomized parameters\\
                        according to their uncertainty. Does not randomize 
                        logOH12 as it is an external scaling parameter
        """

        if SFR < 0.:
            raise ValueError("SFR can not be smaller than zero")

        if not bRand:
            par = ( xu.norm3, xu.cut, xu.gam )
        else:
            par = ( self.par_rand(xu.norm3,xu.sig_norm3), 
                    self.par_rand(xu.cut, xu.sig_cut),
                    self.par_rand(xu.gam, xu.sig_gam) )

        if bLum:
            par = list(par)
            par[3] = par[3] - 1
            par = tuple(par)

        arr = self.vec_calc_SPL(self.lumarr/1.e38,*par)

        return arr * SFR
    
    def Lehmer21(self, logOH12: float = 8.69, SFR: float = 1., bRand: bool = False, bLum: bool = False) -> np.ndarray:
        """
        Initializes model parameters for HMXB LFs of Lehmer+21 BPL with met. dependent exp cutoff
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

        if SFR < 0.:
            raise ValueError("SFR can not be smaller than zero")

        if not bRand:
            par = ( xu.A_h, xu.logLb, xu.logLc, xu.logLc_logZ, xu.g1_h, xu.g2_h, xu.g2_logZ )
        else:
            par = ( self.par_rand(xu.A_h, xu.sig_Ah),
                    self.par_rand(xu.logLb, xu.sig_logLb),
                    self.par_rand(xu.logLc,xu.sig_logLc),
                    self.par_rand(xu.logLc_logZ,xu.sig_logLcZ),
                    self.par_rand(xu.g1_h,xu.sig_g1h),
                    self.par_rand(xu.g2_h,xu.sig_g2h),
                    self.par_rand(xu.g2_logZ,xu.sig_g2logZ)
                )

        if bLum:
            # par = ( xu.A_h, xu.logLb, xu.logLc, xu.logLc_logZ, xu.g1_h - 1 , xu.g2_h - 1, xu.g2_logZ )
            par = list(par)
            par[4] = par[4] - 1.
            par[5] = par[5] - 1.
            par = tuple(par)
        
        arr = self.vec_calc_Lehmer21(self.lumarr/1.e38,*par,logOH12)

        return arr * SFR

import itertools

def model_err(mod: np.ufunc, LumArr: np.ndarray, args: tuple, logOH12: float) -> tuple:

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

def altZh(la,K,Lb1,Lb2,Lc,a1,a2,a3):
    if la < Lb1:
        res = (la**(1-a1)-Lb1**(1-a1))/(a1-1) + Lb1**(a2-a1)*(Lb1**(1-a2)-Lb2**(1-a2))/(a2-1) + Lb1**(a2-a1)*Lb2**(a3-a2)*(Lb2**(1-a3)-Lc**(1-a3))/(a3-1)
    elif la < Lb2:
        res = Lb1**(a2-a1)*(la**(1-a2)-Lb2**(1-a2))/(a2-1) + Lb1**(a2-a1)*Lb2**(a3-a2)*(Lb2**(1-a3)-Lc**(1-a3))/(a3-1)
    elif la < Lc:
        res = Lb1**(a2-a1)*Lb2**(a3-a2)*(la**(1-a3)-Lc**(1-a3))/(a3-1)
    else:
        res = 0
    
    return res * K


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    import time
    import helper
    
    blo = helper.experimental()
    vec = np.vectorize(blo.diff_Nhxb_met)
    par = ( xu.A_h, xu.logLb, xu.logLc, xu.logLc_logZ, xu.g1_h, xu.g2_h, xu.g2_logZ, 7. )
    end = 1.e4
    N = 200000
     
    bla = helper.Integrate()
    s = time.time()
    Nhx_arr = bla.Riemann_log(vec,10,end,N,*par)
    ent = time.time()-s
    print(ent,Nhx_arr)


    hxb = HMXB(Lmin=35,Lmax=41,nchan=10000)
    lxb = LMXB(nchan=10000)
    vecL = np.vectorize(lxb.calc_Zhang12)
    vecalt = np.vectorize(altZh)
    pL = (xu.norm1*100, xu.Lb1/100, xu.Lb2/100, xu.Lcut_L/100, xu.alpha1, xu.alpha2, xu.alpha3)
    palt = (54.48265, xu.Lb1/100, xu.Lb2/100, xu.Lcut_L/100, xu.alpha1, xu.alpha2, xu.alpha3)
    NL = vecL(lxb.lumarr/1.e38,*pL)
    plt.plot(lxb.lumarr,lxb.Zhang12())
    plt.plot(lxb.lumarr,NL,ls='--')
    plt.plot(lxb.lumarr,vecalt(lxb.lumarr/1.e38,*palt),ls='-.')
    plt.show()
    Li = np.array([7,7.2,7.4,7.6,7.8,8.,8.2,8.4,8.6,8.8,9.,9.2])
    Lo = [4.29,4.27,3.97,3.47,2.84,2.2,1.62,1.15,0.8,0.54,0.37,0.25]
    Loerr = [[3.07,2.53,1.9,1.33,.88,.54,.31,.18,.15,.13,.11,.09],[7.63,5.09,3.26,2.01,1.19,0.68,0.37,0.21,0.17,0.17,0.16,0.15]]
    Loerr2 = [[1.45,1.42,1.17,.85,.55,.33,.28,.1,.08,.06,.05,.03],[5.64,3.76,2.38,1.42,.81,0.44,0.23,0.13,0.09,0.08,0.07,0.06]]
    Lo2 = [1.60,1.80,1.83,1.68,1.41,1.09,0.77,0.51,.32,.2,.11,.07]
    Lu = [40.21,40.25,40.25,40.22,40.16,40.06,39.94,39.8,39.64,39.49,39.34,39.21]
    Luerr = [[.66,.5,.38,.28,.2,.15,.11,.09,.1,.12,.13,.12],[.69,.53,.4,.29,.21,.15,.12,.1,.11,.13,.15,.16]]
    OH = np.linspace(7,9.2,12)
    SFR = [0.01,0.1,1,10,100]
    par = ( xu.A_h, xu.logLb, xu.logLc, xu.logLc_logZ, xu.g1_h, xu.g2_h, xu.g2_logZ )
    N39 = np.vectorize(hxb.calc_Lehmer21)
    foo=10**.54

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

    par2 = ( xu.A_h, xu.logLb, xu.logLc, xu.logLc_logZ, xu.g1_h-1, xu.g2_h-1, xu.g2_logZ )
    fig3, ax3 = plt.subplots(figsize=(12,10))
    linen, = plt.plot(OH, np.log10(N39(.001, *par2, logOH12=OH))+38, label='total L, analytic integration' )
    lineN = plt.errorbar(Li, Lu, yerr=Luerr, ms=8.,capsize=3.,capthick=1.,c='k',fmt='x',label='total Lum, Lehmer+21')
    ax3.set_xlabel(r'$12+\log[O/H]$',fontsize=12)
    ax.set_ylabel(r'$L_X / \mathrm{SFR}$',fontsize=12)

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

    ax.legend(fontsize=11)

    plt.show()  

    
    fig2, ax2 = plt.subplots(figsize=(12,10))
    mod, = plt.plot(hxb.lumarr, N39(hxb.lumarr/1.e38, *par, logOH12=8.69) )
    ax2.set_xlabel(r'$L\, [erg/s]$',fontsize=12)
    ax2.set_ylabel(r'$N(>L)$',fontsize=12)

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
    OH_slider = Slider(ax=axOH, label=r'$OH$', valmin=6.5, valmax=10, valinit=8.69)

    def update2(val):
        mod.set_ydata(N39(hxb.lumarr/1.e38, Ah_slider.val, (Lb_slider.val), Lc_slider.val, Lz_slider.val, g1_slider.val, g2_slider.val, gz_slider.val, OH_slider.val))
        fig2.canvas.draw_idle()

    Ah_slider.on_changed(update2)
    Lb_slider.on_changed(update2)
    Lc_slider.on_changed(update2)
    Lz_slider.on_changed(update2)
    g1_slider.on_changed(update2)
    g2_slider.on_changed(update2)
    gz_slider.on_changed(update2)
    OH_slider.on_changed(update2)

    plt.show()
