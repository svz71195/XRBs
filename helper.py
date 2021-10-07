
import numpy as np
from scipy.special import gammaincc, gamma
from numba import njit

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

def calc_sideon_matrix(angmom_vec):
    vec_in = np.asarray(angmom_vec)
    vec_in = vec_in / np.sum(vec_in ** 2).sum() ** 0.5
    vec_p1 = np.cross([1, 0, 0], vec_in)
    vec_p1 = vec_p1 / np.sum(vec_p1 ** 2).sum() ** 0.5
    vec_p2 = np.cross(vec_in, vec_p1)

    matr = np.concatenate((vec_p2, vec_in, vec_p1)).reshape((3, 3))

    return matr


def calc_faceon_matrix(angmom_vec, up=[0.0, 1.0, 0.0]):
    vec_in = np.asarray(angmom_vec)
    vec_in = vec_in / np.sum(vec_in ** 2).sum() ** 0.5
    vec_p1 = np.cross(up, vec_in)
    vec_p1 = vec_p1 / np.sum(vec_p1 ** 2).sum() ** 0.5
    vec_p2 = np.cross(vec_in, vec_p1)

    matr = np.concatenate((vec_p1, vec_p2, vec_in)).reshape((3, 3))

    return matr

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
                      A: float, logLb: float, logLc: float, logLc_logZ: float,
                      g1: float, g2: float, g2_logZ: float,
                      logOH12: float ) -> float:
        """
        Differential function dN/dL for metallicity enhanced HMXB LF in Lehmer+21\\
        Needs to be integrated numerically. See implementation
        of 'self.Riemann_log(func: ufunc, l_min: float, l_max: float, *par: tuple)'.

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
        Lb  = 10**(logLb-38)

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

import xrb_units as xu
class XRB:
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

    def sample_test(self, NXRB: int) -> np.ndarray:
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

class LMXB(XRB):
    
    def __init__(self, nchan: int = 10000, Lmin: float = 35, Lmax: float = 41, Emin: float = 0.05, Emax: float = 50.1) -> None:
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
    def __init__(self, nchan: int = 10000, Lmin: float = 35, Lmax: float = 41, Emin: float = 0.05, Emax: float = 50.1) -> None:
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