import numpy as np
import xrb_units as xu

class XRB:
    def __init__(self, nchan: int = 10000, Lmin: float = 34, Lmax: float = 41, Emin: float = 0.05, Emax: float = 50.1) -> None:
        self.nchan      = nchan
        self.Lmin       = Lmin
        self.Lmax       = Lmax

        if self.Lmax < self.Lmin:
            raise ValueError("Lmax can't be smaller than Lmin!")

        if ( self.Lmin < 0 ):
            raise ValueError("Lmin can't be smaller than 0!")
        
        self.lumarr     = np.logspace(Lmin, Lmax, self.nchan)

        self.Emin       = Emin
        self.Emax       = Emax

        if self.Emax < self.Emin:
            raise ValueError("Emax can't be smaller than Emin!")

        if self.Emin < 0:
            raise ValueError("Emin can't be smaller than 0!")

        # THIS WORKS !!!
        self.modelsL = {
            "Zh12"  : self.Zhang12,
            "0"     : self.Zhang12,
            0       : self.Zhang12
        }
        
        self.modelsH = {
            "Mi12S" : self.Mineo12S,
            "0"     : self.Mineo12S,
            0       : self.Mineo12S,
            "Mi12B" : self.Mineo12B,
            "1"     : self.Mineo12B,
            1       : self.Mineo12B,
            "Le20"  : self.Lehmer20,
            "2"     : self.Lehmer20,
            2       : self.Lehmer20
        }

    def Zhang12(self, bRand: bool = False) -> tuple:
        """
        Initializes model parameters for LMXB LFs of Zhang+12
        returns tuple of parameters for easy pass to other functions
        return (norm1, break1, break2, cut-off, slope1, slope2, slope3)
        -----
        norm1       :   Normalization in units of 1.e11 solar masses
        Lb1         :   First luminosity break in units of 1e36 erg/s
        Lb2         :   Second luminosity break in 1e36 erg/s
        Lcut        :   Luminosity cut-off in 1.e36 erg/s
        alpha1      :   Power-Law slope up to first break
        alpha2      :   Power-Law slope from first to second break
        alpha3      :   Power-Law slope from second break to cut-off
        bRand       :   boolean switching between randomized parameters\\
                            according to their uncertainty
        """
        if not bRand:
            return (xu.norm1, xu.Lb1, xu.Lb2, xu.Lcut_L, xu.alpha1, xu.alpha2, xu.alpha3)
        else:
            return (self.rand(xu.norm1,xu.sig_K1),
                    self.rand(xu.Lb1,xu.sig_Lb1), 
                    self.rand(xu.Lb2,xu.sig_Lb2), 
                    xu.Lcut_L,
                    self.rand(xu.alpha1,xu.sig_a1), 
                    self.rand(xu.alpha2,xu.sig_a2), 
                    self.rand(xu.alpha3,xu.sig_a3)
                )
    
    def Mineo12S(self, bRand: bool = False) -> tuple:
        """
        Initializes model parameters for HMXB LFs of Mineo+12 single PL
        returns tuple of parameters for easy pass to other functions
        return (norm1, break1, break2, cut-off, slope1, slope2, slope3)
        -----
        bRand       :   boolean switching between randomized parameters\\
                            according to their uncertainty
        """
        if not bRand:
            return (xu.xi_s, xu.Lcut_Hs, xu.gamma_s)
        else:
            return (10**self.rand(xu.log_xi_s, xu.log_sig_xi_s), 
                    xu.Lcut_Hs,
                    self.rand(xu.gamma_s,xu.sig_gam_s)
                )

    def Mineo12B(self, bRand: bool = False) -> tuple:
        """
        Initializes model parameters for HMXB LFs of Mineo+12 single PL
        returns tuple of parameters for easy pass to other functions
        return (norm1, break1, break2, cut-off, slope1, slope2, slope3)
        -----
        bRand       :   boolean switching between randomized parameters\\
                            according to their uncertainty
        """
        if not bRand:
            return (xu.xi2_b, xu.LbH ,xu.Lcut_Hb, xu.gamma1_b, xu.gamma2_b)
        else:
            return (self.rand(xu.xi2_b, xu.sig_xi2),
                    self.rand(xu.LbH, xu.sig_LbH),
                    xu.Lcut_Hb,
                    self.rand(xu.gamma1_b, xu.sig_g1),
                    self.rand(xu.gamma2_b, xu.sig_g2)
                )
    
    def Lehmer20(self):
        pass
    
    def rand(self, mu, sigma, size=None):
        return np.random.normal(mu,sigma,size)
    
    def calc_Nlxb(self, 
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

    def model_Nlxb(self, case: str = '0', Mstar: float = 1., bRand: bool = False) -> np.ndarray:
        """
        Vectorization of analytic solutions. Depending on value passed to 'case',\\
            different model parameters can be loaded
        -----
        Mstar       :   Rescaling parameter in units of 1.e11 solar masses
        bRand       :   boolean switching between randomized parameters\\
                            according to their uncertainty
        case        :   Decides which model to use, by passing KeyWord strings:\\
                        'Zh12' -> Zhang+2012\\
                        'Le19' -> Lehmer+2019\\
                        ...
                        Can also be accessed by passing integers starting from 0
        """

        vec_calc_Nlxb = np.vectorize(self.calc_Nlxb)

        try:
            par: tuple = self.modelsL[case](bRand)

        except KeyError:
            raise KeyError("Desired model '"+str(case)+"' not implemented! Available models are",
                           [key for key in self.modelsL.keys() if len(str(key))>2]
                        )

        Nlx_arr = vec_calc_Nlxb(self.lumarr/1.e36, *par)
        return Nlx_arr * Mstar

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

    def calc_Nhxb_SPL(self, lum_in: float, 
                      xi: float, Lcut: float, gamma: float ) -> float:
        """
        Analytic solution of HMXB single Power-Law luminosity function (Mineo+2012)\\
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

    def calc_Nhxb_BPL(self, lum_in: float,
                      xi: float, Lb1: float, Lcut: float,
                      gamma1: float, gamma2: float ) -> float:
        """
        Analytic solution of HMXB broken Power-Law luminosity function (Mineo+2012)\\
        Used for vectorization in model_Nhxb()
        -----
        lum_in      :   input luminosity in units of 1.e38 erg/s
        xi          :   normalization constant
        Lb1         :   luminosity break in units of 1.e38 erg/s
        gamma1      :   Power-Law slope uo to first
        Lcut        :   Luminosity cut-off in 1.e38 erg/s
        SFR         :   Star-Formation-Rate in units of Msun/yr\\
                        Used for rescaling of normalization
        """

        if (lum_in < Lb1):
            return( xi * ( ( lum_in**(1.-gamma1) - Lb1**(1.-gamma1) )/(gamma1-1.)
                + Lb1**(gamma2-gamma1)*( Lb1**(1.-gamma2) - Lcut**(1.-gamma2) ) / (gamma2-1.) )
                )

        elif (lum_in >= Lb1) and (lum_in < Lcut):
            return xi * Lb1**(gamma2-gamma1)*(lum_in**(1.-gamma2) - Lcut**(1.-gamma2)) / (gamma2-1.)

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

        return Nhx_arr * SFR

if __name__ == "__main__":
    x = XRB()
    import matplotlib.pyplot as plt
    plt.plot(x.lumarr,x.model_Nhxb(),c='k',label='True')
    plt.plot(x.lumarr,x.model_Nhxb(0,1,True))
    plt.plot(x.lumarr,x.model_Nhxb(0,1,True))
    plt.plot(x.lumarr,x.model_Nhxb(0,1,True))
    plt.legend()
    plt.show()
