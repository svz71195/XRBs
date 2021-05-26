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
        self.models     = {
            "Zh12 " : self.Zhang12,
            "Mi12S" : self.Mineo12S,
            "Mi12B" : self.Mineo12B
        }

    def Zhang12():
        pass
    
    def Mineo12S():
        pass

    def Mineo12B():
        pass
    
    def calc_Nlxb(self, lum_in: float, 
                  K1: float, K2: float, K3: float,
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

    def model_Nlxb(self,
                   K1: float = xu.norm1,
                   Lb1: float = xu.Lb1, Lb2: float = xu.Lb2, Lcut: float = xu.Lcut_L,
                   alpha1: float = xu.alpha1, alpha2: float = xu.alpha2, alpha3: float = xu.alpha3,
                   Mstar: float = 1., case: int = 0 ) -> np.ndarray:
        """
        Vectorization of analytic solutions. Depending on value passed to 'case',\\
            different model parameters can be loaded
        -----
        K1          :   Normalization in units of 1.e11 solar masses
        Lb1         :   First luminosity break in units of 1e36 erg/s
        Lb2         :   Second luminosity break in 1e36 erg/s
        Lcut        :   Luminosity cut-off in 1.e36 erg/s
        alpha1      :   Power-Law slope up to first break
        alpha2      :   Power-Law slope from first to second break
        alpha3      :   Power-Law slope from second break to cut-off
        Mstar       :   Rescaling parameter in units of 1.e11 solar masses
        case        :   Decides which model to use, default = 0\\
                        0 -> Zhang+2012\\
                        1 -> tbd
        """

        K2: float = self.Knorm(K1,Lb1,Lb2,alpha2)
        K3: float = self.Knorm(K2,Lb2,Lcut,alpha3)

        vec_calc_Nlxb = np.vectorize(self.calc_Nlxb)
        Nlx_arr = vec_calc_Nlxb(self.lumarr/1.e36, K1, K2, K3, Lb1, Lb2, Lcut, alpha1, alpha2, alpha3)

        return Nlx_arr * Mstar

    def Knorm(self, K: float, L1: float, L2: float, alpha: float):
        """
        Calculates normalization for changing slopes in calc_Nlxb()
        -----
        K           :   normalization of previous slope
        L1          :   lower luminosity limit for slope range
        L2          :   higher luminosity limit for slope range 
        alpha       :   slope of the desired range
        """
        return (K*(L1/L2)**alpha)

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
        
        return xi/(gamma-1.)*((lum_in/1.e38)**(1.-gamma)-(Lcut/1.e38)**(1.-gamma))

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

    def model_Nhxb(self, SFR: float = 1., case: int = 0 ):
        """
        Vectorization of analytic solutions of HMXB models. Models can be changed\\
            using 'case'
        """


if __name__ == "__main__":
    x = XRB(nchan=10)
    import matplotlib.pyplot as plt
    plt.plot(x.lumarr,x.model_Nlxb())
    plt.show()
    print( x.models['Nlxb'](1,1,1,1,1,1,1,1,1,1) )