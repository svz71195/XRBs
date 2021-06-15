import numpy as np
import json
import sys
from dataclasses import dataclass

try:
    import tqdm
    has_tqdm: bool = True
except ImportError:
    has_tqdm: bool = False
try:
    import g3read as g3
    has_g3read: bool = True
except ImportError:
    has_g3read: bool = False

try: 
    from astropy.io import fits
    has_astropy: bool = True
except ImportError:
    has_astropy: bool = False



class Magneticum:
    pass


@dataclass
class Galaxy(Magneticum):
    """
    Structure which holds information about a single Magneticum galaxy, like FSUB, Mstar, SFR, etc.
    Is used to create a dictonary of Magneticum galaxies. Also contains X-ray data
    obtained from photon simulations with PHOX
    """
    ##--- Derived from subfind catalogue ---##
    FSUB: int
    center: tuple           = (0.,0.,0.)
    Mstar: float            = 1.    
    SFR: float              = 1.
    Rvir: float             = 1.
    R25K: float             = 1.
    sZ: float               = 1.
    
    ##--- Derived from snapshot ---##
    Rshm: float             = 1.
    ETG: bool               = False
    LTG: bool               = False
    bVal: float             = 1.
    
    ##--- Derived from PHOX .fits files
    Xph_tot: np.ndarray   = np.zeros(10)
    Xph_agn: np.ndarray   = np.zeros(10)
    Xph_gas: np.ndarray   = np.zeros(10)
    Xph_xrb: np.ndarray   = np.zeros(10)

    @property
    def sFSUB(self):
        return str(self.FSUB)

    def load(self):
        """
        Uses built in functions to populate __init__ fields with data based on FSUB
        """

        if has_tqdm:
            with tqdm.tqdm(total=9, file=sys.stdout) as pbar:
        
                self.set_center()
                pbar.update(1)
                
                self.set_Mstar()
                pbar.update(1)

                self.set_SFR()
                pbar.update(1)
        
                self.set_radii()
                pbar.update(1)

                self.set_sZ()
                pbar.update(1)

                self.set_Rshm()
                pbar.update(1)
        
                self.set_galType()
                pbar.update(1)
        
                self.set_bVal()
                pbar.update(1)

                self.set_Xph()
                pbar.update(1)
        else:
            self.set_center()
            self.set_Mstar()
            self.set_SFR()  
            self.set_radii()
            self.set_sZ()
            self.set_Rshm()
            self.set_galType()
            self.set_bVal()
            self.set_Xph()

    
    def set_center(self):
        pass

    def set_Mstar(self):
        pass

    def set_SFR(self):
        pass

    def set_radii(self):
        pass

    def set_sZ(self):
        pass

    def set_Rshm(self):
        pass

    def set_galType(self):
        pass

    def set_bVal(self):
        pass

    def set_Xph(self):
        pass
    
    def get_gas(self):
        pass

    def get_stars(self):
        pass

    def add_luminosities(self):
        pass



    
