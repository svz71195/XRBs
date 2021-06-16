import numpy as np
import json
import sys
from dataclasses import dataclass
from dataclasses import field

try:
    import tqdm
    has_tqdm: bool = True
except ImportError:
    has_tqdm: bool = False
try:
    import g3read as g3
except ImportError:
    print("Could not import g3read. Please check if it is installed.\n It can be found at https://github.com/aragagnin/g3read")

try: 
    from astropy.io import fits
    has_astropy: bool = True
except ImportError:
    has_astropy: bool = False



class Magneticum:
    """
    Collection of methods to retrieve data from Magneticum snapshots and subfind data.
    Requires python module 'g3read' from Antonio!!! You can find g3read at https://github.com/aragagnin/g3read 
    """
    @staticmethod
    def halo_GRNR( groupbase, FSUB ):
        temp0 = g3.read_new(groupbase+".0", "GRNR", 1)
        temp1 = g3.read_new(groupbase+".1", "GRNR", 1)
        temp2 = g3.read_new(groupbase+".2", "GRNR", 1)
        temp3 = g3.read_new(groupbase+".3", "GRNR", 1)
        temp4 = g3.read_new(groupbase+".4", "GRNR", 1)
        return np.concatenate(temp0,temp1,temp2,temp3,temp4)[FSUB]

        


@dataclass
class Galaxy(Magneticum):
    """
    Structure which holds information about a single Magneticum galaxy, like FSUB, Mstar, SFR, etc.
    Is used to create a dictonary of Magneticum galaxies. Also contains X-ray data
    obtained from photon simulations with PHOX
    """
    ##--- Derived from subfind catalogue ---##
    FSUB: int
    GRNR: int               = 0
    center: list[float]     = field(default_factory=list)
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
    Xph_tot: list[float]    = field(default_factory=list)
    Xph_agn: list[float]    = field(default_factory=list)
    Xph_gas: list[float]    = field(default_factory=list)
    Xph_xrb: list[float]    = field(default_factory=list)

    ##--- Derived from which Magneticum Box ---##
    box: str                = "Box4/uhr_test"

    @property
    def sFSUB(self):
        return str(self.FSUB)

    def load(self, groupbase: str, snapbase: str, ph_base: str = None):
        """
        Uses built in functions to populate __init__ fields with data based on FSUB
        """
        if ph_base is None:
            raise Warning("No argument passed for 'ph_base'. If this is deliberate, ignore this warning...")
        if has_tqdm:
            with tqdm.tqdm(total=10, file=sys.stdout) as pbar:
                
                self.set_GRNR(groupbase)
                pbar.update(1)

                self.set_center(groupbase)
                pbar.update(1)
                
                self.set_Mstar(groupbase)
                pbar.update(1)

                self.set_SFR(groupbase)
                pbar.update(1)
        
                self.set_radii(groupbase)
                pbar.update(1)

                self.set_sZ(groupbase)
                pbar.update(1)

                self.set_Rshm(groupbase)
                pbar.update(1)
        
                self.set_galType()
                pbar.update(1)
        
                self.set_bVal(snapbase)
                pbar.update(1)

                if isinstance(ph_base, str):
                    self.set_Xph(ph_base)
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

    def set_GRNR(self, gb: str):
        self.GRNR = super().halo_GRNR(gb, self.FSUB)

    def set_center(self, gb: str) -> None:
        self.center = super().halo_center( gb, self.GRNR )

    def set_Mstar(self, gb):
        Marr = super().halo_Mstar( gb, self.GRNR )
        self.Mstar = Marr[5]

    def set_SFR(self, gb):
        self.SFR = super().halo_SFR( gb, self.FSUB )

    def set_radii(self, gb):
        Rarr = super().halo_radii( gb, self.GRNR )
        self.R25K = Rarr[5]
        self.Rvir = Rarr[0]

    def set_sZ(self, gb):
        self.sZ = super().halo_sZ( gb, self.FSUB )

    def set_Rshm(self, sb):
        pass

    def set_galType(self):
        pass

    def set_bVal(self, sb):
        pass

    def set_Xph(self, pb):
        pass
    
    def get_gas(self, sb):
        pass

    def get_stars(self, sb):
        pass

    def add_luminosities(self, sb):
        _ = self.get_stars(sb)
        pass
    
    @classmethod
    def gal_dict_from_json(cls, json_obj):
        gal_dict = {key: cls(**json_obj[key]) for key in json_obj.keys()}
        return gal_dict


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj) -> dict:
        if isinstance(obj, Galaxy):
            obj_dict = {key: obj.__dict__[key] for key in obj.__dict__}
            return obj_dict
        return json.JSONEncoder.default(self, obj)
    
if __name__ == "__main__":
    d = {"1": {"FSUB": 1, "SFR": 333}, "2": {"FSUB": 2, "Mstar": 23, "Xph_gas": [1.,2.,3.]}}
    x = Galaxy.gal_dict_from_json(d)
    print(x)

    print(json.dumps({"1": Galaxy(1)}, cls=ComplexEncoder))
    print(Galaxy.gal_dict_from_json( json.JSONDecoder().decode( json.dumps({"1": Galaxy(1)},cls=ComplexEncoder) ) ))
