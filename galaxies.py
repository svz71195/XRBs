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
    import g3matcha as matcha
except ImportError:
    exit("Could not import g3read. Please check if it is installed.\n It can be found at https://github.com/aragagnin/g3read")

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
    def get_halo_data( groupbase: str, GRNR: int ) -> tuple:
        """
        returns tuple (center, Mstar, SFR, Mvir, Rvir, R25K, SZ) from group GRNR
        """
        for halo in matcha.yield_haloes( groupbase, with_ids=True, ihalo_start=GRNR, ihalo_end=GRNR, blocks=('GPOS','MSTR','R25K','MVIR','RVIR') ):
            halo_center = halo["GPOS"]
            halo_Mstar = halo["MSTR"] # stellar mass within R25K
            halo_R25K = halo["R25K"]
            halo_Mvir = halo["MVIR"]
            halo_Rvir = halo["RVIR"]

            for subhalo in matcha.yield_subhaloes( groupbase, with_ids=True, halo_ids=halo['ids'], halo_goff=halo['GOFF'], ihalo=halo['ihalo'], blocks=('SSFR','SZ  ') ):
                subhalo_SFR = subhalo["SSFR"]
                subhalo_SZ = subhalo["SZ  "]
                break

        return (halo_center, halo_Mstar, subhalo_SFR, halo_Mvir, halo_Rvir, halo_R25K, subhalo_SZ)

    @staticmethod
    def get_halo_GRNR( groupbase, FSUB ):
        halo_NSUB = np.array([])
        for k in range(16):
            halo_NSUB = np.append( halo_NSUB, g3.read_new(groupbase+"."+str(k), "NSUB", 0, is_snap=False) )
            if FSUB < np.sum(halo_NSUB):
                break
        j = 0
        while ( np.sum(halo_NSUB[:j+1])-FSUB < 0 ):
            j = j+1
        return j


        


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
    center: np.ndarray      = np.zeros(3)
    Mstar: float            = 1.    
    SFR: float              = 1.
    Mvir: float             = 1.
    Rvir: float             = 1.
    R25K: float             = 1.
    sZ: float               = 1.
    
    ##--- Derived from snapshot ---##
    Rshm: float             = 1.
    ETG: bool               = False
    LTG: bool               = False
    bVal: float             = 1.
    
    ##--- Derived from PHOX .fits files
    Xph_tot: np.ndarray     = np.zeros(10)
    Xph_agn: np.ndarray     = np.zeros(10)
    Xph_gas: np.ndarray     = np.zeros(10)
    Xph_xrb: np.ndarray     = np.zeros(10)

    ##--- Derived from which Magneticum Box ---##
    groupbase: str          = "./Box4/uhr_test/groups_136/sub_136"
    snapbase: str           = "./Box4/uhr_test/snapdir_136/snap_136"
    Xph_base: str            = "./fits_136/"

    @property
    def sFSUB(self):
        return str(self.FSUB)

    @property
    def load(self):
        """
        Uses built in functions to populate __init__ fields with data based on FSUB
        """

        if not isinstance(self.groupbase, str):
            raise TypeError("'groupbase' was not set properly. Make sure that 'groupbase' is a valid path to halo catalogues ...")

        if not isinstance(self.snapbase, str):
            raise TypeError("'snapbase' was not set properly. Make sure that 'snapbase' is a valid path to snapshot files ...")

        if not isinstance(self.Xph_base, str):
            raise Warning("'Xph_base' was not set properly. Make sure that 'Xph_base' is a valid path to X-ray fits files produced with PHOX.\n If this is deliberate, ignore this warning...")


        if has_tqdm:
            with tqdm.tqdm(total=10, file=sys.stdout) as pbar:
                
                self.set_GRNR
                pbar.update(1)

                self.center, self.Mstar, self.SFR, self.Mvir, self.Rvir, self.R25K = self.get_halo_data(self.groupbase, self.GRNR)
                pbar.update(1)

                self.set_Rshm()
                pbar.update(1)

                self.set_bVal()
                pbar.update(1)

                self.set_galType()
                pbar.update(1)

                if isinstance( self.Xph_base, str ):
                    self.set_Xph( self.Xph_base )
                pbar.update(1)
        else:
            self.set_GRNR( self.groupbase, self.FSUB)
            self.center, self.Mstar, self.SFR, self.Mvir, self.Rvir, self.R25K = self.get_halo_data(self.groupbase, self.FSUB)
            self.set_Rshm()
            self.set_galType()
            self.set_bVal()
            self.set_Xph()

    def set_groupbase(self, fp: str):
        if isinstance(fp, str):
            self.groupbase = fp
        else:
            raise TypeError("Can not set groupbase to non str object")

    def set_snapbase(self, fp: str):
        if isinstance(fp, str):
            self.snapbase = fp
        else:
            raise TypeError("Can not set snapbase to non str object")

    def set_Xph_base(self, fp: str):
        if isinstance(fp, str):
            self.groupbase = fp
        else:
            raise TypeError("Can not set Xph_base to non str object")

    @property
    def set_GRNR(self) -> None:
        self.GRNR = super().get_halo_GRNR( self.groupbase, self.FSUB)

    @property
    def set_center(self) -> None:
        self.set_GRNR
        self.center = super().get_halo_data( self.groupbase, self.GRNR )[0]

    @property
    def set_Mstar(self) -> None:
        self.set_GRNR
        self.Mstar = super().get_halo_data( self.groupbase, self.GRNR )[1]
        self.Mvir = super().get_halo_data( self.groupbase, self.GRNR )[3]

    @property
    def set_SFR(self):
        self.set_GRNR
        self.SFR = super().get_halo_data( self.groupbase, self.GRNR )[2]

    @property
    def set_radii(self):
        self.set_GRNR
        self.Rvir = super().get_halo_data( self.groupbase, self.GRNR )[4]
        self.R25K = super().get_halo_data( self.groupbase, self.GRNR )[5]

    @property
    def set_sZ(self):
        self.set_GRNR
        self.sZ = super().get_halo_data( self.groupbase, self.GRNR )[6]

    def set_Rshm(self):
        pass
    
    def set_bVal(self):
        pass

    def set_galType(self):
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

    @classmethod
    def gal_dict_from_npy(cls, npy_obj):
        gal_dict = np.load(npy_obj).item()
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
