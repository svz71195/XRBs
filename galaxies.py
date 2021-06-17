import numpy as np
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import sys
import os
from dataclasses import dataclass, field


try:
    import tqdm
    has_tqdm: bool = True
except ImportError:
    has_tqdm: bool = False
try:
    import g3read as g3
    import g3matcha as matcha
except ImportError:
    exit("Could not import g3read. Please check if it is installed and all requirements are present.\n It can be found at https://github.com/aragagnin/g3read")

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
    munit       = 1.e10
    KPC_TO_CM   = 3.086e21
    KEV_TO_ERG  = 1.602e-9
    G           = 6.6726e-8
    mp          = 1.672e-24
    kB          = 1.38e-16
    pc          = 3.086e18
    YR_TO_SEC   = 3.1557e7

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

            for subhalo in matcha.yield_subhaloes( groupbase, with_ids=True, halo_ids=halo['ids'], halo_goff=halo['GOFF'], ihalo=halo['ihalo'], blocks=('SSFR','SZ  ', 'REFF') ):
                subhalo_SFR = subhalo["SSFR"]
                subhalo_SZ = subhalo["SZ  "]
                subhalo_Rshm = subhalo["REFF"]
                break

        return (halo_center, halo_Mstar, subhalo_SFR, halo_Mvir, halo_Rvir, halo_R25K, subhalo_SZ, subhalo_Rshm)

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

    @staticmethod
    def halo_gas(sb, ce, rad):
        return g3.read_particles_in_box(snapbase=sb, center=ce, d=rad, blocks=["POS ", "VEL ", "MASS", "RHO ", "NH  ", "NE  ", "SFR ", "Zs ", "HSML"], ptypes=0)

    @staticmethod
    def halo_stars(sb, ce, rad):
        return g3.read_particles_in_box(snapbase=sb, center=ce, d=rad, blocks=["POS ", "VEL ", "MASS", "iM  ", "AGE ", "Zs ", "HSMS"], ptypes=4)
    

    ### Cosmology
    def load_FlatLCDM(self):
        head = g3.GadgetFile(self.snapbase+".0").header
        self.h = head.HubbleParam
        Om0 = head.Omega0
        self.zz_c = head.redshift
        self.aa_c = 1. / (1. + self.zz_c)
        
        self.cos = FlatLambdaCDM( H0 = self.h*100, Om0 = Om0 )

    def agez(self, z):
        return self.cos.age(z).to("Myr").value

    def age(self, a):
        """
        Age of universe given its scalefactor in Myr
        """
        return self.cos.age(1./a-1.).to("Myr").value

    def age_part(self, a):
        """
        Age of particle born at scalefactor 'a' in Myr
        """
        return ( self.age(self.aa_c) - self.age(a) )

    def lum_dist(self, a):
        """
        luminosity distance given scale factor a in Mpc
        """
        return self.cos.luminosity_distance(1./a-1.).value

    def ang_dist(self, a):
        """
        angular distance given scale factor a in Mpc
        """
        return self.lum_dist(a) / a / a



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
    redshift: float         = 0.
    
    ##--- Derived from snapshot ---##
    Rshm: float             = 1.
    ETG: bool               = False
    LTG: bool               = False
    bVal: float             = 1.
    
    ##--- Derived from PHOX .fits files
    Xph_agn: dict           = {"PHE": np.zeros(3), "XPOS": np.zeros(3), "YPOS": np.zeros(3)}
    Xph_gas: dict           = {"PHE": np.zeros(3), "XPOS": np.zeros(3), "YPOS": np.zeros(3)}
    Xph_xrb: dict           = {"PHE": np.zeros(3), "XPOS": np.zeros(3), "YPOS": np.zeros(3)}
    

    ##--- Derived from which Magneticum Box ---##
    groupbase: str          = "./Box4/uhr_test/groups_136/sub_136"
    snapbase: str           = "./Box4/uhr_test/snapdir_136/snap_136"
    Xph_base: str           = "./fits_136/"


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

        # does path exist?
        if not os.path.isfile(self.groupbase+".0"):
            raise FileNotFoundError(self.groupbase+" does not contain the expected files...")
        
        if not os.path.isfile(self.snapbase+".0"):
            raise FileNotFoundError(self.snapbase+" does not contain the expected files...")

        if not os.path.isdir(self.Xph_base):
            raise FileNotFoundError(self.Xph_base+" is not a valid directory...")

        if has_tqdm:
            with tqdm.tqdm(total=6, file=sys.stdout) as pbar:
                
                self.set_GRNR
                pbar.update(1)

                self.center, self.Mstar, self.SFR, self.Mvir, self.Rvir, self.R25K = self.get_halo_data(self.groupbase, self.GRNR)
                pbar.update(1)

                self.set_Rshm()
                pbar.update(1)

                self.set_bVal()
                pbar.update(1)

                if isinstance( self.Xph_base, str ):
                    self.set_Xph( self.Xph_base )
                pbar.update(1)

                self.load_FlatLCDM()
                pbar.update(1)
        else:
            self.set_GRNR( self.groupbase, self.FSUB)
            self.center, self.Mstar, self.SFR, self.Mvir, self.Rvir, self.R25K = self.get_halo_data(self.groupbase, self.FSUB)
            self.set_Rshm()
            self.set_bVal()
            self.set_Xph()
            self.load_FlatLCDM()

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
        self.Mstar = super().get_halo_data( self.groupbase, self.GRNR )[1] * self.munit / self.h
        self.Mvir = super().get_halo_data( self.groupbase, self.GRNR )[3] * self.munit / self.h

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
        """
        Following description in Teklu+15, stellar half-mass radius from stars within 10% of Rvir
        returns half-mass radius in kpc/h
        """
        
        stars = self.get_stars
        mass = stars["MASS"]
        st_rad = g3.to_spherical(stars["POS "], self.center).T[0]
        st_mass = np.sum(mass[st_rad<=0.1*self.Rvir])

        # Approximate from below
        r = 0.
        for dr in [1.,.1,.01,.001,.0001]:
            while np.sum(mass[st_rad <= r]) <= .5*st_mass:
                r += dr # Overestimates r -> substract one for next iteration
            r -= dr        
        return r
    
    def set_bVal(self):
        pass
    
    @property
    def set_Xph(self):
        fp_gas = self.Xph_base+"gal"+self.sFSUB+"GAS.fits"
        fp_agn = self.Xph_base+"gal"+self.sFSUB+"AGN.fits"
        fp_xrb = self.Xph_base+"gal"+self.sFSUB+"XRB.fits"
        tbl_gas = Table.read(fp_gas)
        tbl_agn = Table.read(fp_agn)
        tbl_xrb = Table.read(fp_xrb)
        
        try:
            self.Xph_gas["PHE"] = tbl_gas["ENERGY"].astype(float)
            self.Xph_agn["PHE"] = tbl_agn["ENERGY"].astype(float)
            self.Xph_xrb["PHE"] = tbl_xrb["ENERGY"].astype(float)
        except KeyError:

            try:
                self.Xph_gas["PHE"] = tbl_gas["PHE"].astype(float)
                self.Xph_agn["PHE"] = tbl_agn["PHE"].astype(float)
                self.Xph_xrb["PHE"] = tbl_xrb["PHE"].astype(float)
            except KeyError:
                raise KeyError("No column name called 'PHE' or 'ENERGY' found")
        
        try:
            self.Xph_gas["XPOS"] = tbl_gas["XPOS"].astype(float)
            self.Xph_agn["XPOS"] = tbl_agn["XPOS"].astype(float)
            self.Xph_xrb["XPOS"] = tbl_xrb["XPOS"].astype(float)
        except KeyError:
            raise KeyError("No coolumn name called 'XPOS' found")

        try:
            self.Xph_gas["YPOS"] = tbl_gas["YPOS"].astype(float)
            self.Xph_agn["YPOS"] = tbl_agn["YPOS"].astype(float)
            self.Xph_xrb["YPOS"] = tbl_xrb["YPOS"].astype(float)
        except KeyError:
            raise KeyError("No coolumn name called 'YPOS' found")

    @property
    def get_gas(self):
        return super().halo_gas( self.snapbase, self.center, self.R25K )
    
    @property
    def gas(self):
        return self.get_gas()

    @property
    def get_stars(self):
        return super().halo_stars( self.snapbase, self.center, self.R25K )
    
    @property
    def stars(self):
        return self.get_stars()

    def add_luminosities(self):
        _ = self.get_stars()
        pass

    @classmethod
    def gal_dict_from_npy(cls, npy_obj):
        gal_dict = np.load(npy_obj).item()
        return gal_dict
