import numpy as np
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import sys
import os
from numba import njit
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
    munit       = 1.e10 # Msun
    KPC_TO_CM   = 3.086e21
    KEV_TO_ERG  = 1.602e-9
    G           = 6.6726e-8
    mp          = 1.672623e-24
    kB          = 1.380658e-16
    pc          = 3.085678e18
    YR_TO_SEC   = 3.1557e7

    @staticmethod
    def get_halo_data( groupbase: str, GRNR: int ) -> tuple:
        """
        returns tuple (center, Mstar, SFR, Mvir, Rvir, R25K, SVEL, SZ, SLEN, SOFF) from group GRNR
        """
        for halo in matcha.yield_haloes( groupbase, with_ids=True, ihalo_start=GRNR, ihalo_end=GRNR, blocks=('GPOS','MSTR','R25K','MVIR','RVIR') ):
            halo_center = halo["GPOS"]
            halo_Mstar = halo["MSTR"][5] # stellar mass within R25K
            halo_R25K = halo["R25K"]
            halo_Mvir = halo["MVIR"]
            halo_Rvir = halo["RVIR"]

            for subhalo in matcha.yield_subhaloes( groupbase, with_ids=True, halo_ids=halo['ids'], halo_goff=halo['GOFF'], ihalo=halo['ihalo'], blocks=('SPOS','SCM ','SVEL','SSFR','SZ  ','SLEN','SOFF') ):
                subhalo_center = subhalo["SPOS"]
                subhalo_vel = subhalo["SVEL"]
                subhalo_SFR = subhalo["SSFR"]
                subhalo_SZ = subhalo["SZ  "]
                subhalo_SLEN = subhalo["SLEN"]
                subhalo_SOFF = subhalo["SOFF"]
                break

        return (subhalo_center, halo_Mstar, subhalo_SFR, halo_Mvir, halo_Rvir, halo_R25K, subhalo_vel, subhalo_SZ, subhalo_SLEN, subhalo_SOFF)

    @staticmethod
    def get_halo_GRNR( groupbase, FSUB ):
        halo_NSUB = np.array([])
        for k in range(16):
            halo_NSUB = np.append( halo_NSUB, g3.read_new(groupbase+"."+str(k), "NSUB", 0, is_snap=False) )
            if FSUB < np.sum(halo_NSUB):
                break
        j = 0
        while ( np.sum(halo_NSUB[:j+1])-FSUB <= 0 ):
            j = j+1
        return j

    @staticmethod
    def get_index_list(halo_pid, ptype_pid):
        ind_all = np.zeros_like(ptype_pid)
        for i in range(len(ptype_pid)):
            if np.any( ptype_pid[i] == halo_pid ):
                ind_all[i] = 1
        return ind_all

    @staticmethod
    def halo_gas(sb, ce, rad):
        return g3.read_particles_in_box(snap_file_name=sb, center=ce, d=rad, blocks=["POS ", "VEL ", "MASS", "RHO ", "NH  ", "NE  ", "SFR ", "Zs  ", "HSML", "ID  "], ptypes=0)

    @staticmethod
    def halo_stars(sb, ce, rad):
        return g3.read_particles_in_box(snap_file_name=sb, center=ce, d=rad, blocks=["POS ", "VEL ", "MASS", "iM  ", "AGE ", "Zs  ", "HSMS", "ID  "], ptypes=4)
    
    @staticmethod
    def spec_ang_mom(mass, pos, vel):
        return np.sum( np.cross( (mass*pos.T).T, vel, axis=1 ), axis=0 ) / np.sum(mass) 

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

    def pos_to_phys(self, pos):
        return pos * self.aa_c / self.h

    def vel_to_phys(self, vel):
        return vel * np.sqrt(self.aa_c)

    def mass_to_phys(self, mass):
        return mass * self.munit / self.h
    
    @staticmethod
    # @njit
    def find_COM(pos, vel, mass, outer):
        steps = np.arange(1,int(outer)+1)[::-1]
        pCOM = np.zeros(3)
        vCOM = np.zeros(3)
        n = len(vel)
        range_frac = .9
        N = int(range_frac * n - 1.)
        for rad in steps:
            rr = np.sqrt( pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2 ) # np.linalg.norm(pos,axis=1) # length of each postition vector
            vv = np.sqrt( vel[:,0]**2 + vel[:,1]**2 + vel[:,2]**2 ) # np.linalg.norm(vel,axis=1) # length of each velocity vector
            
            ii = np.argsort(np.abs(vv))
            
            vrange = np.abs( vv[ ii[N] ] )

            kk = (rr <= rad) & (vv <= vrange)

            p0 = np.array( [ np.sum( mass[kk]*pos[:,0][kk] ), np.sum( mass[kk]*pos[:,1][kk] ), np.sum( mass[kk]*pos[:,2][kk] ) ] ) / np.sum(mass[kk])
            v0 = np.array( [ np.sum( mass[kk]*vel[:,0][kk] ), np.sum( mass[kk]*vel[:,1][kk] ), np.sum( mass[kk]*vel[:,2][kk] ) ] ) / np.sum(mass[kk])

            pos -= p0
            vel -= v0

            pCOM += p0
            vCOM += v0

        return (pCOM, vCOM)




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
    Svel: float             = 1.
    sZ: float               = 1.
    redshift: float         = 0.
    
    ##--- Derived from snapshot ---##
    Rshm: float             = 1.
    # ETG: bool               = False
    # LTG: bool               = False
    bVal: float             = 1.
    
    ##--- Derived from PHOX .fits files
    Xph_agn: dict           = field(default_factory=dict)
    Xph_gas: dict           = field(default_factory=dict)
    Xph_xrb: dict           = field(default_factory=dict)
    
    ##--- Derived from which Magneticum Box ---##
    groupbase: str          = "./Box4/uhr_test/groups_136/sub_136"
    snapbase: str           = "./Box4/uhr_test/snapdir_136/snap_136"
    Xph_base: str           = "./fits_136/"


    @property
    def sFSUB(self):
        return f"{self.FSUB:06d}"

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

        if False:
            with tqdm.tqdm(total=7, file=sys.stdout) as pbar:
                
                self.load_FlatLCDM()
                pbar.update(1)

                self.set_GRNR()
                pbar.update(1)

                self.center, self.Mstar, self.SFR, self.Mvir, self.Rvir, self.R25K, self.Svel, self.sZ, self.SLEN, self.SOFF = self.get_halo_data(self.groupbase, self.GRNR)
                pbar.update(1)

                self.set_index_list()
                pbar.update(1)

                self.set_Rshm()
                pbar.update(1)

                self.set_bVal()
                pbar.update(1)

                if isinstance( self.Xph_base, str ):
                    self.set_Xph()
                pbar.update(1)

        else:
            self.load_FlatLCDM()
            self.redshift = self.zz_c
            self.set_GRNR()
            self.center, self.Mstar, self.SFR, self.Mvir, self.Rvir, self.R25K, self.Svel, self.sZ, self.SLEN, self.SOFF = self.get_halo_data(self.groupbase, self.GRNR)
            self.set_index_list()
            self.set_Rshm()
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

    def set_GRNR(self) -> None:
        self.GRNR = super().get_halo_GRNR( self.groupbase, self.FSUB)

    def set_index_list(self):
        PIDs = g3.read_new(groupbase+".0", "PID ",2)
        halo_pids = PIDs[self.SOFF:self.SOFF+self.SLEN+1]
        self.indlist = super().get_index_list(halo_pids, self.get_stars()["ID  "]).astype(bool)

    def set_center(self) -> None:
        self.set_GRNR
        self.center = super().get_halo_data( self.groupbase, self.GRNR )[0]

    def set_Mstar(self) -> None:
        self.set_GRNR
        self.Mstar = super().get_halo_data( self.groupbase, self.GRNR )[1] * self.munit / self.h
        self.Mvir = super().get_halo_data( self.groupbase, self.GRNR )[3] * self.munit / self.h

    def set_SFR(self):
        self.set_GRNR
        self.SFR = super().get_halo_data( self.groupbase, self.GRNR )[2]

    def set_radii(self):
        self.set_GRNR
        self.Rvir = super().get_halo_data( self.groupbase, self.GRNR )[4]
        self.R25K = super().get_halo_data( self.groupbase, self.GRNR )[5]

    def set_sZ(self):
        self.set_GRNR
        self.sZ = super().get_halo_data( self.groupbase, self.GRNR )[6]

    def set_Rshm(self):
        """
        Following description in Teklu+15, stellar half-mass radius from stars within 10% of Rvir
        returns half-mass radius in kpc/h
        """
        
        stars = self.get_stars()
        mass = stars["MASS"][self.indlist]
        
        st_rad =  g3.to_spherical(stars["POS "][self.indlist], self.center).T[0]
        less = st_rad <= .1*self.Rvir
        
        st_mass = np.sum(mass[less])
        # print(st_mass)
        # Approximate from below
        r = 0.
        for dr in [1., .1, .01, .001, 1.e-4, 1.e-5]:
            while np.sum(mass[st_rad <= r+dr]) <= .5*st_mass:
                r += dr   
        self.Rshm = round(self.pos_to_phys(r),5)
        # print(self.Rshm)
        
        

        # jj = np.argsort(st_rad)
        # mcur = 0.
        # for k in range(len(st_rad)):
        #     mcur += mass[jj[k]]
        #     if mcur > .5*st_mass:
        #         break
        # r0 = self.pos_to_phys(st_rad[jj[k]])
        # print(k, self.Rshm - r0)

    def set_bVal(self):
        """
        Following Teklu+15, Schulze+18, cuts: b >= -4.35 -> LTG // b <= -4.73 -> ETG
        """
    
        stars = self.get_stars()
        mass = self.mass_to_phys( stars["MASS"][self.indlist] )
        pos = self.pos_to_phys( stars["POS "][self.indlist] - self.center )
        vel = self.vel_to_phys( stars["VEL "][self.indlist] - self.Svel )
    
        k, l = self.find_COM(pos,vel,mass,3.*self.Rshm)
    
        # print(k,l)
        pos = pos - k
        # vel = vel - l
        rad = g3.to_spherical(pos, [0,0,0]).T[0]
        mask = ( rad <= 3.*self.Rshm )


        st_mass_ph = np.sum(mass[mask]) 
        # print(f"{st_mass_ph = :.2e}")

        self.Mstar = self.mass_to_phys(st_mass_ph)
        self.bVal = np.log10( np.linalg.norm(self.spec_ang_mom(mass[mask], pos[mask], vel[mask])) ) - 2./3. * np.log10(st_mass_ph)        
    
    def set_Xph(self):
        fp_gas = self.Xph_base+"gal"+self.sFSUB+"GAS.fits"
        fp_agn = self.Xph_base+"gal"+self.sFSUB+"AGN.fits"
        fp_xrb = self.Xph_base+"gal"+self.sFSUB+"XRB.fits"
        tbl_gas = Table.read(fp_gas)
        tbl_agn = Table.read(fp_agn)
        tbl_xrb = Table.read(fp_xrb)
        
        try:
            self.Xph_gas["PHE"] = np.array(tbl_gas["PHOTON_ENERGY"])
            self.Xph_agn["PHE"] = np.array(tbl_agn["PHOTON_ENERGY"])
            self.Xph_xrb["PHE"] = np.array(tbl_xrb["PHOTON_ENERGY"])
        except KeyError:

            try:
                self.Xph_gas["PHE"] = np.array(tbl_gas["PHE"])
                self.Xph_agn["PHE"] = np.array(tbl_agn["PHE"])
                self.Xph_xrb["PHE"] = np.array(tbl_xrb["PHE"])
            except KeyError:
                raise KeyError("No column name called 'PHE' or 'PHOTON_ENERGY' found")
        
        try:
            self.Xph_gas["XPOS"] = np.array(tbl_gas["POS_X"])
            self.Xph_agn["XPOS"] = np.array(tbl_agn["POS_X"])
            self.Xph_xrb["XPOS"] = np.array(tbl_xrb["POS_X"])
        except KeyError:
            raise KeyError("No coolumn name called 'XPOS' found")

        try:
            self.Xph_gas["YPOS"] = np.array(tbl_gas["POS_Y"])
            self.Xph_agn["YPOS"] = np.array(tbl_agn["POS_Y"])
            self.Xph_xrb["YPOS"] = np.array(tbl_xrb["POS_Y"])
        except KeyError:
            raise KeyError("No coolumn name called 'YPOS' found")
        

    def get_gas(self):
        return super().halo_gas( self.snapbase, self.center, self.Rvir )
    
    @property
    def gas(self):
        return self.get_gas()

    def get_stars(self):
        return super().halo_stars( self.snapbase, self.center, self.Rvir )
    
    @property
    def stars(self):
        return self.get_stars()

    def add_luminosities(self):
        stars = self.get_stars
        rad = self.pos_to_phys( g3.to_spherical( stars["POS "], self.center ).T[0] )
        iM = self.mass_to_phys( stars["iM  "] )
        Tdiff = self.age_part(stars["AGE "]) / 1.e3 # in Gyr
        Zs = self.mass_to_phys( stars["Zs  "] )

        CB07, tempiAS = self.load_CB07()

        Zstar = np.sum(Zs[1:],axis=1) / (iM - np.sum(Zs,axis=1)) / .02
        
    @staticmethod
    def load_CB07(obs = False, num = None):
        LT_ADD_GAL_TO_SUB = 12
        CB07 = np.zeros(LT_ADD_GAL_TO_SUB + 218 * LT_ADD_GAL_TO_SUB + 6 * 219 * LT_ADD_GAL_TO_SUB)
        tempiAS = 10**(np.loadtxt("/HydroSims/Projekte/Safe/CB07/tempi") - 9.) # in Gyr

        if not obs:
            for imet in range(6):
                t = f"{imet+2:1d}"
                CB07File='/HydroSims/Projekte/Safe/CB07/Chabrier/cb2007_hr_stelib_m'+t+'2_ssp.multi_mag_vega'
                d = np.loadtxt(CB07File)
                d.T

                j = 0

                for i in range(219):
                    for imag in range(12):
                        CB07[imag + (i * LT_ADD_GAL_TO_SUB) + (imet * 219 * LT_ADD_GAL_TO_SUB)] = d[j]
                    j += 1
        else:
            for imet in range(6):
                t = f"{imet+2:1d}"
                t2 = f"{num:03d}"
                CB07File='/HydroSims/Projekte/Safe/CB07/Chabrier/cb2007_hr_stelib_m'+t+'2_ssp.multi_mag_vega_'+t2
                d = np.loadtxt(CB07File)
                d.T

                j = 0

                for i in range(219):
                    for imag in range(12):
                        CB07[imag + (i * LT_ADD_GAL_TO_SUB) + (imet * 219 * LT_ADD_GAL_TO_SUB)] = d[j]
                    j += 1

        return (CB07, tempiAS)

    @classmethod
    def gal_dict_from_npy(cls, npy_obj):
        gal_dict = np.load(npy_obj, allow_pickle=True).item()
        return gal_dict

    def get_num_XRB(self, Tobs: float = 1., Aeff: float = 1.):
        """
        Returns tuple containing number of XRBs and array of luminosities
        """
        indx_pckg_end = np.where( np.diff(self.Xph_xrb) <= 0)[0]
        numXRB = len(indx_pckg_end) + 1
        lum = np.zeros(numXRB)
        for i,ind in enumerate(indx_pckg_end):
            lum[i] = self.calc_lum( self.Xph_xrb[ind,ind+1], Tobs, Aeff, self.lum_dist )

        return (numXRB, lum)

    @staticmethod
    def calc_lum( phArr: np.ndarray, Tobs: float = 1., Aeff: float = 1., Dlum: float = 1. ):
       return phArr * super().KEV_TO_ERG / Tobs / Aeff * Dlum * Dlum


if __name__ == "__main__":
    
    groupbase = "/home/lcladm/Studium/Masterarbeit/test/dorc/uhr_test/groups_136/sub_136"
    snapbase = "/home/lcladm/Studium/Masterarbeit/test/dorc/uhr_test/snapdir_136/snap_136"
    phbase = "/home/lcladm/Studium/Masterarbeit/R136_AGN_fix/fits/"

    head = g3.GadgetFile(snapbase+".0").header
    h = head.HubbleParam
    mass25K = np.zeros(4002)
    halo_FSUB = np.zeros(4002)
    halo_rad25K = np.zeros(4002)
    for i,halo in enumerate(tqdm.tqdm(matcha.yield_haloes(groupbase,with_ids=True,ihalo_start=0,ihalo_end=4000,blocks=('MSTR','FSUB','R25K'),use_cache=False))):
        mass25K[i]         = halo['MSTR'][5] * 1e10 / h
        halo_FSUB[i]       = halo['FSUB']
        halo_rad25K[i]     = halo['R25K']
    sel = (mass25K >= 9.e9) & (mass25K <= 1.e14) & (halo_rad25K > 0.)
    halo_FSUB = halo_FSUB[sel]

    gal_dict = {}
    for fsub in tqdm.tqdm(halo_FSUB):
        key = f"{int(fsub):06d}"
        temp_gal = Galaxy(int(fsub),groupbase=groupbase,snapbase=snapbase,Xph_base=phbase)
        temp_gal.load
        gal_dict[key] = temp_gal

    np.save("gal_data.npy", gal_dict)
        
    # # x = Galaxy(1414,groupbase=groupbase,snapbase=snapbase,Xph_base=phbase)
    # y = Galaxy(6463,groupbase=groupbase,snapbase=snapbase,Xph_base=phbase)
    # z = Galaxy(10859,groupbase=groupbase,snapbase=snapbase,Xph_base=phbase)
    # #x.load
    # y.load
    # z.load
    # #print(x)
    # print(y)
    # print(z)

    #x.add_luminosities()