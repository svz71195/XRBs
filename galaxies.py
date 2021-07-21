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
        for halo in matcha.yield_haloes( groupbase, with_ids=True, ihalo_start=GRNR, ihalo_end=GRNR, blocks=('GPOS','MSTR','R25K','MVIR','RVIR','NSUB') ):
            halo_center = halo["GPOS"]
            halo_Mstar = halo["MSTR"][5] # stellar mass within R25K
            halo_R25K = halo["R25K"]
            halo_Mvir = halo["MVIR"]
            halo_Rvir = halo["RVIR"]
            # print(halo["NSUB"])
            
            for subhalo in matcha.yield_subhaloes( groupbase, with_ids=True, halo_ids=halo['ids'], halo_goff=halo['GOFF'], ihalo=halo['ihalo'], blocks=('SPOS','SCM ','SVEL','SSFR','SZ  ') ):
                subhalo_center = subhalo["SPOS"]
                subhalo_vel = subhalo["SVEL"]
                subhalo_SFR = subhalo["SSFR"]
                subhalo_SZ = subhalo["SZ  "]
                # subhalo_SLEN = subhalo["SLEN"]
                # subhalo_SOFF = subhalo["SOFF"]
                # print(subhalo_SLEN, subhalo_SOFF, len(subhalo["ids"]))
                break
                
        # print(halo_center, subhalo_center)
        return (subhalo_center, halo_Mstar, subhalo_SFR, halo_Mvir, halo_Rvir, halo_R25K, subhalo_vel, subhalo_SZ)

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
    def get_index_list_bool(halo_pid, ptype_pid):
        """
        halo_pid: IDs of particles bound in halo
        ptype_pid: IDs of particles of selected ptype in box
        Finds index position of ptype_pid for particles belonging to halo
        Match particle ids that are bound to the halo, super fast!
        """
        ind_all = np.zeros_like(ptype_pid)
        hid = np.sort(halo_pid)
        pid = np.sort(ptype_pid)
        pid_ind = np.argsort(ptype_pid)
        lend = True
        
        icountarr1 = 0
        icountarr2 = 0
        with tqdm.tqdm(colour='red') as pbar:
            while lend:
                if pid[icountarr2] == hid[icountarr1]:
                    ind_all[pid_ind[icountarr2]] = 1
                    icountarr1 += 1
                    icountarr2 += 1
                    
                else:
                    if pid[icountarr2] < hid[icountarr1]:
                        icountarr2 += 1
                    else:
                        icountarr1 += 1

                if icountarr2 == len(pid) or icountarr1 == len(hid):
                    lend = False
                pbar.update(1)

        return ind_all.astype(bool)

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

    def lum_dist_a(self, a):
        """
        luminosity distance given scale factor a in cm
        """
        self.load_FlatLCDM()
        return self.cos.luminosity_distance(1./a-1.).to("cm").value

    def lum_dist_z(self, z):
        """
        luminosity distance given scale factor a in cm
        """
        self.load_FlatLCDM()
        return self.cos.luminosity_distance(z).to("cm").value

    def ang_dist(self, a):
        """
        angular distance given scale factor a in cm
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
    def find_COM(Bpos, Bvel, Bmass, outer):

        pCOM = np.zeros(3)
        vCOM = np.zeros(3)
        tCOM = np.zeros(3)

        rr = np.sqrt( Bpos[:,0]**2 + Bpos[:,1]**2 + Bpos[:,2]**2 )
        mask = (rr <= outer)
        n = len(np.where(mask==True)[0])
        nlimit = min( 1000, int( np.ceil( 0.01*n ) ) )
        while n >= nlimit:
            pCOM[0] = np.sum( Bmass[mask]*Bpos[:,0][mask] ) / np.sum(Bmass[mask])
            pCOM[1] = np.sum( Bmass[mask]*Bpos[:,1][mask] ) / np.sum(Bmass[mask])
            pCOM[2] = np.sum( Bmass[mask]*Bpos[:,2][mask] ) / np.sum(Bmass[mask])
            Bpos = Bpos - pCOM
            tCOM = tCOM + pCOM
            rr = g3.to_spherical(Bpos, [0,0,0]).T[0]
            outer = outer*(1-.025)

            mask = (rr <= outer)
            n = len(np.where(mask==True)[0])

        vv = g3.to_spherical(Bvel[mask], np.median(Bvel[mask], axis=0)).T[0]
        v_max = np.percentile(vv, .9)
        mask2 = (vv <= v_max)

        vCOM[0] = np.sum( Bmass[mask][mask2]*Bvel[:,0][mask][mask2] ) / np.sum(Bmass[mask][mask2])
        vCOM[1] = np.sum( Bmass[mask][mask2]*Bvel[:,1][mask][mask2] ) / np.sum(Bmass[mask][mask2])
        vCOM[2] = np.sum( Bmass[mask][mask2]*Bvel[:,2][mask][mask2] ) / np.sum(Bmass[mask][mask2])

        
        return (tCOM, vCOM)




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
    redshift: float         = 0.01
    
    ##--- Derived from snapshot ---##
    Rshm: float             = 1.
    bVal: float             = 1.
    logOH12_s: float        = 8.69
    logFeH_s: float         = 0.
    Zgal_s: float           = 0.
    logOH12_g: float        = 8.69
    Zgal_g: float           = 0.
    
    
    ##--- Derived from PHOX .fits files
    # Xph_agn: dict           = field(default_factory=dict)
    # Xph_gas: dict           = field(default_factory=dict)
    # Xph_xrb: dict           = field(default_factory=dict)
    
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
            # self.redshift = self.zz_c
            self.set_GRNR()
            self.center, self.Mstar, self.SFR, self.Mvir, self.Rvir, self.R25K, self.Svel, self.sZ = self.get_halo_data(self.groupbase, self.GRNR)
            self.set_index_list()
            self.set_Rshm()
            self.set_bVal()
            # self.set_Xph()
            self.set_st_met()
            self.set_gas_met()


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

    @property
    def Dlum(self):
        return super().lum_dist_z(self.redshift)
        
    def set_index_list(self):
        for halo in matcha.yield_haloes( groupbase, with_ids=True, ihalo_start=self.GRNR, ihalo_end=self.GRNR, blocks=('GPOS','FSUB','NSUB') ):
            i=0
            for subhalo in matcha.yield_subhaloes( groupbase, with_ids=True, halo_ids=halo['ids'], halo_goff=halo['GOFF'], ihalo=halo['ihalo'], blocks=('SOFF','SLEN') ):
                if self.FSUB == halo['FSUB']+i:
                    break
                i+=1
        self.indlist = super().get_index_list_bool(subhalo['ids'], self.get_stars()["ID  "])

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
        pos = self.pos_to_phys(stars["POS "][self.indlist] - self.center)
        vel = self.vel_to_phys(stars["VEL "][self.indlist]) - self.Svel
                
        st_rad =  g3.to_spherical(pos, [0,0,0]).T[0]
        less = st_rad <= .1*self.pos_to_phys(self.Rvir)
        
        st_mass = np.sum(mass[less])
        
        # Approximate from below
        r = 0.
        for dr in [1., .1, .01, .001, 1.e-4, 1.e-5]:
            while np.sum(mass[st_rad <= r+dr]) <= .5*st_mass:
                r += dr   

        k, _ = self.find_COM(pos,vel,mass,5*r)
        
        st_rad =  g3.to_spherical(pos, k).T[0]
        less = st_rad <= .1*self.pos_to_phys(self.Rvir)

        st_mass = np.sum(mass[less])
        
        r = 0.
        for dr in [1., .1, .01, .001, 1.e-4, 1.e-5]:
            while np.sum(mass[st_rad <= r+dr]) <= .5*st_mass:
                r += dr

        self.Rshm = round(r,5)

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # less = st_rad <= 3*r

        # ax.scatter(pos[::4,0][less[::4]],pos[::4,1][less[::4]],pos[::4,2][less[::4]],alpha=.051,edgecolors='none')
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")
        # plt.show()
        

    def set_bVal(self):
        """
        Following Teklu+15, Schulze+18, cuts: b >= -4.35 -> LTG // b <= -4.73 -> ETG
        """
    
        stars = self.get_stars()
        mass = self.mass_to_phys( stars["MASS"][self.indlist] )
        pos = self.pos_to_phys( stars["POS "][self.indlist] - self.center )
        vel = self.vel_to_phys( stars["VEL "][self.indlist] ) - self.Svel
    
        nCOM, nVOM = self.find_COM(pos,vel,mass,4.*self.Rshm)
    
        pos = pos - nCOM
        vel = vel - nVOM
        rad = g3.to_spherical(pos, [0,0,0]).T[0]
        mask = ( rad <= 3.*self.Rshm )

        st_mass_ph = np.sum(mass[mask])
        # print(f"{(st_mass_ph):.3e}") 

        self.Mstar = st_mass_ph 

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

    def set_st_met(self):
        stars = self.stars
        age = self.age_part(stars["AGE "]) # Myrs
        age_c = (age <= 100)
        mass = stars["MASS"]
        iM = self.mass_to_phys( stars["iM  "] )
        Zs = self.mass_to_phys( stars["Zs  "] )

        if len(mass[age_c]) > 0:
                
            Zstar = np.sum(Zs[:,1:][age_c],axis=1) / (iM[age_c] - np.sum(Zs[age_c],axis=1)) / .02 # normalized by solar metallicity
            logOH12_st = np.log10( Zs[:,3][age_c] / 16 / (iM[age_c] - np.sum(Zs[age_c],axis=1)) ) + 12
            logFeH_st = np.log10( Zs[:,-2][age_c] / 55.85 / (iM[age_c] - np.sum(Zs[age_c],axis=1)) ) + 4.33

            self.logOH12_s = np.average(logOH12_st, weights = mass[age_c])
            self.logFeH_s = np.average(logFeH_st, weights = mass[age_c])
            self.Zgal_s = np.average(Zstar, weights = mass[age_c])

        else:
            Zstar = np.sum(Zs[:,1:],axis=1) / (iM - np.sum(Zs,axis=1)) / .02 # normalized by solar metallicity
            logOH12_st = np.log10( Zs[:,3] / 16 / (iM - np.sum(Zs,axis=1)) ) + 12
            logFeH_st = np.log10( Zs[:,-2] / 55.85 / (iM - np.sum(Zs,axis=1)) ) + 4.33

            self.logOH12_s = np.average(logOH12_st, weights = mass)
            self.logFeH_s = np.average(logFeH_st, weights = mass)
            self.Zgal_s = np.average( Zstar, weights = mass )

    def set_gas_met(self):
        gas = self.gas
        sfr = gas["SFR "]
        sfr_c = (sfr > 0)
        mass = self.mass_to_phys( gas["MASS"] )
        Zs = self.mass_to_phys( gas["Zs  "] )

        if len(mass[sfr_c]) > 0:
                
            Zgas = np.sum(Zs[:,1:][sfr_c],axis=1) / (mass[sfr_c] - np.sum(Zs[sfr_c],axis=1)) / .02 # normalized by solar metallicity
            logOH12_gas = 12 + np.log10( Zs[:,3][sfr_c] / 16 / (mass[sfr_c] - np.sum(Zs[sfr_c],axis=1)) )

            self.logOH12_g = np.average(logOH12_gas, weights = mass[sfr_c])
            self.Zgal_g = np.average(Zgas, weights = mass[sfr_c])

        else:
            Zgas = np.sum(Zs[:,1:],axis=1) / (mass - np.sum(Zs,axis=1)) / .02 # normalized by solar metallicity
            logOH12_gas = 12 + np.log10( Zs[:,3] / 16 / (mass - np.sum(Zs,axis=1)) )

            self.logOH12_g = np.average(logOH12_gas, weights = mass)
            self.Zgal_g = np.average(Zgas, weights = mass)
            
    def rotate_faceon(self):
        stars = self.get_stars()
        pos = self.pos_to_phys( stars["POS "] - self.center )
        vel = self.vel_to_phys(stars["VEL "]) - self.Svel
        mass = stars["MASS"]

        nCOM, nVOM = self.find_COM(pos,vel,mass,4.*self.Rshm)
    
        pos = pos - nCOM
        vel = vel - nVOM

        vec_in = np.asarray(self.spec_ang_mom(mass,pos,vel))
        vec_in = vec_in / np.sum(vec_in**2)**.5
        vec_p1 = np.cross([0.,1.,0.], vec_in)
        vec_p1 = vec_p1 / np.sum(vec_p1**2)**.5
        vec_p2 = np.cross(vec_in, vec_p1)

        matr = np.concatenate((vec_p1, vec_p2, vec_in)).reshape((3, 3))

        return np.column_stack( np.matmult(matr,pos[:,0]), np.matmult(matr,pos[:,1]), np.matmult(matr,pos[:,2]) )


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

    

    # @staticmethod
def calc_lum( phArr: np.ndarray, Tobs: float = 1., Aeff: float = 1., Dlum: float = 1. ):
    # m = (phArr > 0.5) & (phArr < 8.)
    flux = np.sum(phArr)*1.602e-9
    lumi = ( flux / Aeff / Tobs * Dlum * Dlum * 4.*np.pi )
        
    return lumi

def get_num_XRB_base( Xph_base: str, sFSUB: str, Tobs: float = 1., Aeff: float = 1., Dlum: float = 1., Lc: float = -1.):
    """
    Returns tuple containing number of XRBs and array of luminosities
    """
    fp_hxb = Xph_base+"gal"+sFSUB+"HXB.fits"
    fp_hxrb = Xph_base+"gal"+sFSUB+"HXRB.fits"
    fp_lxb = Xph_base+"gal"+sFSUB+"LXB.fits"
    fp_lxrb = Xph_base+"gal"+sFSUB+"LXRB.fits"
    fp_xrb = Xph_base+"gal"+sFSUB+"XRB.fits"

    # include cases where XRB types are not separated
    try:
        tbl_hxb = Table.read(fp_hxb)
    except FileNotFoundError:
        try:
            tbl_hxb = Table.read(fp_hxrb)
        except FileNotFoundError:
            try:
                tbl_hxb = Table.read(fp_xrb)
            except FileNotFoundError:
                tbl_hxb = {}
                tbl_hxb["PHOTON_ENERGY"] = np.array([])

    try:
        tbl_lxb = Table.read(fp_lxb)
    except FileNotFoundError:
        try: 
            tbl_lxb = Table.read(fp_lxrb)
        except FileNotFoundError:
            tbl_lxb = {}
            tbl_lxb["PHOTON_ENERGY"] = np.array([])

    phE_h = np.array( tbl_hxb["PHOTON_ENERGY"] )
    phE_l = np.array( tbl_lxb["PHOTON_ENERGY"] )

    indx_pckg_end_h = np.where( np.diff(phE_h) < 0)[0]
    indx_pckg_end_l = np.where( np.diff(phE_l) < 0)[0]
    numHXB = len(indx_pckg_end_h)
    numLXB = len(indx_pckg_end_l)

    lumH = np.zeros(numHXB)
    lumL = np.zeros(numLXB)
    
    for i in range(numHXB):
        if i == 0:
            lumH[i] = calc_lum( phE_h[0:indx_pckg_end_h[0]+1], Tobs, Aeff, Dlum )
        else:
            lumH[i] = calc_lum( phE_h[indx_pckg_end_h[i-1]+1:indx_pckg_end_h[i]+1], Tobs, Aeff, Dlum )

    for i in range(numLXB):
        if i == 0:
            lumL[i] = calc_lum( phE_l[0:indx_pckg_end_l[0]+1], Tobs, Aeff, Dlum )
        else:
            lumL[i] = calc_lum( phE_l[indx_pckg_end_l[i-1]+1:indx_pckg_end_l[i]+1], Tobs, Aeff, Dlum )

    if Lc < 0:
        return [(numHXB, lumH), (numLXB, lumL)]
    else:
        lumH = lumH[lumH>Lc]
        lumL = lumL[lumL>Lc]
        numH = len(lumH)
        numL = len(lumL)
        return [(numH, lumH), (numL, lumL)]

def get_num_XRB( phE_h: list, Tobs: float = 1., Aeff: float = 1., Dlum: float = 1., Lc: float = -1.):
    """
    Returns tuple containing number of XRBs and array of luminosities
    """

    indx_pckg_end_h = np.where( np.diff(phE_h) < 0)[0]
    numHXB = len(indx_pckg_end_h)

    lumH = np.zeros(numHXB)
    
    for i in range(numHXB):
        if i == 0:
            lumH[i] = calc_lum( phE_h[0:indx_pckg_end_h[0]+1], Tobs, Aeff, Dlum )
        else:
            lumH[i] = calc_lum( phE_h[indx_pckg_end_h[i-1]+1:indx_pckg_end_h[i]+1], Tobs, Aeff, Dlum )

    if Lc < 0:
        return (numHXB, lumH)
    else:
        lumH = lumH[lumH>Lc]
        numH = len(lumH)
        return (numH, lumH)

def calc_galLumX(Xph_base: str, sFSUB: str, Tobs: float = 1., Aeff: float = 1., Dlum: float = 1.):
    fp_gas = Xph_base+"gal"+sFSUB+"GAS.fits"
    fp_agn = Xph_base+"gal"+sFSUB+"AGN.fits"
    fp_hxb = Xph_base+"gal"+sFSUB+"HXB.fits"
    fp_lxb = Xph_base+"gal"+sFSUB+"LXB.fits"
    tbl_gas = Table.read(fp_gas)
    tbl_agn = Table.read(fp_agn)
    tbl_lxb = Table.read(fp_hxb)
    tbl_hxb = Table.read(fp_lxb)

    phE_g = np.array( tbl_gas["PHOTON_ENERGY"] )
    phE_a = np.array( tbl_agn["PHOTON_ENERGY"] )
    phE_h = np.array( tbl_hxb["PHOTON_ENERGY"] )
    phE_l = np.array( tbl_lxb["PHOTON_ENERGY"] )
        
    collect = (phE_g, phE_a, phE_h, phE_l)

    # .5-50keV, 0.5-2keV, 0.5-8keV, 0.5-10keV, 2-8keV, 2-10keV

    XrayLum = {}
    for key,x in zip(["agn","gas","hxb","lxb"],collect):
        tot = (x < 50.) & (x > .5)
        sx = (x < 2.) & (x > .5)
        mxq = (x < 8.) & (x > .5)
        mxp = (x < 10.) & (x > .5)
        hxq = (x < 8.) & (x > 2.)
        hxp = (x < 10.) & (x > 2.)

        XrayLum[key] = ( calc_lum(x[tot],Tobs,Aeff,Dlum), calc_lum(x[sx],Tobs,Aeff,Dlum), calc_lum(x[mxq],Tobs,Aeff,Dlum), calc_lum(x[mxp],Tobs,Aeff,Dlum), calc_lum(x[hxq],Tobs,Aeff,Dlum), calc_lum(x[hxp],Tobs,Aeff,Dlum) )

    return XrayLum

def get_spectra(Xph_base: str, sFSUB: str, Nbin: int = 500, Emin: float = .5, Emax: float = 10., norm = False, Tobs: float = -1, Aeff: float = -1):
    fp_gas = Xph_base+"gal"+sFSUB+"GAS.fits"
    fp_agn = Xph_base+"gal"+sFSUB+"AGN.fits"
    fp_hxb = Xph_base+"gal"+sFSUB+"HXB.fits"
    fp_lxb = Xph_base+"gal"+sFSUB+"LXB.fits"
    fp_xrb = Xph_base+"gal"+sFSUB+"XRB.fits"

    try:
        tbl_gas = Table.read(fp_gas)
    except FileNotFoundError:
        tbl_gas = {}
        tbl_gas["PHOTON_ENERGY"] = np.array([])

    try:
        tbl_agn = Table.read(fp_agn)
    except FileNotFoundError:
        tbl_agn = {}
        tbl_agn["PHOTON_ENERGY"] = np.array([])

    # include cases where XRB types are not separated
    try:
        tbl_hxb = Table.read(fp_hxb)
    except FileNotFoundError:
        try:
            tbl_hxb = Table.read(fp_xrb)
        except FileNotFoundError:
            tbl_hxb = {}
            tbl_hxb["PHOTON_ENERGY"] = np.array([])

    try:
        tbl_lxb = Table.read(fp_lxb)
    except FileNotFoundError:
        tbl_lxb = {}
        tbl_lxb["PHOTON_ENERGY"] = np.array([])

    phE_g = np.array( tbl_gas["PHOTON_ENERGY"] )
    phE_a = np.array( tbl_agn["PHOTON_ENERGY"] )
    phE_h = np.array( tbl_hxb["PHOTON_ENERGY"] )
    phE_l = np.array( tbl_lxb["PHOTON_ENERGY"] )

    spec_bins = np.logspace(np.log10(Emin),np.log10(Emax),Nbin)
    spec_g, _ = np.histogram(phE_g,spec_bins)
    spec_a, _ = np.histogram(phE_a,spec_bins)
    spec_h, _ = np.histogram(phE_h,spec_bins)
    spec_l, _ = np.histogram(phE_l,spec_bins)

    if norm:
        numph = len(phE_g) + len(phE_a) + len(phE_h) + len(phE_l)
        return (spec_bins, spec_g / numph, spec_a / numph, spec_h / numph, spec_l / numph)
    else:
        return (spec_bins, spec_g, spec_a, spec_h, spec_l)

def get_ph_pos(Xph_base: str, sFSUB: str, Emin: float = .5, Emax: float = 10.):
    fp_gas = Xph_base+"gal"+sFSUB+"GAS.fits"
    fp_agn = Xph_base+"gal"+sFSUB+"AGN.fits"
    fp_hxb = Xph_base+"gal"+sFSUB+"HXB.fits"
    fp_lxb = Xph_base+"gal"+sFSUB+"LXB.fits"
    fp_xrb = Xph_base+"gal"+sFSUB+"XRB.fits"

    try:
        tbl_gas = Table.read(fp_gas)
    except FileNotFoundError:
        tbl_gas = {}
        tbl_gas["POS_X"] = np.array([])
        tbl_gas["POS_Y"] = np.array([])

    try:
        tbl_agn = Table.read(fp_agn)
    except FileNotFoundError:
        tbl_agn = {}
        tbl_agn["POS_X"] = np.array([])
        tbl_agn["POS_Y"] = np.array([])

    # include cases where XRB types are not separated
    try:
        tbl_hxb = Table.read(fp_hxb)
    except FileNotFoundError:
        try:
            tbl_hxb = Table.read(fp_xrb)
        except FileNotFoundError:
            tbl_hxb = {}
            tbl_hxb["POS_X"] = np.array([])
            tbl_hxb["POS_Y"] = np.array([])

    try:
        tbl_lxb = Table.read(fp_lxb)
    except FileNotFoundError:
        tbl_lxb = {}
        tbl_lxb["POS_X"] = np.array([])
        tbl_lxb["POS_Y"] = np.array([])

    # Create position vectors (x,y): np.array([[x1,y1],[x2,y2],...,[xn,yn]])
    pos_g = np.column_stack( ( tbl_gas["POS_X"], tbl_gas["POS_Y"] ) )
    pos_a = np.column_stack( ( tbl_agn["POS_X"], tbl_agn["POS_Y"] ) )
    pos_h = np.column_stack( ( tbl_hxb["POS_X"], tbl_hxb["POS_Y"] ) )
    pos_l = np.column_stack( ( tbl_lxb["POS_X"], tbl_lxb["POS_Y"] ) )



if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import matplotlib.colors as mclr
    import xrb_main as xm
    import matplotlib.ticker as ticker
    from matplotlib import cm

    groupbase = "/home/lcladm/Studium/Masterarbeit/test/dorc/uhr_test/groups_136/sub_136"
    snapbase = "/home/lcladm/Studium/Masterarbeit/test/dorc/uhr_test/snapdir_136/snap_136"
    phbase = "/home/lcladm/Studium/Masterarbeit/R136_AGN_fix/fits/"
    phbase_2 = "/home/lcladm/Studium/Masterarbeit/R333/fits/"
    phbase_3 = "./fits_g/"


    head = g3.GadgetFile(snapbase+".0").header
    h = head.HubbleParam
    zz = head.redshift

    # halo_positions = g3.read_new(groupbase+".0","GPOS",0)
    # halo_mass_str = g3.read_new(groupbase+".0","MSTR",0)
    # halo_mass = halo_mass_str[:,5]
    # halo_FSUB = g3.read_new(groupbase+".0","FSUB",0)
    # halo_rad25K = g3.read_new(groupbase+".0","R25K",0)

    # for k in range(1,4):
    #     # halo_positions = np.append(halo_positions, g3.read_new(groupbase+"."+str(k),'GPOS',0), axis=0 )
    #     halo_mass_str  = g3.read_new(groupbase+"."+str(k),'MSTR',0)
    #     halo_mass = np.append(halo_mass, halo_mass_str[:,5] ) #mass25K
    #     halo_FSUB = np.append(halo_FSUB, g3.read_new(groupbase+"."+str(k),'FSUB',0) )
    #     halo_rad25K = np.append(halo_rad25K, g3.read_new(groupbase+"."+str(k),'R25K',0))
    # sel = (halo_mass >= 8.e-1*h) & (halo_mass <= 1.e4*h) & (halo_rad25K > 0.)
    # halo_FSUB = halo_FSUB[sel]
    # del halo_mass, halo_mass_str, halo_rad25K

    # print(len(halo_mass), len(halo_FSUB) )
    # input()


    # head = g3.GadgetFile(snapbase+".0").header
    # h = head.HubbleParam
    # mass25K = np.zeros(4002)
    # halo_FSUB = np.zeros(4002)
    # halo_rad25K = np.zeros(4002)
    # for i,halo in enumerate(tqdm.tqdm(matcha.yield_haloes(groupbase,with_ids=True,ihalo_start=0,ihalo_end=2000,blocks=('MSTR','FSUB','R25K'),use_cache=False))):
    #     mass25K[i]         = halo['MSTR'][5] * 1e10 / h
    #     halo_FSUB[i]       = halo['FSUB']
    #     halo_rad25K[i]     = halo['R25K']
    # sel = (mass25K >= 9.e9) & (mass25K <= 1.e14) & (halo_rad25K > 0.)
    # halo_FSUB = halo_FSUB[sel]

    # gal_dict = {}
    # # gal_dict_v = Galaxy.gal_dict_from_npy("gal_data.npy")
    # ii = 0
    # for ihal,fsub in enumerate(tqdm.tqdm(halo_FSUB)):
        
    #     print(fsub,ihal)
    #     key = f"{int(fsub):06d}"
    #     temp_gal = Galaxy(int(fsub),redshift=.01,groupbase=groupbase,snapbase=snapbase,Xph_base=phbase)
    #     temp_gal.load
       
    #     print(temp_gal.bVal, temp_gal.Rshm)
    #     gal_dict[key] = temp_gal
    #     del temp_gal
    #     ii+=1
    #     if ii%50 == 0:
    #         np.save("gal_data_temp.npy", gal_dict)
        

    # np.save("gal_data.npy", gal_dict)

    # exit()

    logFeH_s = []
    Mstar = []
    bVal = []
    nxb = []
    i=0
    gal_dict = Galaxy.gal_dict_from_npy("gal_data.npy")
    for key in tqdm.tqdm(gal_dict.keys()):
        x = gal_dict[key]
        #x.load_FlatLCDM()

        # stars = x.stars
        # age = x.age_part( stars["AGE "] )
        # ac = (age <= 100)
        # mass = stars["MASS"][ac]
        # if len(mass) == 0:
        #     continue
        # iM = x.mass_to_phys(stars["iM  "])[ac]
        # Zs = x.mass_to_phys(stars["Zs  "])[ac]

        # logFeH_st = np.log10( Zs[:,-2] / 55.85 / (iM - np.sum(Zs,axis=1)) ) + 4.33
        # wh = Zs[:,-2] > 0
        # logFeH_s.append(np.average(logFeH_st[wh], weights = mass[wh]))
        logFeH_s.append( x.logOH12_s )
        Mstar.append(np.log10(x.Mstar))
        ll = get_num_XRB_base(phbase,x.sFSUB,1.e6,1.e3,x.lum_dist_z(x.redshift),Lc=1.e38)
        NXB = ll[0][0]
        nxb.append(NXB)
        bVal.append(x.bVal)
        break
        


        #plt.plot(np.log10(x.Mstar),logFeH_s,lw=0.,c=x.bVal,marker='o',ms=3.,norm=)
    plt.scatter(Mstar,logFeH_s,c=bVal,marker='o',s=15.,cmap='jet_r')
    cb = plt.colorbar()
    cb.set_label(label="$b$-val",fontsize=14)
    plt.ylabel(r"[Fe/H]",fontsize=14)
    plt.xlabel(r"$\log(M_*)$",fontsize=14)
    plt.show()

    plt.scatter(logFeH_s,nxb,c=bVal,marker='o',s=15.,cmap='jet_r')
    cb = plt.colorbar()
    cb.set_label(label="$b$-val",fontsize=14)
    plt.xlabel(r"12 + [O/H]",fontsize=14)
    plt.ylabel(r"$N_{\mathrm{XRB}}(>10^{38}\, \mathrm{erg\,s^{-1}})$",fontsize=14)
    plt.show()

    hxb = []
    Z = []
    L = []
    LL = []
    LerrUp = []
    LerrLw = []
    SFR = []
    M25K = []
    sSFR = []
    for key in tqdm.tqdm(gal_dict.keys()):
        x = gal_dict[key]
        # gas = x.get_gas()
        stars = x.get_stars()
        # rad1 = x.pos_to_phys(g3.to_spherical(gas['POS '],x.center).T[0])
        # mask1 = rad1 < x.pos_to_phys(x.R25K)
        rad2 = x.pos_to_phys(g3.to_spherical(stars['POS '],x.center).T[0])
        age = x.age_part(stars["AGE "])
        mask2 = (rad2 < x.pos_to_phys(x.R25K)) & (age<100)
        m25k = x.mass_to_phys(np.sum(stars["MASS"][(rad2 < x.pos_to_phys(x.R25K))]))
        M25K.append( m25k )
        aSFR = x.mass_to_phys(np.sum(stars["MASS"][mask2]))/.95e8
        
        # fp_hxrb = phbase_3+"gal"+str(x.FSUB)+"HXRB.fits"
        # tbl_hxrb = Table.read(fp_hxrb)
        fp_xrb = phbase+"gal"+x.sFSUB+"XRB.fits"
        fp_gas = phbase+"gal"+x.sFSUB+"GAS.fits"
        try:
            tbl_hxrb = Table.read(fp_xrb)
            tbl_gas = Table.read(fp_gas)
        except FileNotFoundError:
            continue
        Xph_xrb = np.array(tbl_hxrb["PHOTON_ENERGY"])
        Xph_xrb = Xph_xrb[(Xph_xrb>=.5)&(Xph_xrb<8.)]
        Xph_gas = np.array(tbl_gas["PHOTON_ENERGY"])
        Xph_gas = Xph_gas[(Xph_gas>=.5)&(Xph_gas<8.)]
        # phx_pos = np.column_stack(tbl_hxrb["POS_X"],tbl_hxrb["POS_Y"])
        NXB, lum = get_num_XRB(Xph_xrb, 1.e6, 1.e3, x.lum_dist_z(x.redshift), Lc=1.e39)
        Lx = calc_lum(Xph_xrb,1.e6,1.e3,x.lum_dist_z(zz))
        Lg = calc_lum(Xph_gas,1.e6,1.e3,x.lum_dist_z(zz))
        
        # print(np.sum(gas['SFR '][mask1]),x.SFR,aSFR,np.log10(Lx),np.log10(Lx/x.SFR),np.log10(Lx/aSFR),x.logOH12_s,x.logOH12_g)
        # if x.SFR <= 0.:
        if aSFR <= 0.:
            # continue
            hxb.append((NXB))
            SFR.append(0)
            Z.append(x.logOH12_s)
            L.append(Lx+Lg)
            LL.append(Lx)
            sSFR.append( aSFR/m25k )
            
        else:
            hxb.append((NXB/aSFR))
            SFR.append(aSFR)
            Z.append(x.logOH12_s)
            L.append((Lx+Lg)/aSFR)
            LL.append(Lx/aSFR)
            sSFR.append( aSFR/m25k )
            
        if int(key) >= 88000:
            break
        
    

    # cmap = cm.jet()
    norm=mclr.LogNorm(vmin=1.e-3,vmax=200.,clip=True)
    curve_samp = np.load("Lehm21_samp.npy")
    # print(cm.jet(norm(1)))

    OH = np.linspace(6.8,9.6,41)
    ssfr = [.01,.1,1.,10,100]
    modelN = np.zeros_like(OH)
    ind = np.where(xm.Lehmer21().lumarr>=1.e39)[0][0]
    for i,oh in enumerate(OH):
        modelN[i] = xm.Lehmer21(logOH12=oh).model()[ind]
    plt.plot(OH,modelN,c='k')
    plt.scatter(Z,hxb,c=SFR,alpha=.5,marker='o',s=15.,norm=mclr.LogNorm(vmin=1.e-3,vmax=200.,clip=True),cmap='jet_r')
    cb = plt.colorbar()
    cb.set_label(label=r"$\mathrm{SFR}\,[M_{\odot}\, \mathrm{yr}^{-1}]$",fontsize=14)
    cb.ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    plt.xlabel(r"12 + [O/H]",fontsize=14)
    plt.ylabel(r"$N_{\mathrm{HXB}}(>10^{39}\, \mathrm{erg\,s^{-1}})$ / SFR",fontsize=14)
    plt.show()

    modelL = np.zeros_like(OH)
    Lu = [40.21,40.25,40.25,40.22,40.16,40.06,39.94,39.8,39.64,39.49,39.34,39.21]
    err = [[.66,.5,.38,.28,.2,.15,.11,.09,.1,.12,.13,.12],[.69,.53,.4,.29,.21,.15,.12,.1,.11,.13,.15,.16]]
    errOH = np.linspace(7,9.2,12)
    for i,oh in enumerate(OH):
        modelL[i] = xm.Lehmer21(logOH12=oh).model(bLum=True)[0]
    plt.plot(OH,np.log10(modelL)+38,c='k')
    plt.fill_between(errOH,np.array(Lu)+np.array(err[0]),np.array(Lu)-np.array(err[1]),color='k',alpha=.2,interpolate=True)
    for j,sfc in enumerate(curve_samp):
        if j==2 or j==1 or j==0:
            plt.plot(errOH,np.log10(sfc[0]),c=cm.jet_r(norm(ssfr[j])),alpha=.9)
            plt.fill_between(errOH,np.log10(sfc[0]-sfc[1]),np.log10(sfc[0]+sfc[2]),color=cm.jet_r(norm(ssfr[j])),alpha=.2,interpolate=True)
        
    plt.scatter(Z,np.log10(LL),c=SFR,alpha=.6,marker='o',s=15.,norm=mclr.LogNorm(vmin=1.e-3,vmax=200.,clip=True),cmap='jet_r')
    # plt.errorbar(Z,np.log10(L),yerr=[np.log10(np.array(L))-np.log10(np.array(LerrLw)), np.log10(np.array(LerrUp))-np.log10(np.array(L))], ms=8.,capsize=3.,capthick=1.,ecolor='k')
    cb = plt.colorbar()
    cb.set_label(label=r"$\mathrm{SFR}\,[M_{\odot}\, \mathrm{yr}^{-1}]$",fontsize=14)
    cb.ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    plt.xlabel(r"12 + [O/H]",fontsize=14)
    plt.ylabel(r"$\log(L_{\mathrm{HXB}}^{.5-8\,\mathrm{keV}} / \mathrm{SFR})$",fontsize=14)
    plt.show()

    def curve(sss):
        beta = 10**39.92
        alpha = 10**29.857
        return np.log10( (alpha/sss) + (beta))
    plt.plot(np.log10(np.logspace(-14,-8,500)),curve(np.array(np.logspace(-14,-8,500))),c='k' )
    plt.scatter(np.log10(sSFR),np.log10(L),c=SFR,alpha=.6,marker='o',s=15.,norm=mclr.LogNorm(vmin=1.e-3,vmax=200.,clip=True),cmap='jet_r')
    cb = plt.colorbar()
    cb.set_label(label=r"$\mathrm{SFR}\,[M_{\odot}\, \mathrm{yr}^{-1}]$",fontsize=14)
    cb.ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    plt.xlabel(r"$\log(\mathrm{sSFR})$",fontsize=14)
    plt.ylabel(r"$\log(L_{\mathrm{X}}^{.5-8\,\mathrm{keV}} / \mathrm{SFR})$",fontsize=14)
    plt.show()
    

    #x.add_luminosities()