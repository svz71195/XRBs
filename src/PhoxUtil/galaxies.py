import numpy as np
from astropy.cosmology import FlatLambdaCDM
# import astropy.units as u
import sys
import os
from dataclasses import dataclass, field
from numba import njit
import numba as nb

KEV_TO_ERG = 1.60218e-9
KPC_TO_CM = 3.0856e21
MPC_TO_CM = 3.0856e24

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
    
    # @staticmethod
    # def get_halo_data( groupbase: str, GRNR: int ) -> tuple:
    def get_halo_data(self) -> None:
        """
        returns tuple (center, Mstar, SFR, Mvir, Rvir, R25K, SVEL, SZ, SLEN, SOFF) from group GRNR
        """
        for halo in matcha.yield_haloes( self.groupbase,
                                    # with_ids = True,
                                    ihalo_start = self.GRNR, 
                                    ihalo_end = self.GRNR,
                                    blocks=('GPOS','MSTR','R25K','MVIR','RVIR'),
                                    use_cache=True ):
            # halo_center = halo["GPOS"]
            self.Mstar = halo["MSTR"][5] # stellar mass within R25K
            self.R25K = halo["R25K"]
            self.Mvir = halo["MVIR"]
            self.Rvir = halo["RVIR"]
            # print(halo["NSUB"])
            
            for subhalo in matcha.yield_subhaloes( self.groupbase,
                                    # with_ids=True, halo_ids=halo['ids'],
                                    # halo_goff=halo['GOFF'],
                                    ihalo=halo['ihalo'],
                                    blocks=('SPOS','SVEL','SSFR'),
                                    use_cache=True ):
                
                self.center = subhalo["SPOS"]
                self.Svel = subhalo["SVEL"]
                self.SFR = subhalo["SSFR"]
                break

    def get_halo_GRNR(self):
        halo_NSUB = np.array([])
        for k in range(self.numfiles):
            halo_NSUB = np.append( halo_NSUB, g3.read_new(f"{self.groupbase}.{k}", "NSUB", 0, is_snap=False) )
            if self.FSUB < np.sum(halo_NSUB):
                break
        for j,_ in enumerate(halo_NSUB): 
            if (np.sum(halo_NSUB[:j+1])-self.FSUB) > 0:
                break
        self.GRNR = j

    def halo_gas(self):
        return g3.read_particles_in_box(
            snap_file_name = self.snapbase,
            center = self.center, d = self.R25K,
            blocks = ["POS ", "VEL ", "MASS", "RHO ", "SFR ", "Zs  ", "HSML", "ID  "],
            ptypes = 0 )

    def halo_stars(self):
        return g3.read_particles_in_box(
            snap_file_name = self.snapbase, 
            center = self.center, d = self.R25K,
            blocks = ["POS ", "VEL ", "MASS", "iM  ", "AGE ", "Zs  ", "HSMS", "ID  "],
            ptypes = 4 )

    ### Cosmology

    def agez(self, z):
        return self.lcdm.age(z).to("Myr").value

    def age(self, a):
        """
        Age of universe given its scalefactor in Myr
        """
        return self.lcdm.age(1./a-1.).to("Myr").value

    def age_part(self, a):
        """
        Age of particle born at scalefactor 'a' in Myr
        """
        return ( self.age(self.aa_c) - self.age(a) )

    def lum_dist_a(self, a):
        """
        luminosity distance given scale factor a in Mpc
        """
        return self.lcdm.luminosity_distance(1./a-1.).value

    def lum_dist_z(self, z):
        """
        luminosity distance given scale factor a in Mpc
        """
        return self.lcdm.luminosity_distance(z).value

    def ang_dist_a(self, a):
        """
        angular distance given scale factor a in Mpc
        """
        return self.lcdm.angular_diameter_distance(1./a-1.).value
    
    def ang_dist_z(self, z):
        """
        angular distance given scale factor a in Mpc
        """
        return self.lcdm.angular_diameter_distance(z).value

    def pos_to_phys(self, pos):
        return pos * self.aa_c / self.h

    def vel_to_phys(self, vel):
        return vel * np.sqrt(self.aa_c)

    def mass_to_phys(self, mass):
        return mass * self.munit / self.h
    
    def dens_to_phys(self, rho):
        """
        Converts rho to g/cm^3
        """
        return rho * self.munit*1e33 / (self.KPC_TO_CM)**3 * self.h**2 * (1+self.zz_c)**3


def spec_ang_mom(mass, pos, vel):
    return np.sum( np.cross( (mass*pos.T).T, vel, axis=1 ), axis=0 ) / np.sum(mass) 

@njit#(["UniTuple(f4[:],2)(f4[:,:,:],f4[:,:,:],f4[:],f4)"])
def find_COM(Bpos, Bvel, Bmass, outer):
    """
    Shrinking sphere algorithm.
    -----
    Bpos, Bvel, Bmass: input positions, velocities and mass of particles
    outer: starting radius of shrinking sphere
    """

    pCOM = np.zeros(3,dtype=nb.f4)
    vCOM = np.zeros(3,dtype=nb.f4)
    tCOM = np.zeros(3,dtype=nb.f4)
    v_med = np.zeros(3,dtype=nb.f4)
    # rr = np.empty(Bpos.shape[0],dtype=nb.f4)

    rr = np.sqrt( Bpos[:,0]**2 + Bpos[:,1]**2 + Bpos[:,2]**2 )
    mask = (rr <= outer)
    n = Bpos[:,0][mask].shape[0]
    nlimit = min( 1000, int( np.ceil( 0.01*n ) ) )
    while n >= nlimit:
        pCOM[0] = np.sum( Bmass[mask]*Bpos[:,0][mask] ) / np.sum(Bmass[mask])
        pCOM[1] = np.sum( Bmass[mask]*Bpos[:,1][mask] ) / np.sum(Bmass[mask])
        pCOM[2] = np.sum( Bmass[mask]*Bpos[:,2][mask] ) / np.sum(Bmass[mask])
        Bpos = Bpos - pCOM
        tCOM = tCOM + pCOM
        rr = np.sqrt(Bpos[:,0]**2+Bpos[:,1]**2+Bpos[:,2]**2)
        outer = outer*(1-.025)

        mask = (rr <= outer)
        n = Bpos[:,0][mask].shape[0]

    v_med = np.array([np.median(Bvel[mask,0]), np.median(Bvel[mask,1]), np.median(Bvel[mask,2])])
    # v_med = np.median(Bvel[mask], axis=0)
    vv = np.sqrt((Bvel[mask,0]-v_med[0])**2+(Bvel[mask,1]-v_med[1])**2+(Bvel[mask,2]-v_med[2])**2)
    v_max = np.percentile(vv, .9)
    mask2 = (vv <= v_max)

    vCOM[0] = np.sum( Bmass[mask][mask2]*Bvel[mask,0][mask2] ) / np.sum(Bmass[mask][mask2])
    vCOM[1] = np.sum( Bmass[mask][mask2]*Bvel[mask,1][mask2] ) / np.sum(Bmass[mask][mask2])
    vCOM[2] = np.sum( Bmass[mask][mask2]*Bvel[mask,2][mask2] ) / np.sum(Bmass[mask][mask2])

    return (tCOM, vCOM)

@njit(["b1[:](i8[:],i8[:])"])
def get_index_list_bool(ptype_pid, halo_pid):
    """
    Finds index position of ptype_pid for particles belonging to halo
    Match particle ids that are bound to the halo, super fast!
    -----
    out:        Boolean mask array with length of ptype_pid.
                Entry is set to True if indeces match
                between halo and ptype.
    halo_pid:   IDs of particles bound to halo. dtype = np.int64
    ptype_pid:  IDs of particles of selected ptype in box. dtype = np.int64
    """
    out = np.empty(ptype_pid.shape[0], dtype=nb.boolean)
    halo_pid = set(halo_pid)
    for i in range(ptype_pid.shape[0]):
        if ptype_pid[i] in halo_pid:
            out[i]=True
        else:
            out[i]=False
    return out



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
    # redshift: float         = 0.01 # Observed redshift
    
    ##--- Derived from snapshot ---##
    Rshm: float             = 1.
    bVal: float             = 1.
    logOH12_s: float        = 8.69
    logFeH_s: float         = 0.
    Zgal_s: float           = 0.
    logOH12_g: float        = 8.69
    Zgal_g: float           = 0.
    
    ##--- Derived from which Magneticum Box ---##
    groupbase: str          = "/HydroSims/Magneticum/Box4/uhr_test/groups_136/sub_136"
    snapbase: str           = "/HydroSims/Magneticum/Box4/uhr_test/snapdir_136/snap_136"

    def __post_init__(self):
        self.sFSUB: str     = f"{self.FSUB:06d}"

        head            = g3.GadgetFile(self.snapbase+".0").header
        self.h          = head.HubbleParam
        self.Om0        = head.Omega0
        self.zz_c       = head.redshift
        self.aa_c       = 1. / (1. + self.zz_c)
        self.lcdm       = FlatLambdaCDM( H0 = self.h*100, Om0 = self.Om0 )
        self.boxsize    = head.BoxSize
        self.numfiles   = head.num_files

        self.get_halo_GRNR()
        self.get_halo_data()
        self.set_star_quants() # Rshm, bVal
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
        
    def mask_index_list(self, check_ids):
        for halo in matcha.yield_haloes( self.groupbase, with_ids=True, ihalo_start=self.GRNR, ihalo_end=self.GRNR, blocks=('GPOS','FSUB','NSUB'), use_cache=True ):
            i=0
            for subhalo in matcha.yield_subhaloes( self.groupbase, with_ids=True, halo_ids=halo['ids'], halo_goff=halo['GOFF'], ihalo=halo['ihalo'], blocks=('SOFF','SLEN'), use_cache=True ):
                if self.FSUB == halo['FSUB']+i:
                    break
                i+=1
        return get_index_list_bool(check_ids, subhalo['ids'])

    def set_star_quants(self):
        """
        Following description in Teklu+15, stellar half-mass radius from stars within 10% of Rvir
        returns half-mass radius in kpc/h
        """
        
        stars = self.get_stars()
        indlist = self.mask_index_list(stars["ID  "])
        mass = stars["MASS"][indlist]
        pos = self.pos_to_phys(stars["POS "][indlist] - self.center)
        vel = self.vel_to_phys(stars["VEL "][indlist]) - self.Svel
                
        st_rad =  g3.to_spherical(pos, [0,0,0]).T[0]
        less = st_rad <= .1*self.pos_to_phys(self.Rvir)
        
        st_mass = np.sum(mass[less])
        
        # Approximate Rshm from below
        r = 0.
        for dr in [1., .1, .01, .001, 1.e-4, 1.e-5]:
            while np.sum(mass[st_rad <= r+dr]) <= .5*st_mass:
                r += dr   

        pCOM, vCOM = find_COM(pos,vel,mass,5*r)
        
        st_rad =  g3.to_spherical(pos, pCOM).T[0]
        less = (st_rad <= .1*self.pos_to_phys(self.Rvir))
        st_mass = np.sum(mass[less])

        r = 0.
        for dr in [1., .1, .01, .001, 1.e-4, 1.e-5]:
            while np.sum(mass[st_rad <= r+dr]) <= .5*st_mass:
                r += dr

        self.Rshm = round(r,5)

        vel = vel - vCOM
        mask = ( st_rad <= 3.*self.Rshm )

        self.bVal = \
            np.log10( np.linalg.norm(spec_ang_mom(mass[mask], pos[mask], vel[mask])) ) \
            - 2./3. * np.log10(self.mass_to_phys(np.sum(mass[mask])))       

    def get_gas(self):
        return self.halo_gas()

    def get_gas_pos(self):
        gas = self.get_gas()
        pos = gas["POS "]

        return pos

    def get_stars(self):
        return self.halo_stars()

    def get_st_pos(self):
        stars = self.get_stars()
        pos = stars["POS "]

        return pos

    def set_st_met(self):
        stars = self.get_stars()
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
    
    def get_st_met_idv(self, bWeight: bool = False):
        """
        Returns metallicities for individual stars. Can also return masses for weights...
        """
        stars = self.get_stars()
        mass = stars["MASS"]
        iM = stars["iM  "]
        Zs = stars["Zs  "]
        
        Zstar = np.sum(Zs[:,1:],axis=1) / (iM - np.sum(Zs,axis=1)) / 0.02

        if bWeight:
            return (Zstar, mass)
        else:
            return Zstar

    def set_gas_met(self):
        gas = self.get_gas()
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

             

if __name__ == "__main__":

    Galaxy(5416)
    groupbase = "/home/lcladm/Studium/Masterarbeit/test/dorc/uhr_test/groups_136/sub_136"
    snapbase = "/home/lcladm/Studium/Masterarbeit/test/dorc/uhr_test/snapdir_136/snap_136"

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

