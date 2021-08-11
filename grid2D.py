
import numpy as np
from numba import njit
import g3read_units as g3u


###----- UNIT HANDLER -----###
class Uregexpr(object):
    def __init__(self, expression):
        self.expression = expression
    def evaluate(self, ureg):
        return ureg.parse_expression(self.expression)
    def __str__(self):
        return 'ureg '+self.expression

# type of input paramters, Uregexpr means the input may be  a string like '1000. kpc'
types={
    "CENTER_X": Uregexpr,
    "CENTER_Y": Uregexpr,
    "CENTER_Z": Uregexpr,
    "IMG_XY_SIZE": Uregexpr,
    "IMG_Z_SIZE": Uregexpr,
    "IMG_SIZE": int,

    }

xyz_keys = ['CENTER_X', 'CENTER_Y', 'CENTER_Z'] 

def parse_ureg_expressions(kv, to_parse, ureg):
    """
    Reads input values with units (e.g. '3000 kpc') and parses it in a  `pint` value).
    Input values came from the config file and are passed in a dict `kv`.
    `to_parse` is a dictionary that stores the type of the value of a given key (e.g. the number of pixels IMG_SIZE is `int`),
    and `ureg` is a `pint` instance
    """  
    res = {}
    for k in to_parse:
        res[k] = kv[k].evaluate(ureg)
    return res



###----- KERNELS -----###
@njit
def WendlandC6_2D(u: float, h_inv: float):
    """
    Evaluate the WendlandC6 spline at position u, where u:= x*h_inv, for a 2D projection.
    Values correspond to 295 neighbours.
    """

    norm2D = 78. / (7. * 3.1415926)

    if u < 1.0:
        n = norm2D * h_inv**2
        u_m1 = (1.0 - u)
        u_m1 = u_m1 * u_m1  # (1.0 - u)^2
        u_m1 = u_m1 * u_m1  # (1.0 - u)^4
        u_m1 = u_m1 * u_m1  # (1.0 - u)^8
        u2 = u*u
        return ( u_m1 * ( 1.0 + 8*u + 25*u2 + 32*u2*u )) * n
    else:
        return 0.


###----- GRID -----###
@njit
def calc_area_weights(dw: np.ndarray,
                masked_x: np.ndarray, masked_y: np.ndarray, 
                masked_h: np.ndarray, kernel,
                bin_min: int, delta_bin: float, n_bins: int):

    N = len(masked_x)

    tot_max_bin_spread = 0

    for k in range(N):
        xpx = (masked_x[k]-bin_min)/delta_bin #pos of x in terms of px (from 0 to n_bins-1)
        ypx = (masked_y[k]-bin_min)/delta_bin #pos of y in terms of px
        bin_i = int(xpx) #calculates distance in i-direction from minimum bin
        bin_j = int(ypx) #calculates distance in j-direction
        Hsml = masked_h[k]/delta_bin # in number of px
        
        max_bin_spread = int(Hsml)

        tot_max_bin_spread+=max_bin_spread
        distr_weight = 0.

        for i in range(bin_i - max_bin_spread, bin_i + max_bin_spread + 1):
            if i>=n_bins or i<0:
                continue
            for j in range(bin_j - max_bin_spread, bin_j + max_bin_spread + 1):
                if j>=n_bins or j<0:
                    continue
                u = ( (i+.5-xpx)**2 + (j+.5-ypx)**2 )**.5 / Hsml # distance of particle to px center
                if u > 1.0:
                    continue
                
                dx = min(xpx + Hsml, i+1) - max(xpx-Hsml, i)
                dy = min(ypx + Hsml, j+1) - max(ypx-Hsml, j)
                dxdy = dx*dy # area of px

                if Hsml < 1.:
                    wk = dxdy
                else:
                    wk = kernel(u, 1./Hsml) * dxdy
                
                distr_weight += wk

        dw[k] += distr_weight
    if N>0:
        return tot_max_bin_spread/N
    else:
        return np.nan

@njit
def add_to_grid(final_image_t: np.ndarray, weight_image: np.ndarray,
                masked_x: np.ndarray, masked_y: np.ndarray, 
                masked_h: np.ndarray, masked_w: np.ndarray, masked_q: np.ndarray, 
                dz: float, kernel, distr_weight: np.ndarray,
                bin_min: int, delta_bin: float, n_bins: int):
    """
    This routine adds a chunk of particles with sky positions `masked_x, masked_y`, 
    smoothing length  `masked_h,` and value `masked_w` (e.g. paticle mass) to a FIT buffer `finalt_image_t`.
    Data is inserted in chunks in order to be able to process objects that do not fit into memory.
    """
    
    N = len(masked_x)

    tot_max_bin_spread = 0

    for k in range(N):
        xpx = (masked_x[k]-bin_min)/delta_bin #pos of x in terms of px (from 0 to n_bins-1)
        ypx = (masked_y[k]-bin_min)/delta_bin #pos of y in terms of px
        bin_i = int(xpx) #calculates distance in i-direction from minimum bin
        bin_j = int(ypx) #calculates distance in j-direction
        Hsml = masked_h[k]/delta_bin # in number of px
        
        max_bin_spread = int(Hsml)
        temp =0

        kernel_norm = 3.1415926 * Hsml**2 # area of particle in px^2

        tot_max_bin_spread+=max_bin_spread
        q = masked_q[k]

        if q == 0:
            continue

        for i in range(bin_i - max_bin_spread, bin_i + max_bin_spread + 1):
            if i>=n_bins or i<0:
                continue
            for j in range(bin_j - max_bin_spread, bin_j + max_bin_spread + 1):
                if j>=n_bins or j<0:
                    continue
                u = ( (i+.5-xpx)**2 + (j+.5-ypx)**2 )**.5 / Hsml # distance of particle to px center
                if u > 1.0:
                    continue
                
                dx = min(xpx + Hsml, i+1) - max(xpx - Hsml, i)
                dy = min(ypx + Hsml, j+1) - max(ypx - Hsml, j)
                dxdy = dx*dy # area of px

                if Hsml < 1.:
                    wk = dxdy
                else:
                    wk = kernel(u, 1./Hsml) * dxdy

                temp += wk

                area_norm = kernel_norm / distr_weight[k] * masked_w[k] * dz[k] / delta_bin
                px_weight = area_norm * wk
                # print(f"xpx = {xpx:.3f}, ypx = {ypx:.3f},\nbin_i = {bin_i}, bin_j = {bin_j},\ndx = {dx}, dy = {dy},\nmax_bin_spread = {max_bin_spread},\ni = {i}, j = {j},\nHsml = {Hsml}, u = {u:.3f},\nwk = {wk:.3f}, temp = {temp:.6f}\npx_weight = {px_weight:.3f},\nkernel_norm = {kernel_norm:.3f},\ndistr_weight[k] = {distr_weight[k]:.6f}")
                # input()
                final_image_t[j][i] += q * px_weight
                weight_image[j][i] += px_weight
                #Implement different contributions based on bin distance (kernels)... Basic idea would be to see how large the contribution of each particle is to each pixel is
    if N>0:
        return tot_max_bin_spread/N
    else:
        return np.nan


def mapping2D(pos: np.ndarray, hsml: np.ndarray,
              qty: np.ndarray, weights: np.ndarray, 
              mapParam: dict, kernel = WendlandC6_2D):
    """
    here we read input data `kv`, read chunks of particles and send them to `add_to_grid`
    """
    #set the snapshots' scalefactor and hubble factor into units.
    # units = g3u.get_units(mapParam['SNAP_PATH'])
    #produce a set of pint units from the snapshots data
    # ureg = units.get_u()

    img_xy_size = mapParam['IMG_XY_SIZE']
    img_z_size  = mapParam['IMG_Z_SIZE']
    img_pxsize = mapParam['IMG_PXSIZE'] # number of pixels per side
    img_center = np.array(mapParam['IMG_CENTER'])

    
    n_bins = [img_pxsize]*2
    final_image_t = np.zeros(n_bins)
    weight_image = np.zeros(n_bins)
    dw = np.zeros_like(qty) 

    bins = [np.linspace(-.5,.5,img_pxsize+1), np.linspace(-.5, .5, img_pxsize+1)]
    
    N_part_type = len(qty)
    print(N_part_type)

    if N_part_type == 0:
        raise ValueError("Quantity array is empty... Abort")

    
    rel_poses = pos - img_center

    pos_x = rel_poses[:,0]
    pos_y = rel_poses[:,1]
    pos_z = rel_poses[:,2]
    
    # Convert into pixel position
    norm_x = (pos_x/img_xy_size)#.to('').magnitude 
    norm_y = (pos_y/img_xy_size)#.to('').magnitude 
    norm_z = (pos_z/img_z_size)#.to('').magnitude 
    norm_h = (hsml/img_xy_size)#.to('').magnitude

    # Select indeces for cube around center based on image size
    mask = (norm_x<=.5)&(norm_x>=-.5)&(norm_y<=.5)&(norm_y>=-.5)&(norm_z<=.5)&(norm_z>=-.5) 
            
    masked_x = norm_x[mask]
    masked_y = norm_y[mask]
    masked_h = norm_h[mask]
    masked_w = weights[mask]
    dz       = 4./3.* masked_h
    masked_q = qty[mask]#(qty[mask]).to(mapParam['RESULT_UNITS']).magnitude
    print(f"'g2D': sum of qty_arr = {np.sum(masked_q):.2e}")
    print(f"'g2D': sum of qty_arr * V_arr = {np.sum(masked_q*4/3*3.1415926*hsml[mask]**3.):.2e}")

    avg_bin_spread = calc_area_weights(dw,masked_x,masked_y,masked_h,kernel,bins[0][0], bins[0][1]-bins[0][0], img_pxsize)

    avg_bin_spread = add_to_grid(final_image_t, weight_image, masked_x, masked_y, masked_h, masked_w, masked_q, dz, kernel, dw, bins[0][0], bins[0][1]-bins[0][0], img_pxsize)
    print ('# avg bin spread ', avg_bin_spread)
    
    final_image = np.nan_to_num(final_image_t)

    return final_image, weight_image

def mappingParam(center: list[3], xy_size: float, Npx: int):
    mapPar = {
        "IMG_CENTER"    : center,
        "IMG_XY_SIZE"   : xy_size,
        "IMG_Z_SIZE"    : xy_size,
        "IMG_PXSIZE"    : Npx,
        "LEN_PER_PX"    : xy_size/Npx
    }
    return mapPar
