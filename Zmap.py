from PhoxUtil.galaxies import Galaxy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mclr
import g3read as g3
from astropy.table import Table

mpl.rcParams["backend"] = "Qt5Agg"
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["mathtext.rm"] = "serif"
mpl.rcParams["mathtext.bf"] = "serif:bold"
mpl.rcParams["mathtext.it"] = "serif:italic"
mpl.rcParams["mathtext.sf"] = "serif"
mpl.rcParams["mathtext.fontset"] = "dejavuserif"
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["ytick.labelsize"] = 12


# groupbase = "/home/lcladm/Studium/Masterarbeit/test/dorc/uhr_test/groups_136/sub_136"
# snapbase = "/home/lcladm/Studium/Masterarbeit/test/dorc/uhr_test/snapdir_136/snap_136"
groupbase = "/ptmp2/vladutescu/paper21/seed/919/groups_136/sub_136"
snapbase = "/ptmp2/vladutescu/paper21/seed/919/snapdir_136/snap_136"
phbase = "/home/lcladm/Studium/Masterarbeit/R136_AGN_fix/fits/"
phbase_2 = "/home/lcladm/Studium/Masterarbeit/R333/fits/"
phbase_3 = "./fits_g/"
phbase_new = "/ptmp2/vladutescu/paper21/seed/919/fits/"


head = g3.GadgetFile(snapbase+".0").header
h = head.HubbleParam
zz = head.redshift

gal_dict = Galaxy.gal_dict_from_npy("gal_data.npy")
x = gal_dict["013633"]
x.snapbase = snapbase
Z,m = x.get_st_met_idv(bWeight=True)
stars = x.get_stars()
# gas = x.get_gas()
# sfr = gas["SFR "]
# print(sfr[sfr>0])
# print(np.median(sfr[sfr>0]),np.percentile(sfr[sfr>0],(16,86)))
pos = x.pos_to_phys(stars["POS "] - x.center)
age = stars["AGE "]
hsms = x.pos_to_phys( stars["HSMS"] )
# hsml = x.pos_to_phys( gas["HSML"] )
rad = g3.to_spherical(pos,[0,0,0]).T[0]
R25K = x.pos_to_phys(x.R25K)
print(R25K,x.center)
# print(f"{x.Mstar:.2e}")
mask = (rad < R25K*5) #& (x.age_part(age)<100) 
xpos = pos[:,0][mask]
ypos = pos[:,1][mask]
pos = pos[mask]
hsms = hsms[mask]
age = age[mask]
# print(hsms)
# print(hsml)
Z = Z[mask]
m = x.mass_to_phys(m[mask])
fpH = phbase_new+f"gal{x.FSUB:0>6d}HXB.fits"
fpL = phbase_new+f"gal{x.FSUB:0>6d}LXB.fits"
tblH = Table.read(fpH)
tblL = Table.read(fpL)
xphH = tblH["POS_X"]
yphH = tblH["POS_Y"]
xphL = tblL["POS_X"]
yphL = tblL["POS_Y"]

import PhoxUtil.grid2D as g2D

Nbins = 128*2
R25K = R25K/2
mapPar = g2D.mappingParam([0,0,0],R25K,Nbins)
edges = np.linspace(-R25K/2.,R25K/2.,Nbins)
extent=[np.amin(edges),np.amax(edges),np.amin(edges),np.amax(edges)]
Apx = (mapPar["LEN_PER_PX"])**2
Vpx = Apx * mapPar["LEN_PER_PX"]

from scipy.stats import gaussian_kde
    
kH = gaussian_kde(np.vstack([xphH, yphH]))
kL = gaussian_kde(np.vstack([xphL, yphL]))
xi, yi = np.mgrid[extent[0]:extent[1]:100*1j,extent[2]:extent[3]:100*1j]
ziH = kH(np.vstack([xi.flatten(), yi.flatten()]))
ziL = kL(np.vstack([xi.flatten(), yi.flatten()]))

#set zi to 0-1 scale inverted
ziH = 1-(ziH-ziH.min())/(ziH.max() - ziH.min())
ziH = ziH.reshape(xi.shape)
ziL = 1-(ziL-ziL.min())/(ziL.max() - ziL.min())
ziL = ziL.reshape(xi.shape)

#set up plot
origin = 'lower'
levels = [0.68,.9,.95,.99]


# hm = hsms>2.8*np.mean(hsms)
# 1/(4/3*hsms[:]**3*3.141526)
hsms = hsms/4.
rho = m[:]/(4/3*(hsms[:]/1)**3*3.141526)
ones = np.ones_like(m[:]/rho[:]/hsms[:]**3.)
print(len(ones))

iMq, iMw = g2D.mapping2D(pos[:],hsms[:],rho[:],ones[:],mapPar)
# iMq, iMw = g2D.mapping2D(gas["POS "]-x.center,hsml[:],gas["RHO "],np.ones_like(gas["MASS"]/(gas["RHO "])/hsml[:]**3.),mapPar)
iMn = np.nan_to_num(iMq / iMw)
print(f"'Zmap': sum of mass array given to 'g2D': {np.sum(m):.2e}")
mass_in_im = np.sum(iMq*Vpx)
print(f"'Zmap': sum of every pixel in image = {mass_in_im:.2e}")
fig2 = plt.figure(figsize=(6,6))
ax = fig2.add_subplot(111)
# im1 = ax[0].imshow(iMq,extent=extent,origin='lower',norm=mclr.LogNorm(vmin=5.e-4,clip=True),interpolation='none')
im2 = ax.imshow(iMq,extent=extent,origin='lower',norm=mclr.LogNorm(),interpolation='none',cmap="viridis")
csH = ax.contour(xi, yi, ziH, levels = levels,
            linestyles=['solid','dashed','dashdot','dotted'], # iterable so that each level has different style
            colors=['k'], # iterable so that each level has same color
            origin=origin,
            extent=extent)
csL = ax.contour(xi, yi, ziL, levels = levels,
            linestyles=['solid','dashed','dashdot','dotted'], # iterable so that each level has different style
            colors=['r'], # iterable so that each level has same color
            origin=origin,
            extent=extent)

ax.clabel(csH, fmt=(lambda x: f'{x*100:.0f}%'), fontsize=8)
ax.clabel(csL, fmt=(lambda x: f'{x*100:.0f}%'), fontsize=8)
csH.collections[0].set_label("HXB")
csL.collections[0].set_label("LXB")
# cb = plt.colorbar(im2, shrink=.7)
# cb.set_label(label=r"scalefactor",fontsize=14)
# cb.ax.get_yaxis().set_major_formatter(lambda x, _: f'{x:g}')

plt.xlabel("kpc", fontsize=14.)
plt.ylabel("kpc", fontsize=14.)
plt.legend()
plt.tight_layout(pad=0.15)
plt.show()
