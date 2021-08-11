import galaxies as gal
from galaxies import Galaxy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mclr
import g3read as g3
from astropy.table import Table


groupbase = "/home/lcladm/Studium/Masterarbeit/test/dorc/uhr_test/groups_136/sub_136"
snapbase = "/home/lcladm/Studium/Masterarbeit/test/dorc/uhr_test/snapdir_136/snap_136"
phbase = "/home/lcladm/Studium/Masterarbeit/R136_AGN_fix/fits/"
phbase_2 = "/home/lcladm/Studium/Masterarbeit/R333/fits/"
phbase_3 = "./fits_g/"


head = g3.GadgetFile(snapbase+".0").header
h = head.HubbleParam
zz = head.redshift

gal_dict = Galaxy.gal_dict_from_npy("gal_data.npy")
for key in gal_dict.keys():
    # if int(key) == 13633:
    if int(key) == 10859:
        x = gal_dict[key]
        Z,m = x.get_st_met_idv(bWeight=True)
        stars = x.get_stars()
        gas = x.get_gas()
        pos = x.pos_to_phys(stars["POS "] - x.center)
        age = stars["AGE "]
        hsms = x.pos_to_phys( stars["HSMS"] )
        hsml = x.pos_to_phys( gas["HSML"] )
        rad = g3.to_spherical(pos,[0,0,0]).T[0]
        R25K = x.pos_to_phys(x.R25K)
        print(R25K,x.center)
        # print(f"{x.Mstar:.2e}")
        mask = (rad < R25K*5) #& (x.age_part(age)>1000) 
        xpos = pos[:,0][mask]
        ypos = pos[:,1][mask]
        pos = pos[mask]
        hsms = hsms[mask]
        age = age[mask]
        # print(hsms)
        # print(hsml)
        Z = Z[mask]
        m = x.mass_to_phys(m[mask])
        fp = phbase_2+"gal"+str(x.GRNR)+"HXRB.fits"
        tbl = Table.read(fp)
        xph = tbl["POS_X"]
        yph = tbl["POS_Y"]
    else:
        continue

import grid2D as g2D

Nbins = 128*2
R25K = R25K/1
mapPar = g2D.mappingParam([0,0,0],R25K,Nbins)
edges = np.linspace(-R25K/2.,R25K/2.,Nbins)
extent=[np.amin(edges),np.amax(edges),np.amin(edges),np.amax(edges)]
Apx = (mapPar["LEN_PER_PX"])**2
Vpx = Apx * mapPar["LEN_PER_PX"]

xray,_ ,_ = np.histogram2d(xph,yph,bins=[edges,edges])
xray = xray.T

# hm = hsms>2.8*np.mean(hsms)
# 1/(4/3*hsms[:]**3*3.141526)
# hsms = hsms/4.
rho = m[:]/(4/3*(hsms[:]/1)**3*3.141526)
ones = np.ones_like(m[:]/rho[:]/hsms[:]**3.)

iMq, iMw = g2D.mapping2D(pos[:],hsms[:],rho[:],ones[:],mapPar)
# iMq, iMw = g2D.mapping2D(gas["POS "]-x.center,hsml[:],gas["RHO "],np.ones_like(gas["MASS"]/(gas["RHO "])/hsml[:]**3.),mapPar)
iMn = np.nan_to_num(iMq / iMw)
print(f"'Zmap': sum of mass array given to 'g2D': {np.sum(m):.2e}")
mass_in_im = np.sum(iMq*Vpx)
print(f"'Zmap': sum of every pixel in image = {mass_in_im:.2e}")
fig2 = plt.figure(figsize=(6,6))
ax = fig2.add_subplot(111)
# im1 = ax[0].imshow(iMq,extent=extent,origin='lower',norm=mclr.LogNorm(vmin=5.e-4,clip=True),interpolation='none')
im2 = ax.imshow(iMq,extent=extent,origin='lower',norm=mclr.LogNorm(vmin=5.e-4,clip=True),interpolation='none')
# plt.colorbar(im1,ax=ax[0])
plt.colorbar(im2)
plt.contour(xray,extent=extent,origin='lower',cmap='Reds',levels=5,norm=mclr.LogNorm(vmin=.02,vmax=1000.,clip=True),alpha=.7)
plt.xlabel("kpc", fontsize=14.)
plt.ylabel("kpc", fontsize=14.)

plt.show()

counts, _, _ = np.histogram2d(xpos,ypos,bins=[edges,edges])
tM, _, _ = np.histogram2d(xpos,ypos,bins=[edges,edges],weights=m)
quant, _, _ = np.histogram2d(xpos,ypos,bins=[edges,edges],weights=Z*m)
# zeros = (counts == 0)
non_zeros = (counts > 0)
data = np.zeros_like(counts)
data[non_zeros] = quant[non_zeros] / tM[non_zeros] #/ counts[non_zeros]
# data[zeros] = 1.e-5
data = data.T
# print(data)
im = plt.imshow(data,extent=extent,origin='lower',norm=mclr.LogNorm(vmin=5.e-4,clip=True),interpolation='none')
cb = plt.colorbar()
cb.set_label(label=r"$\log(\,Z\,/\,Z_{\odot})$",fontsize=14)
cb.ax.get_yaxis().set_major_formatter(lambda x, _: f"{np.log10(x):g}")
plt.contour(xray,extent=extent,origin='lower',cmap='Reds',levels=5,norm=mclr.LogNorm(vmin=.02,vmax=1000.,clip=True),alpha=.7)
plt.xlabel("kpc", fontsize=14.)
plt.ylabel("kpc", fontsize=14.)
plt.show()

quant = quant.T
im2 = plt.imshow(quant,extent=extent,origin='lower',norm=mclr.LogNorm(vmin=5.e-4,clip=True),interpolation='gaussian')
cb = plt.colorbar()
cb.set_label(label=r"$Z\,/\,Z_{\odot}*m$",fontsize=14)
cb.ax.get_yaxis().set_major_formatter(lambda x, _: f"{np.log10(x*1.e10/h):.2f}")
# plt.contour(xray,extent=[np.amin(edges),np.amax(edges),np.amin(edges),np.amax(edges)],origin='lower',cmap='Reds',levels=5,norm=mclr.LogNorm(vmin=.02,vmax=1000.,clip=True),alpha=.7)
plt.xlabel("kpc", fontsize=14.)
plt.ylabel("kpc", fontsize=14.)
plt.show()

tM = tM.T
im2 = plt.imshow(tM,extent=extent,origin='lower',norm=mclr.LogNorm(vmin=5.e-4,clip=True),interpolation='gaussian')
cb = plt.colorbar()
cb.set_label(label=r"$\log(M_*\,/\,M_{\odot})$",fontsize=14)
cb.ax.get_yaxis().set_major_formatter(lambda x, _: f"{np.log10(x*1.e10/h):.2f}")
mass_in_im = np.sum(tM)*1.e10/h
print(f"{mass_in_im:.2e}")
# plt.contour(xray,extent=[np.amin(edges),np.amax(edges),np.amin(edges),np.amax(edges)],origin='lower',cmap='Reds',levels=5,norm=mclr.LogNorm(vmin=.02,vmax=1000.,clip=True),alpha=.7)
plt.xlabel("kpc", fontsize=14.)
plt.ylabel("kpc", fontsize=14.)
plt.show()

counts = counts.T
im2 = plt.imshow(counts,extent=extent,origin='lower',norm=mclr.LogNorm(vmin=1,clip=True),interpolation='gaussian')
cb = plt.colorbar()
cb.set_label(label=r"counts per px",fontsize=14)
cb.ax.get_yaxis().set_major_formatter(lambda x, _: f"{x:g}")
# plt.contour(xray,extent=[np.amin(edges),np.amax(edges),np.amin(edges),np.amax(edges)],origin='lower',cmap='Reds',levels=5,norm=mclr.LogNorm(vmin=.02,vmax=1000.,clip=True),alpha=.7)
plt.xlabel("kpc", fontsize=14.)
plt.ylabel("kpc", fontsize=14.)
plt.show()


