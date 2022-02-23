import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mclr
import g3read as g3

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

head = g3.GadgetFile(snapbase+".0").header
h = head.HubbleParam
print(head.BoxSize)
zz = head.redshift

lpos = []
lmass = []
lhsms = []
for k in range(head.num_files):
    # stars = g3.read_new(f"{snapbase}.{k}",["MASS","HSMS","POS "],4,is_snap=True)
    stars = g3.read_new(f"{snapbase}.{k}",["MASS","HSML","POS "],0,is_snap=True)
    # stars = g3.read_particles_in_box(snapbase,[0,0,0],R25K,["MASS","HSMS","POS "],4)
    lpos.append(stars["POS "])
    lmass.append(stars["MASS"])
    lhsms.append(stars["HSML"])
pos = np.concatenate(lpos)-np.array([head.BoxSize,head.BoxSize,head.BoxSize])/2.
mass = np.concatenate(lmass)
hsms = np.concatenate(lhsms)
rho = mass[:]/(4/3*(hsms[:]/1)**3*3.141526)
ones = np.ones_like(mass)

import grid2D as g2D

Nbins = 128*10*1
R25K = head.BoxSize
mapPar = g2D.mappingParam([0,0,0],R25K,Nbins)
edges = np.linspace(-R25K/2.,R25K/2.,Nbins)
extent=[np.amin(edges),np.amax(edges),np.amin(edges),np.amax(edges)]
Apx = (mapPar["LEN_PER_PX"])**2
Vpx = Apx * mapPar["LEN_PER_PX"]

# plt.plot(pos[:,0],pos[:,1],lw=0.,marker="o")
# plt.show()

# pos = np.array([[0,0,0]])
# hsms = np.array([400])
# rho = np.array([1])
# ones = np.array([1])

iMq, iMw = g2D.mapping2D(pos[:],hsms[:],rho[:],ones[:],mapPar)
# iMq, iMw = g2D.mapping2D(gas["POS "]-x.center,hsml[:],gas["RHO "],np.ones_like(gas["MASS"]/(gas["RHO "])/hsml[:]**3.),mapPar)
iMn = np.nan_to_num(iMq / iMw)
print(f"'Zmap': sum of mass array given to 'g2D': {np.sum(mass):.2e}")
mass_in_im = np.sum(iMq*Vpx)
print(f"'Zmap': sum of every pixel in image = {mass_in_im:.2e}")
fig2 = plt.figure(figsize=(9,9))
ax = fig2.add_subplot(111)
ax.set_xticks([])
ax.set_yticks([])
# im1 = ax[0].imshow(iMq,extent=extent,origin='lower',norm=mclr.LogNorm(vmin=5.e-4,clip=True),interpolation='none')
im2 = ax.imshow(iMq,extent=extent,origin='lower',norm=mclr.LogNorm(vmin=1e-5*iMq.max(),clip=True),interpolation='none',cmap="viridis")
plt.tight_layout()
plt.show() 
