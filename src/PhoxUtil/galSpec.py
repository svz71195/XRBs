import numpy as np
import matplotlib as mpl
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.colors as mclr
import matplotlib.ticker as ticker
# import grid2D as g2D
import PhoxUtil.xrb_main as xm
xrb = xm.Mineo12S()
model = xrb.model(bLum=True)[0]*1.e38
xrb2 = xm.Lehmer21()
model2 = xrb2.model(bLum=True)[0]*1.e38
from PhoxUtil.galaxies import Galaxy
from PhoxUtil.phox_data import *
from PhoxUtil.Utils import calc_bin_centers, calc_flux_per_bin, calc_lum_cgs, get_num_XRB
from astropy.table import Table
from tqdm import tqdm

### UNIT CONVERSION ###
KEV_TO_ERG = 1.60218e-9
KPC_TO_CM = 3.0856e21
MPC_TO_CM = 3.0856e24

### COMPONENT PLOT KWARGS ###
clrAGN = 'crimson'
clrLXB = 'yellowgreen'
clrHXB = 'darkgreen'
clrGAS = 'royalblue'

# print("Done Importing Modules")

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


def yield_photon_energy(filename: str, Emin=0.,Emax=50.1):
    """
    Returns photon energies of fits file found in directory
    """
    try:
        tbl = Table.read(filename)
    except FileNotFoundError:
        print(f"Could not locate {filename}, skipping...")
        return np.array([])
    try:
        phE = tbl["PHOTON_ENERGY"]
    except KeyError:
        print(f"Could not find photon energies in file {filename}")
        return np.array([])
    
    return phE[(phE>=Emin)&(phE<=Emax)]

def yield_photon_pos(filename: str):
    """
    Returns photon energies of fits file found in directory
    """
    try:
        tbl = Table.read(filename)
    except FileNotFoundError:
        print(f"Could not locate {filename}, skipping...")
        return np.array([]), np.array([])
    try:
        pos_x = tbl["POS_X"]
        pos_y = tbl["POS_Y"]
    except KeyError:
        print(f"Could not find photon energies in file {filename}")
    
    return pos_x,pos_y
    

def plot_contour_log(xdata: np.ndarray, ydata: np.ndarray, color: str,
                cmap: str = None,
                kde_range: list = None,
                ax = None):
    """
    Generate 2D-contour plot of x,y data
    -----
    kde_range: [xmin,xmax,ymin,ymax]
    """
    from scipy.stats import gaussian_kde
    xdata = np.log10(xdata)
    ydata = np.log10(ydata)
    if xdata.size < 500:
        sizex = xdata.size
    else:
        sizex = 500
    if ydata.size < 500:
        sizey = ydata.size
    else:
        sizey = 500
    
    if kde_range is None:
        xmin = xdata.min()
        xmax = xdata.max()
        ymin = ydata.min()
        ymax = ydata.max()
    else:
        xmin = kde_range[0]
        xmax = kde_range[1]
        ymin = kde_range[2]
        ymax = kde_range[3]
    
    k = gaussian_kde(np.vstack([xdata, ydata]))
    xi, yi = np.mgrid[xmin:xmax:sizex*1j,ymin:ymax:sizey*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    #set zi to 0-1 scale inverted
    zi = 1-(zi-zi.min())/(zi.max() - zi.min())
    zi = zi.reshape(xi.shape)

    #set up plot
    origin = 'lower'
    levels = [0.,0.68,.9,.95,.99]

    if ax is None:
        fig = plt.figure(figsize=(5,5),dpi=500)
        ax = fig.add_subplot(111)

    if cmap is None:
        CS = ax.contour(xi, yi, zi,levels = levels,
                    linestyles=['solid','solid','dashed','dashdot','dotted'], # iterable so that each level has different style
                    colors=color, # iterable so that each level has same color
                    origin=origin,
                    extent=[xdata.min(),xdata.max(),ydata.min(),ydata.max()])
        ax.clabel(CS, fmt=(lambda x: f'{x*100:.0f}%'), fontsize=8)

    if cmap is not None:
        CS = ax.contourf(xi, yi, zi,levels = levels,
                    # linestyles=['solid'], # iterable so that each level has different style
                    # colors=color, # iterable so that each level has same color
                    cmap=cmap,
                    norm=mclr.Normalize(vmin=.6,vmax=1.2),
                    origin=origin,
                    alpha=.5,
                    extent=[xdata.min(),xdata.max(),ydata.min(),ydata.max()]) 

        # ax.clabel(CS, fmt=(lambda x: f'{x*100:.0f}%'), fontsize=8)               

    # ax.clabel(CS, fmt=(lambda x: f'{x*100:.0f}%'), fontsize=8)
    
    return ax

def plot_contour(xdata: np.ndarray, ydata: np.ndarray, color: str,
                kde_range: list = None, ax=None):
    """
    Generate 2D-contour plot of x,y data
    -----
    kde_range = [xmin,xmax,ymin,ymax]
    """
    from scipy.stats import gaussian_kde
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    if xdata.size < 500:
        sizex = xdata.size
    else:
        sizex = 500
    if ydata.size < 500:
        sizey = ydata.size
    else:
        sizey = 500

    if kde_range is None:
        xmin = xdata.min()
        xmax = xdata.max()
        ymin = ydata.min()
        ymax = ydata.max()
    else:
        xmin = kde_range[0]
        xmax = kde_range[1]
        ymin = kde_range[2]
        ymax = kde_range[3]
    
    k = gaussian_kde(np.vstack([xdata, ydata]))
    xi, yi = np.mgrid[xmin:xmax:sizex*1j,ymin:ymax:sizey*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    #set zi to 0-1 scale inverted
    zi = 1-(zi-zi.min())/(zi.max() - zi.min())
    zi = zi.reshape(xi.shape)

    #set up plot
    origin = 'lower'
    levels = [0.68,.9,.95,.99]

    if ax is None:
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)

    CS = ax.contour(xi, yi, zi,levels = levels,
                linestyles=['solid','dashed','dashdot','dotted'], # iterable so that each level has different style
                colors=[color], # iterable so that each level has same color
                origin=origin,
                extent=[xdata.min(),xdata.max(),ydata.min(),ydata.max()])

    ax.clabel(CS, fmt=(lambda x: f'{x*100:.0f}%'), fontsize=8)
    
    return ax


def broken_Lx_SFR(SFR: float):
    """
    Lx-SFR relation for discrete number of point sources
    following Gilfanov, Grimm, Sunyaev 2004
    -----
    xrb: class object of a xrb model
    """
    
    SFR_break = xrb.Lcut**(xrb.gamma-1.)/(xrb.xi/(2.-xrb.gamma))
    Lx = model # xrb.model(bLum=True)[0]*1.e38
    if SFR < SFR_break:
        return Lx*SFR**(1/(xrb.gamma-1))/SFR_break**((2.-xrb.gamma)/(xrb.gamma-1))
    else:
        return Lx*SFR

def broken_Lx_SFR_Lehm(SFR: float):
    """
    Lx-SFR relation for discrete number of point sources
    following Gilfanov, Grimm, Sunyaev 2004
    -----
    xrb: class object of a xrb model
    """
    SFR_break = (10**(xrb2.logLc-38))**(xrb2.g1_h-1.)/(xrb2.A_h/(2.-xrb2.g1_h))
    Lx = model2 # xrb2.model(bLum=True)[0]*1.e38
    if SFR < SFR_break:
        return Lx*SFR**(1/(xrb2.g1_h-1))/SFR_break**((2.-xrb2.g1_h)/(xrb2.g1_h-1))
    else:
        return Lx*SFR

def plot_Lx_SFR(gald: dict, bGAS: bool = False):
    """
    Plot Lx-SFR relation for HMXB emission
    based on Lx obtained from fits files and SFR from
    galaxy sample
    """
    m=(gald.sfr_arr>.1)
    m=m&(gald.mstar_arr>3.5e10)#&(gald.mstar_arr<1e10)
    n=(gald.LxG_arr>0.)
    nn=(gald.LxZ_arr>0)
    nm=(gald.LxH_arr>0.)
    # SFR = np.array(SFR[m])
    # Lx = np.array(Lx[m])
    vSFR = np.logspace(-2.5,2.5,1000)

    if bGAS:
        ax = plot_contour_log(gald.sfr_arr[m&n],gald.LxG_arr[m&n],clrGAS,kde_range=[-3,2.5,36,43])
        ax.plot(np.log10(gald.sfr_arr[m&n]),np.log10(gald.LxG_arr[m&n]),lw=0.,c="g",marker=".",ms=3.,alpha=.5)
        # ax2 = ax.twiny()
        # ax2.axis([9.5,12.5,36.5,43])
        # ax2 = plot_contour_log(gald.mstar_arr[m&n],gald.LxG_arr[m&n],color="k",cmap="Reds_r",kde_range=[9.5,12.5,36,43],ax=ax2)
        ax = plot_gas_sample(ax)
        ax.set_xticks([-2,-1.,0.,1.,2.])
        # ax.axis([np.log10(SFR.min()),np.log10(SFR.max()),np.log10(Lx.min()),np.log10(Lx.max())])
        ax.axis([-2,2.5,36.5,43])
        # ax2.axis([9.5,12.5,36.5,43])
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{10**y:g}'))
        ax.set_xlabel(r"$\mathrm{SFR}\, [M_{\odot}\,\mathrm{yr}^{-1}]$",fontsize=14)
        ax.set_ylabel(r"$\log L_{X,\mathrm{gas}}^{0.5-2\,\mathrm{keV}}\,[\mathrm{erg}\,\,\mathrm{s}^{-1}]$",fontsize=14)
        # popt, pcov = curve_fit(lambda x,A,B: A+B*x,np.log10(SFR),np.log10(Lx),[40.5,1],bounds=([39,.5],[41,2]))#,sigma=1/np.sqrt(SFR))
        # popt2, pcov2 = curve_fit(lambda x,A: A+x,np.log10(SFR),np.log10(Lx),[40.5],sigma=1/np.sqrt(SFR))
        # print(popt,np.sqrt(np.diag(pcov)))
        # print(popt2,np.sqrt(np.diag(pcov2)))
        # otor, = ax.plot(np.log10(vSFR),popt[0]+popt[1]*np.log10(vSFR), c="tab:orange",ls="-",lw=1.3,label=f"{popt[0]:.2f}+{popt[1]:.2f}"+r"$\log \mathrm{SFR}$")
        # otor2, = ax.plot(np.log10(vSFR),popt2[0]+np.log10(vSFR), c="k",ls="--",lw=1.3,label=f"{popt2[0]:.2f}+"+r"$\log \mathrm{SFR}$")
        otor, = ax.plot(np.log10(vSFR),39.58+np.log10(vSFR), c="k",ls="--",lw=1.3,label="L22")
        ax.fill_between(np.log10(vSFR),39.75+np.log10(vSFR),39.3+np.log10(vSFR),color='gray',alpha=.6)
    else:
        bLS_vec = np.vectorize(broken_Lx_SFR)(vSFR)
        bLS_vecL = np.vectorize(broken_Lx_SFR_Lehm)(vSFR)
        ax = plot_contour_log(gald.sfr_arr[nm],gald.LxH_arr[nm],clrHXB,kde_range=[-3,2.5,36,43])
        ax = plot_contour_log(gald.sfr_arr[nn],gald.LxZ_arr[nn],"purple",cmap="Blues_r",kde_range=[-3,2.5,36,43],ax=ax)
        ax = plot_hxb_sample(ax)
        ax.set_xticks([-2,-1.,0.,1.,2.])
        # ax.axis([np.log10(SFR.min()),np.log10(SFR.max()),np.log10(Lx.min()),np.log10(Lx.max())])
        ax.axis([-2,2.5,36.5,43.])
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{10**y:g}'))
        ax.set_xlabel(r"$\mathrm{SFR}\, [M_{\odot}\,\mathrm{yr}^{-1}]$",fontsize=14)
        ax.set_ylabel(r"$\log L_X^{0.5-8\,\mathrm{keV}}\,[\mathrm{erg}\,\,\mathrm{s}^{-1}]$",fontsize=14)
        mg12a, = ax.plot(np.log10(vSFR),np.log10(model)+np.log10(vSFR),c="k",label="M12a,SP")
        mg12b, = ax.plot(np.log10(vSFR),np.log10(bLS_vec),c="k",ls="--",label="G04b")
        Lehm21, = ax.plot(np.log10(vSFR),np.log10(model2)+np.log10(vSFR),c="tab:blue",label=r"L21($Z_{\odot}$)")
        # ax.plot(np.log10(vSFR),np.log10(bLS_vecL),c="k",ls=":",label="G03")

    plt.tight_layout(pad=0.15)
    plt.grid()
    plt.legend()
    plt.show()

def plot_Lx_Mstar_g(gald):
    """
    Plot Lx-SFR relation for HMXB emission
    based on Lx obtained from fits files and SFR from
    galaxy sample
    """
    vMstar = np.logspace(9,13,2000)
    n=(gald.LxG_arr>0.)

    ax = plot_contour_log(gald.mstar_arr[n],gald.LxG_arr[n],clrGAS,kde_range=[9.5,12.5,36.,42])
    ax.set_yticks([38,39,40,41,42])
    # ax.axis([np.log10(Mstar.min()),np.log10(Mstar.max()),np.log10(Lx.min()),np.log10(Lx.max())])
    ax.axis([9.5,12.5,36,42])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:g}'))
    ax.set_xlabel(r"$\log M_*$",fontsize=14)
    ax.set_ylabel(r"$\log L_X^{0.5-2\,\mathrm{keV}}\,[\mathrm{erg}\,\,\mathrm{s}^{-1}]$",fontsize=14)
    ax.plot(np.log10(vMstar),np.log10(vMstar)+29)
    plt.tight_layout(pad=0.15)
    plt.legend(fontsize=12)
    plt.grid()
    plt.show()

def plot_Lx_Mstar(gald):
    """
    Plot Lx-SFR relation for HMXB emission
    based on Lx obtained from fits files and SFR from
    galaxy sample
    """
    vMstar = np.logspace(-3,3,2000)

    ax = plot_contour_log(gald.mstar_arr/1e11,gald.LxL_arr,clrLXB,kde_range=[-2,2,38.,42])
    ax = plot_lxb_sample(ax)
    ax.set_xticks([-2.,-1.,0.,1.,2])
    ax.set_yticks([38,39,40,41,42])
    # ax.axis([np.log10(Mstar.min()),np.log10(Mstar.max()),np.log10(Lx.min()),np.log10(Lx.max())])
    ax.axis([-2,2,38,42])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y+11:g}'))
    ax.set_xlabel(r"$\log M_*$",fontsize=14)
    ax.set_ylabel(r"$\log L_X^{0.5-8\,\mathrm{keV}}\,[\mathrm{erg}\,\,\mathrm{s}^{-1}]$",fontsize=14)
    zg12, = ax.plot(np.log10(vMstar),np.log10(xm.Zhang12().model(bLum=True)[0])+36+np.log10(vMstar),c="k",label="1:1") # Lmin=37.69897
    plt.tight_layout(pad=0.15)
    plt.legend(fontsize=12)
    plt.grid()
    plt.show()   

def plot_XLF_gas(gald):

    def tlum(l, L):
        return L[L>l].sum()

    marr = np.logspace(9.8,11.5,21)
    tlg_r25k = np.array([1.4269354022112897e+41,1.0039840108386683e+42,1.7878517849704826e+42,3.116518948871417e+42,4.84911357443666e+42,6.413818281965785e+42,7.516638793996878e+42,8.735477515272737e+42,9.416509163240738e+42,1.0224234246182952e+43,1.0793046372157409e+43,1.1671940126580985e+43,1.2358704355310928e+43,1.3220666830386438e+43,1.4126688794278388e+43,1.5027951194447686e+43,1.548368262689186e+43,1.7242087297339733e+43,1.8773765226445334e+43,2.0434024990355228e+43,2.238370224938116e+43])
    tlo7_r25k = np.array([5.7362895587450835e+38,4.6201381740204765e+39,9.132717899060306e+39,1.454015021948668e+40,2.4888070047516516e+40,3.283590141892655e+40,3.7396484383572024e+40,4.1490005175986565e+40,4.413378741359214e+40,4.683811916139283e+40,4.916278573166924e+40,5.237208832809393e+40,5.523694026104125e+40,5.97553031981255e+40,6.826533629734831e+40,7.459001579727737e+40,7.647482621709691e+40,8.546430456000898e+40,9.175429718917703e+40,1.0335218957562038e+41,1.1050634612983223e+41])
    tlo8_r25k = np.array([2.874996522554189e+39,1.2093124800134239e+40,1.7456229939668682e+40,2.889943828327703e+40,4.468234209142748e+40,5.463455727389604e+40,6.284296698407571e+40,7.366433155301338e+40,7.7953047510086505e+40,8.245605649080219e+40,8.559518985987505e+40,9.041242249836638e+40,9.4490553469319e+40,9.829391476848904e+40,1.018146175437991e+41,1.0673432418703808e+41,1.0961407510504e+41,1.1793614362274064e+41,1.2534021660289266e+41,1.348883562473452e+41,1.4635437063754739e+41])
    tlg_r200 = np.array([1.4328238917505175e+41,1.2895541892840435e+42,2.0780889705078905e+42,3.54681824863261e+42,5.3079060781446297e+42,6.948352017943177e+42,8.106048234898858e+42,9.389552340853031e+42,1.0131288667331753e+43,1.1315572503493478e+43,1.2077906970012842e+43,1.3070597396652913e+43,1.3935498652998506e+43,1.490107152098411e+43,1.600383481097162e+43,1.7619884058288519e+43,1.842876109873588e+43,2.1589995300927086e+43,2.368398204998065e+43,2.6536767578583747e+43,2.902586901062004e+43])
    tlo7_r200 = np.array([5.85704267114131e+38,5.495314896151335e+39,1.00403115263927e+40,1.58751007323793e+40,2.6427529418435403e+40,3.480255361911101e+40,3.9838550766847914e+40,4.438064175975104e+40,4.750386415363734e+40,5.172103046006204e+40,5.525295345129127e+40,5.939164710459181e+40,6.340651061882103e+40,6.875515967128165e+40,7.948854521002429e+40,9.106813568052346e+40,9.570458571680615e+40,1.1121085487295058e+41,1.2704812850955721e+41,1.4844822710956722e+41,1.5955914154367953e+41])
    tlo8_r200 = np.array([2.8842794988615716e+39,1.4627378227735015e+40,2.0041579004394695e+40,3.2452811303597606e+40,4.858381291140705e+40,5.941377858540252e+40,6.80159115603479e+40,8.013817022443427e+40,8.5153160871934655e+40,9.382702552815443e+40,9.84216288669449e+40,1.0437045079887409e+41,1.1006950060808589e+41,1.1489781604605188e+41,1.2002218680758181e+41,1.3135991472840035e+41,1.3717590470005224e+41,1.5469516719028165e+41,1.6592557119990396e+41,1.8351245483710244e+41,1.9856984139792055e+41])


    lumarr = np.logspace(36,43,3000)
    norm = (48/.704*.93)**3
    m = (gald.mstar_arr < 4e10)
    # n = np.invert(m)
    xlf_g = xm.XRB().XLF(lumarr,gald.LxG_arr[m])/norm
    xlf_O7 = xm.XRB().XLF(lumarr,gald.LxO7_arr[m])/norm
    xlf_O8 = xm.XRB().XLF(lumarr,gald.LxO8_arr[m])/norm
    # print(xlf_g.max())
    # for mass in marr:
    #     n = (gald.mstar_arr < mass)
    #     print(tlum(lumarr[0],gald.LxO8_arr[n]))

    plt.figure(figsize=(4,4),dpi=500)
    plt.title(r"$M_* < 4\cdot 10^{10}\,,R_{200}$")
    plt.loglog()
    plt.plot(lumarr,xlf_g,label=".5-2 keV, gas")
    plt.plot(lumarr,xlf_O7,label="OVII (.569-.579 keV)")
    plt.plot(lumarr,xlf_O8,label="OVIII (.649-.659 keV")
    plt.xlabel(r"$L_X \,[\mathrm{erg}\,\mathrm{s}^{-1}]$",fontsize=13)
    plt.ylabel(r"$N(>L_X)\,/\,\mathrm{Mpc^{-3}}$")
    plt.tight_layout()
    plt.legend(fontsize=8)
    plt.show()

    plt.figure(figsize=(4,4),dpi=500)
    plt.loglog()
    plt.plot(marr,tlg_r200/norm,label=".5-2 keV, gas")
    plt.plot(marr,tlo7_r200/norm,label="OVII (.569-.579 keV)")
    plt.plot(marr,tlo8_r200/norm,label="OVIII (.649-.659 keV")
    plt.xlabel(r"$M_* \,[M_{\odot}]$",fontsize=13)
    plt.ylabel(r"$L_{\mathrm{tot}}(<M_*)\,/\,\mathrm{Mpc^{-3}}$")
    plt.tight_layout()
    plt.legend(fontsize=8)
    plt.show()


def plot_Lehmer16(gald):
    """
    sSFR relation from Lehmer+16
    """
    ssfr_line   = np.logspace(-14,-8,500)
    beta_L16    = 10**39.92
    alpha_L16   = 10**29.857
    beta_L16_z0 = 10**39.66
    alpha_L16_z0 = 10**29.62
    beta_L19    = 10**39.71
    alpha_L19   = 10**29.25
    Lehm        = lambda x,a,b: np.log10( a/x +b )
    brokenH = np.vectorize(broken_Lx_SFR)
    
    msk = (gald.sfr_arr > .2)
    Lx_arr = (gald.LxH_arr + gald.LxL_arr)/gald.sfr_arr
    ssfr_arr = gald.sfr_arr/gald.mstar_arr

    ax = plot_contour_log(ssfr_arr[msk],Lx_arr[msk],'tab:green',kde_range=[-13.5,-8.5,38.5,44])
    msk = (gald.sfr_arr < .2)
    ax = plot_contour_log(ssfr_arr[msk],Lx_arr[msk],'tab:olive',kde_range=[-13.5,-8.5,38.5,44],ax=ax)
    ax.axis([-13.5,-8.5,38.5,43])
    # ax.plot(np.log10(ssfr_line),Lehm(ssfr_line,alpha_L16,beta_L16),c='k',label="L16(global)")
    # ax.plot(np.log10(ssfr_line),Lehm(ssfr_line,alpha_L16_z0,beta_L16_z0),c='k',ls='--',label=r"L16($z=0$)")
    ax.plot(np.log10(ssfr_line),Lehm(ssfr_line,alpha_L19,beta_L19),c='k',ls='-.',label="Lehmer+19")
    ax.plot(np.log10(ssfr_line),np.log10(brokenH(ssfr_line*1e10)/(ssfr_line*1e10)),c="tab:blue",ls="--" )
    ax.plot(np.log10(ssfr_line),np.log10(xm.Zhang12().model(bLum=True)[0]*1e25/ssfr_line),c="tab:red",ls="--" )
    ax.plot(np.log10(ssfr_line),np.log10(brokenH(ssfr_line*1e10)/(ssfr_line*1e10)+xm.Zhang12().model(bLum=True)[0]*1e25/ssfr_line),c="tab:purple",ls="--",lw=2.5 )
    ax.plot(np.log10(ssfr_line),np.log10(brokenH(ssfr_line*1e11)/(ssfr_line*1e11)+xm.Zhang12().model(bLum=True)[0]*1e25/ssfr_line),c="tab:purple",ls="--",lw=1.5,alpha=.7 )
    ax.plot(np.log10(ssfr_line),np.log10(brokenH(ssfr_line*1e9)/(ssfr_line*1e9)+xm.Zhang12().model(bLum=True)[0]*1e25/ssfr_line),c="tab:purple",ls="--",lw=1.5,alpha=.7 )

    # Mineo14 sample
    LxSamp = np.array([39.52, 40.53, 39.75, 38.17, 40.48, 40.3, 41.02, 39.87, 40.79, 40.57, 40.92, 38.53, 40.32, 37.92, 38.31, 39.05, 40.88, 39.08, 40.39, 38.53, 39.77, 41.4, 41.63, 41.88, 41.42, 41.81, 41.39, 41.23, 40.92])
    sSFRsamp = np.log10([4.1/.7,11.7/4.7,0.44/.1,7.8e-2/2.8e-2,3.8/.91,6./4.,7.1/.98,31./1.7,4.6/2.6,5.4/3.1,16.8/2.1,.17/3.3e-2,1.8/.39,.09/7e-2,.38/5.9e-2,.18/9.1e-2,5.3/6.3,.29/.22,14.7/4.9,.29/.15,1.8/.37,289.9/10.3,139.4/7.5,139.6/7.,156.8/9.9,54.8/17.3,137.1/13.1,20.1/1.5,15.8/6.6])-10
    sfrSamp = np.array([4.1,11.6,.44,7.8e-2,3.8,6.,7.1,3.1,4.6,5.4,16.8,.17,1.8,.09,.38,.18,5.3,.29,14.7,.29,1.8,289.9,139.4,139.6,156.8,54.8,137.1,20.1,15.8])
    # Soria22 sample
    with open("Soria22.dat","r") as fp:
        Lines = fp.readlines()
    with open("Soria22L.dat","r") as fp:
        Lines2 = fp.readlines()
    massS22 = np.zeros(len(Lines))
    sfrS22 = np.zeros(len(Lines))
    sfrsig = np.zeros(len(Lines))
    lxS22 = np.zeros(len(Lines))
    lxsig = np.zeros(len(Lines))
    for i,line in enumerate(Lines):
        if "#" in line:
            continue
        l = line.strip().split()
        massS22[i] = float(l[0].split(u"\u00B1")[0])
        sfrS22[i] = float(l[1].split(u"\u00B1")[0])
        sfrsig[i] = float(l[1].split(u"\u00B1")[1])
    
    for i, line in enumerate(Lines2):
        if "#" in line:
            continue
        lxS22[i] = float(line.strip().split("[")[0].strip("<"))
        try:
            lxsig_str = line.strip().split("[")[1].strip("]").split("–")
            lxsig[i] = (float(lxsig_str[1])-float(lxsig_str[0]))/2.
        except IndexError:
            continue

    msk = (lxsig>0)
    massS22 = massS22[msk]
    sfrS22 = sfrS22[msk]
    ssfrS22 = sfrS22 / 10**massS22
    lxS22 = lxS22[msk]
    sfrsig = sfrsig[msk]
    lxsig = lxsig[msk]

    ax.plot(sSFRsamp,np.log10((10**LxSamp)/sfrSamp),lw=0.,marker='o',markerfacecolor='none',ms=6.,c='k',label="Mineo+14",alpha=.4)
    ax.plot(np.log10(ssfrS22),np.log10(lxS22/sfrS22)+39,lw=0.,marker='s',markerfacecolor='none',ms=6.,c='k',label="Soria+22",alpha=.4)
    # ax.plot(np.log10(np.sum(sfrS22)/np.sum(10**massS22)),np.log10(np.sum(lxS22*1e39)/np.sum(sfrS22)),marker="*",ms=12.,lw=0.,mfc="none")
    # ax.plot(sSFRsamp,np.log10((10**LxSampCorr)/sfrSamp),lw=0.,marker='o',markerfacecolor='none',ms=6.,c='k',label="M14")
    ax.set_xlabel(r"$\log(\mathrm{sSFR})$",fontsize=14)
    ax.set_ylabel(r"$\log(L_{\mathrm{X}}^{.5-8\,\mathrm{keV}} / \mathrm{SFR})$",fontsize=14)
    plt.tight_layout(pad=.15)
    plt.legend(fontsize=12)
    # plt.yticks([39,40,41,42])
    plt.grid()
    plt.show()


def plot_lxb_sample(ax):
    mstar_log = np.array([.62, .58, .51, .89, .74, .59, -.16, .37, .08, .9, .84, .48, .97, .92, -.16, .88, .82, 1.07, .34, .63, .53, 1.09, .45, -.29])-1.
    # lx_log = np.array([.5,.7,.4,1.3,.9,.5,.0,.4,.2,1.1,1.2,.8,1.3,1.3,-.2,.8,1.1,1.5,.7,1.,.9,1.3,.6,-.1])+39.
    lx_log = np.zeros_like(mstar_log)
    with open("Lehmer20.dat") as f:
        Lines = f.readlines()
        lx_list = []
        i = 0
        ngc_p = 1e7
        for line in Lines:
            # prepare line
            l = line.strip().split()
            # current galaxy id
            ngc_c = int(l[0])
            # new galaxy? Sum luminosities
            if ngc_c > ngc_p:
                # print(ngc_c,ngc_p)
                lx_log[i] = np.log10(np.sum(10**np.array(lx_list)))
                lx_list = []
                i += 1
            # only use flag 1 lxbs
            if l[-2] == "1":
                lx_list.append(float(l[-3]))
            # store current id for next iteration
            ngc_p = int(l[0])
        lx_log[i] = np.log10(np.sum(10**np.array(lx_list)))
        i += 1

    # print(lx_log)
    ax.plot(mstar_log,lx_log,lw=0.,marker="^",ms=6.,markerfacecolor='none',c='k',label="L20")
    return ax

def plot_hxb_sample(ax):
    sfrSamp = np.log10([4.1,11.6,.44,.21,7.8e-2,12,3.8,.52,10.5,6.,7.1,3.1,4.6,5.4,16.8,.17,1.8,.09,4.,3.7,.38,1.5,.18,5.3,.29,14.7,.29,1.8,17.6])
    lhxbSamp = np.array([39.34,40.40,39.69,39.72,37.93,40.38,40.47,39.37,39.86,39.79,40.78,39.79,40.60,40.23,40.45,38.33,40.05,37.98,39.68,39.93,38.52,39.47,38.90,40.53,39.10,40.25,38.23,39.68,41.23])
    ax.plot(sfrSamp,lhxbSamp,lw=0.,marker="D",ms=6.,mfc="none",c="k",label="M12a")
    return ax

def plot_gas_sample(ax):
    sfrSamp = np.log10([.44,7.8e-2,3.1,4.6,16.8,.17,5.4,14.7,.29])
    lhxbSamp = np.array([39.55,38.94,40.46,40.57,41.1,38.1,40.24,40.11,39.37])
    # print(10**lhxbSamp/10**(sfrSamp))
    # ax.plot(sfrSamp,lhxbSamp,lw=0.,marker="s",ms=6.,mfc="none",c="k",label="M12b")
    sfrSamp = np.log10([15.5,11.8,2.4])
    lhxbSamp = np.log10([5.8,6.3,4.3])+40
    ax.plot(sfrSamp,lhxbSamp,lw=0.,marker="p",ms=7.,mfc="none",c="k",label="B13")
    sfrSamp = np.log10([9.2,8.6,3.6,3.,8.1,4.6,3.,.2,4.2,.04])
    lhxbSamp = np.log10([5.15,3.7,.55,.53,3.2,.34,.78,.071,.6,.0065])+40
    ax.plot(sfrSamp,lhxbSamp,lw=0.,marker="*",ms=8.,mfc="none",c="k",label="S04")
    sfrSamp = np.log10([4.1,11.6,.44,7.8e-2,3.8,6.,7.1,3.1,4.6,5.4,16.8,.17,1.8,.09,.38,.18,5.3,.29,14.7,.29,1.8])
    lhxbSamp = np.array([39.59,39.62,38.46,38.03,39.65,39.96,40.33,39.06,39.52,40.17,40.41,37.48,39.12,37.82,38.36,37.57,39.73,37.77,39.77,38.06,39.21])
    lhxbadd = np.array([1.01e39,3.68e39,9.21e37,1.55e37,1.02e39,2.52e39,1.63e39,8.46e38,1.6e39,1.94e39,4.11e39,3.56e37,4.68e38,2.53e37,7.91e37,4.34e37,2.89e39,1.13e38,4.82e39,8.01e37,5.13e38])
    # print((10**lhxbSamp+lhxbadd)/10**(sfrSamp))
    ax.plot(sfrSamp,np.log10(10**lhxbSamp+lhxbadd),lw=0.,marker="s",ms=7.,mfc="none",c="r",label="M12b")

    return ax


def plot_Lx_OH(gald,bMet=True):
    from matplotlib import cm
    from helper import QuickStats as qs
    norm=mclr.LogNorm(vmin=1.e-2,vmax=200.,clip=True)
    curve_dict = np.load("Lehm21_samp.npy",allow_pickle=True).item()

    oh_sampl = curve_dict["OH"]
    Lehm21_normed = [40.21,40.25,40.25,40.22,40.16,40.06,39.94,39.8,39.64,39.49,39.34,39.21]
    Lehm21_sfr01 = [38.82,38.89,38.93,38.89,38.83,38.77,38.71,38.67,38.65,38.62,38.62,38.60]
    err = [[.66,.5,.38,.28,.2,.15,.11,.09,.1,.12,.13,.12],[.69,.53,.4,.29,.21,.15,.12,.1,.11,.13,.15,.16]]
    err01 = [[.5,.56,.59,.56,.52,.46,.42,.39,.37,.35,.35,.35],[1.53,1.56,1.59,1.64,1.65,1.58,1.37,1.,.75,.64,.58,.55]]
    errOH = np.arange(7,9.3,0.2)

    fig = plt.figure(figsize=(6,5),dpi=500)
    plt.plot(errOH,Lehm21_normed,c='k',label="L21")
    plt.fill_between(errOH,np.array(Lehm21_normed)-np.array(err[0]),np.array(Lehm21_normed)+np.array(err[1]),color='k',alpha=.2,interpolate=True)
    plt.plot(errOH,Lehm21_sfr01,c='k',ls="--",label="L21 (SFR=0.1)")
    plt.fill_between(errOH,np.array(Lehm21_sfr01)-np.array(err01[0]),np.array(Lehm21_sfr01)+np.array(err01[1]),color=cm.jet_r(norm(0.1)),alpha=.0,interpolate=True,hatch="////")
    # for key,val in curve_dict.items():
    #     print(key,val)
    #     if key == "OH":
    #         continue
    #     if "sfr" in key:
    #         plt.plot(oh_sampl,np.log10(val[0]),c=cm.jet_r(norm(float(key[3:]))),alpha=.9)
    #         plt.fill_between(oh_sampl,np.log10(val[1]),np.log10(val[2]),color=cm.jet_r(norm(float(key[3:]))),alpha=.2,interpolate=True)

    if bMet:
        ml = gald.LxZ_arr > 0.
        Lx_arr = gald.LxZ_arr[ml]
    else:
        ml = gald.LxH_arr > 0.
        Lx_arr = gald.LxH_arr[ml]
    sfr_arr = gald.sfr_arr[ml]
    oh_arr = gald.oh_arr[ml]
    mask1 = (sfr_arr > 10)
    mask2 = (sfr_arr <= 10) & (sfr_arr > 5.)
    mask3 = (sfr_arr <= 5.) & (sfr_arr > 1.)
    mask4 = (sfr_arr <= 1.) & (sfr_arr > .8)
    mask5 = (sfr_arr <= .8)

    relL = Lx_arr/sfr_arr
    # relL[relL<=0.] = 1.e39
    # print(len(relL[gald.oh_arr>7]))
    # plt.hist(np.clip(gald.oh_arr,a_min=7,a_max=10))
    # plt.show()
    for oh in np.arange(7.25,9.75,.25):
        if oh <= 7.25:
            ohmask = (oh_arr>=(oh-.25)) & (oh_arr<(oh+.25))
        else:
            ohmask = (oh_arr>=(oh-.25)) & (oh_arr<(oh+.25))
        med  = np.log10(np.mean(relL[ohmask])) 
        med1 = np.log10(np.mean(relL[ohmask&mask1]))
        med2 = np.log10(np.mean(relL[ohmask&mask2]))
        med3 = np.log10(np.mean(relL[ohmask&mask3]))
        med4 = np.log10(np.mean(relL[ohmask&mask4]))
        med5 = np.log10(np.mean(relL[ohmask&mask5]))
        
        if not np.isnan(med):
            mean, sig = np.mean(relL[ohmask]), np.std(relL[ohmask])
            print(mean,sig)
            per = qs.log_err(mean,sig)
        else:
            per = 0
        if not np.isnan(med1):
            mean, sig = np.mean(relL[mask1&ohmask]), np.std(relL[mask1&ohmask])
            per1 = qs.log_err(mean,sig)
        else:
            per1 = 0
        if not np.isnan(med2):
            mean, sig = np.mean(relL[mask2&ohmask]), np.std(relL[mask2&ohmask])
            per2 = qs.log_err(mean,sig)
        else:
            per2 = 0
        if not np.isnan(med3):
            mean, sig = np.mean(relL[mask3&ohmask]), np.std(relL[mask3&ohmask])
            per3 = qs.log_err(mean,sig)
        else:
            per3 = 0
        if not np.isnan(med4):
            mean, sig = np.mean(relL[mask4&ohmask]), np.std(relL[mask4&ohmask])
            per4 = qs.log_err(mean,sig)
        else:
            per4 = 0
        if not np.isnan(med5):
            mean, sig = np.mean(relL[mask5&ohmask]), np.std(relL[mask5&ohmask])
            per5 = qs.log_err(mean,sig)
        else:
            per5 = 0

        err  = plt.errorbar(oh      ,med  ,yerr=per ,xerr=.125,c="k",ls=":",ms=8.,capsize=3.,capthick=1.,fmt='x',mfc="none" )
        err1 = plt.errorbar(oh-.1   ,med1 ,yerr=per1,c=cm.jet_r(norm(np.mean(sfr_arr[mask1]))),ls=":",ms=8.,capsize=3.,capthick=1.,fmt='x',mfc="none" )
        err2 = plt.errorbar(oh-.05  ,med2 ,yerr=per2,c=cm.jet_r(norm(np.mean(sfr_arr[mask2]))),ls=":",ms=8.,capsize=3.,capthick=1.,fmt='s',mfc="none" )
        err3 = plt.errorbar(oh      ,med3 ,yerr=per3,c=cm.jet_r(norm(np.mean(sfr_arr[mask3]))),ls=":",ms=8.,capsize=3.,capthick=1.,fmt='o',mfc="none" )
        err4 = plt.errorbar(oh+.05  ,med4 ,yerr=per4,c=cm.jet_r(norm(np.mean(sfr_arr[mask4]))),ls=":",ms=8.,capsize=3.,capthick=1.,fmt='h',mfc="none" )
        err5 = plt.errorbar(oh+.1   ,med5 ,yerr=per5,c=cm.jet_r(norm(np.mean(sfr_arr[mask5]))),ls=":",ms=8.,capsize=3.,capthick=1.,fmt='*',mfc="none" )

        for e in [err1,err2,err3,err4,err5]:
            e[-1][0].set_linestyle(':')
        
    # plt.scatter(OH,np.log10(relL),c=SFR,alpha=.6,marker='o',s=15.,norm=norm,cmap='jet_r')
    fig = plt.gcf()
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap="jet_r"),shrink=.9)
    cb.set_label(label=r"$\mathrm{SFR}\,[M_{\odot}\, \mathrm{yr}^{-1}]$",fontsize=14)
    cb.ax.get_yaxis().set_major_formatter(lambda x,_: f"{x:.0f}" if x >= 1 else f"{x:.1f}" if x >= .1 else f"{x:.2f}" if x >= .01 else 0)
    plt.xlabel(r"$12 + \log[O/H]$",fontsize=14)
    plt.ylabel(r"$\log(L_{\mathrm{HXB}}^{.5-8\,\mathrm{keV}} / \mathrm{SFR})$",fontsize=14)
    plt.axis([7,9.5,38,41])
    plt.yticks([38,39,40,41])
    plt.tight_layout()
    plt.legend(loc="upper right")
    plt.show()

def mean_XLF(gald):

    sfr_h = gald.sfr_arr>.1
    xlf_h = gald.xlf_h[(gald.xlf_h[:,0]>0)&sfr_h].mean(axis=0)

    xlf_z = gald.xlf_z[(gald.xlf_z[:,0]>0)&sfr_h].mean(axis=0)

    xlf_l = gald.xlf_l[gald.xlf_l[:,0]>0].mean(axis=0)
    
    lumarr = gald.lumarr
    M12_xlf = xm.Mineo12S().model()
    Z12_xlf = xm.Zhang12().model()
    L21_xlf = xm.Lehmer21().model()
    single_phot_lum = ( calc_lum_cgs(.5,gald.area,gald.time,gald.dist),
                        calc_lum_cgs(3.,gald.area,gald.time,gald.dist),
                        calc_lum_cgs(8.,gald.area,gald.time,gald.dist) )

    fig = plt.figure(figsize=(5,5),dpi=500)
    plt.plot(lumarr,xlf_h,label="HXB")
    plt.plot(lumarr,xlf_z,label="HZB")
    plt.plot(lumarr,xlf_l,label="LXB")
    plt.axvline(single_phot_lum[1],ymin=1e-6,ymax=1e3,c="grey",ls="-",alpha=.8)
    plt.axvspan(xmin=single_phot_lum[0],xmax=single_phot_lum[2],ymin=1e-6,ymax=1e3,color="grey",alpha=.3)
    plt.plot(lumarr, M12_xlf,c='k',ls='--',lw=1.75,label=r"M12")
    plt.plot(lumarr, xm.Lehmer21(Lmin=35,logOH12=7.2).model(),c='k',ls=':',lw=1.05)#,label=r"L21 (7.2)")
    plt.plot(lumarr, L21_xlf,c='k',ls='-.',lw=1.75,label=r"L21 $(Z=Z_{\odot})$")
    plt.plot(lumarr, xm.Lehmer21(Lmin=35,logOH12=9.2).model(),c='k',ls=':',lw=1.05)#,label=r"L21 (9.2)")
    plt.plot(lumarr, Z12_xlf,c='k',ls='-',lw=1.75,label=r"Z12")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"$L_{XRB}^{0.5-8\,\mathrm{keV}}\,[\mathrm{erg\,s^{-1}}]$",fontsize=14)
    plt.ylabel(r"$N(>L)$",fontsize=14)
    plt.axis([1e36,5e40,1e-2,5e2])
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    plt.tight_layout(pad=.15)
    plt.legend(ncol=2)
    plt.show()

def plot_count_ratio(gald):
    phEh_list = gald.phEh_len
    phEl_list = gald.phEl_len
    phEg_list = gald.phEg_len
    
    bins = np.logspace(0,2,11)
    ce = calc_bin_centers(bins)
    hist1m = np.histogram(np.clip(1+(phEh_list+phEl_list).T[0,:]/phEg_list.T[0,:],a_min=bins[0],a_max=bins[-1]),bins,density=True)[0]*np.diff(bins)
    hist1s = np.histogram(np.clip(1+(phEh_list+phEl_list).T[1,:]/phEg_list.T[1,:],a_min=bins[0],a_max=bins[-1]),bins,density=True)[0]*np.diff(bins)
    hist1h = np.histogram(np.clip(1+(phEh_list+phEl_list).T[2,:]/phEg_list.T[2,:],a_min=bins[0],a_max=bins[-1]),bins,density=True)[0]*np.diff(bins)
    hist2m = np.histogram(np.clip(1+phEh_list.T[0,:]/phEg_list.T[0,:],a_min=bins[0],a_max=bins[-1]),bins,density=True)[0]*np.diff(bins)
    hist2s = np.histogram(np.clip(1+phEh_list.T[1,:]/phEg_list.T[1,:],a_min=bins[0],a_max=bins[-1]),bins,density=True)[0]*np.diff(bins)
    hist2h = np.histogram(np.clip(1+phEh_list.T[2,:]/phEg_list.T[2,:],a_min=bins[0],a_max=bins[-1]),bins,density=True)[0]*np.diff(bins)
    hist3m = np.histogram(np.clip(1+phEl_list.T[0,:]/phEg_list.T[0,:],a_min=bins[0],a_max=bins[-1]),bins,density=True)[0]*np.diff(bins)
    hist3s = np.histogram(np.clip(1+phEl_list.T[1,:]/phEg_list.T[1,:],a_min=bins[0],a_max=bins[-1]),bins,density=True)[0]*np.diff(bins)
    hist3h = np.histogram(np.clip(1+phEl_list.T[2,:]/phEg_list.T[2,:],a_min=bins[0],a_max=bins[-1]),bins,density=True)[0]*np.diff(bins)
    # hist4m = np.histogram(np.clip(1+phEh_list.T[0,:]/phEg_list.T[0,:],a_min=bins[0],a_max=bins[-1]),bins,density=True)[0]*np.diff(bins)
    # hist4s = np.histogram(np.clip(1+phEh_list.T[1,:]/phEg_list.T[1,:],a_min=bins[0],a_max=bins[-1]),bins,density=True)[0]*np.diff(bins)
    # hist4h = np.histogram(np.clip(1+phEh_list.T[2,:]/phEg_list.T[2,:],a_min=bins[0],a_max=bins[-1]),bins,density=True)[0]*np.diff(bins)
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(211)
    ax3 = fig2.add_subplot(212,sharex=ax2)
    ax2.axis([bins[0],bins[-1],0.005,1.15])
    ax3.axis([bins[0],bins[-1],-0.05,1.05])
    ax2.tick_params(labelbottom=False)
    # ax2.set_xscale("log")
    ax2.loglog()
    ax2.grid()
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    # ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax2.set_xticks([1.0,2,5.0,10.,20.,50.,100.])
    ax2.step(ce,hist1m,where="mid",lw=0.9,c="tab:orange",label="XRB",marker='D',ms=5.,fillstyle='none')
    ax2.step(ce,hist1s,where="mid",ls="--",lw=0.9,c="tab:orange",label="",marker='D',ms=5.,fillstyle='none')
    ax2.step(ce,hist1h,where="mid",ls="-.",lw=0.9,c="tab:orange",label="",marker='D',ms=5.,fillstyle='none')
    ax2.step(ce,hist2m,where="mid",lw=0.9,c=clrHXB,label="HXB",marker='D',ms=5.,fillstyle='none')
    ax2.step(ce,hist2s,where="mid",ls="--",lw=0.9,c=clrHXB,label="",marker='D',ms=5.,fillstyle='none')
    ax2.step(ce,hist2h,where="mid",ls="-.",lw=0.9,c=clrHXB,label="",marker='D',ms=5.,fillstyle='none')
    ax2.step(ce,hist3m,where="mid",lw=0.9,c=clrLXB,label="LXB",marker='D',ms=5.,fillstyle='none')
    ax2.step(ce,hist3h,where="mid",ls="--",lw=0.9,c=clrLXB,label="",marker='D',ms=5.,fillstyle='none')
    ax2.step(ce,hist3s,where="mid",ls="-.",lw=0.9,c=clrLXB,label="",marker='D',ms=5.,fillstyle='none')
    
    ax3.plot(ce,np.cumsum(hist1m),ls="-",lw=0.9,c="tab:orange",label="",marker='D',ms=5.,fillstyle='none')
    ax3.plot(ce,np.cumsum(hist1s),ls="--",lw=0.9,c="tab:orange",label="",marker='D',ms=5.,fillstyle='none')
    ax3.plot(ce,np.cumsum(hist1h),ls="-.",lw=0.9,c="tab:orange",label="",marker='D',ms=5.,fillstyle='none')
    ax3.plot(ce,np.cumsum(hist2m),ls="-",lw=0.9,c=clrHXB,label="",marker='D',ms=5.,fillstyle='none')
    ax3.plot(ce,np.cumsum(hist2s),ls="--",lw=0.9,c=clrHXB,label="",marker='D',ms=5.,fillstyle='none')
    ax3.plot(ce,np.cumsum(hist2h),ls="-.",lw=0.9,c=clrHXB,label="",marker='D',ms=5.,fillstyle='none')
    ax3.plot(ce,np.cumsum(hist3m),ls="-",lw=0.9,c=clrLXB,label="",marker='D',ms=5.,fillstyle='none')
    ax3.plot(ce,np.cumsum(hist3s),ls="--",lw=0.9,c=clrLXB,label="",marker='D',ms=5.,fillstyle='none')
    ax3.plot(ce,np.cumsum(hist3h),ls="-.",lw=0.9,c=clrLXB,label="",marker='D',ms=5.,fillstyle='none')
    from matplotlib.lines import Line2D
    handles, labels = ax2.get_legend_handles_labels()
    line = Line2D([0], [0], label='0.5-8 keV', color='k',ls="-",lw=.9)
    line2 = Line2D([0], [0], label='0.5-2 keV', color='k',ls="--",lw=.9)
    line3 = Line2D([0], [0], label='2-8 keV', color='k',ls="-.",lw=.9)
    handles.extend([line,line2,line3])
    ax2.legend(handles=handles,ncol=2)
    ax3.grid()
    ax3.set_xlabel(r"$r = 1+\frac{c_{\mathrm{XRB}}}{c_{\mathrm{GAS}}}$",fontsize=16)
    ax2.set_ylabel('fraction',fontsize=14)
    ax3.set_ylabel(r'fraction',fontsize=14)
    fig2.subplots_adjust(hspace=0.)
    plt.tight_layout(pad=0.1)
    plt.show()

def plot_Fragos13b(ax):
    """
    Just for comparison, arbitrary units
    """
    erg = []
    ff = []
    with open("/home/moon/vladutescu/Projects/XRBs/Fragos13b_spec.dat","r") as f:
        Lines = f.readlines()
    for i,line in enumerate(Lines):
        if i < 78:
            continue
        l = line.strip().split()
        # print(l)
        if l[0]=='1':
            erg.append(float(l[1]))
            ff.append(float(l[-1]))
    # print(erg,ff)
    ax.plot(erg,ff,lw=3.,ls="--",c="r")

def plot_Lehmer22(ax, smooth, lum_fac = 1):
    """
    Just for comparison
    """
    from astropy.convolution import convolve
    erg_f = np.linspace(.5,10,20001)
    cent = calc_bin_centers(erg_f)
    with open("/home/moon/vladutescu/Projects/XRBs/specOutFileApec2","r") as f:
        Lines = f.readlines()
    with open("/home/moon/vladutescu/Projects/XRBs/specOutFilePL","r") as f:
        Lines2 = f.readlines()
    for line,line2 in zip(Lines,Lines2):
        ff = line.strip().split()
        ff2 = line2.strip().split()
        
    lum = np.array(ff,dtype=np.float64)*lum_fac/np.diff(erg_f)*cent**2*42.93**2*4*np.pi
    lum2 = np.array(ff2,dtype=np.float64)*lum_fac/np.diff(erg_f)*cent**2*42.93**2
    # print(lum)
    ax.plot(cent,convolve(lum+lum2,smooth),lw=1.,ls="--",c="r")



def plot_average_spec(gald):

    from astropy.convolution import Gaussian1DKernel, convolve
    from PhoxUtil.Utils import QuickStats as qs

    conv_lum = gald.dist**2*4*np.pi/gald.area/gald.time*KEV_TO_ERG
    # print(gald.dist)
    bins = gald.bins
    ce = calc_bin_centers(bins)
    mce = (ce>=.5)&(ce<=10.)
    sfr_arr = gald.sfr_arr
    msfr = (sfr_arr >= 0.5)
    mstar_arr = gald.mstar_arr
    phEh_hist = gald.phEh_hist#*ce**2*conv_lum # convert to E*F_E*dlum**2
    phEl_hist = gald.phEl_hist#*ce**2*conv_lum
    phEg_hist = gald.phEg_hist#*ce**2*conv_lum
    phE_hist_xrb = phEh_hist + phEl_hist
    phE_hist_tot = phE_hist_xrb + phEg_hist
    nph_arr = np.sum(phE_hist_tot,axis=1)
    nph_arr = np.ones_like(nph_arr)
    # print(nph_arr)

    phE_tot_avg = np.mean((phE_hist_tot.T/nph_arr).T,axis=0)#*ce**2
    # sig_tot = np.std((phE_hist_tot.T/nph_arr).T,axis=0)#qs.log_err(phE_tot_avg, np.std((phE_hist_tot.T/nph_arr).T,axis=0) )
    phE_xrb_avg = np.mean((phE_hist_xrb.T/nph_arr).T,axis=0)#*ce**2
    phE_hxb_avg = np.mean((phEh_hist.T/nph_arr).T,axis=0)#*ce**2
    phE_lxb_avg = np.mean((phEl_hist.T/nph_arr).T,axis=0)#*ce**2
    phE_gas_avg = np.mean((phEg_hist.T/nph_arr).T,axis=0)#*ce**2

    smooth = Gaussian1DKernel(stddev=2.0)
    plt.figure(figsize=(5,5),dpi=500)
    plt.plot(ce[mce],convolve(phE_tot_avg[mce],smooth),c="k",label="comb.")
    plt.plot(ce[mce],convolve(phE_xrb_avg[mce],smooth),c="tab:orange",label="XRB",ls="--")
    plt.plot(ce[mce],convolve(phE_hxb_avg[mce],smooth),c=clrHXB,label="HXB",ls=":")
    plt.plot(ce[mce],convolve(phE_lxb_avg[mce],smooth),c=clrLXB,label="LXB",ls=":")
    plt.plot(ce[mce],convolve(phE_gas_avg[mce],smooth),c=clrGAS,label="GAS",ls=":")
    # plt.fill_between(ce[mce],convolve(10**(np.log10(phE_tot_avg[mce])+sig_tot[mce]),smooth),convolve(10**(np.log10(phE_tot_avg[mce])-sig_tot[mce]),smooth),color="k",alpha=.15,edgecolor="none")
    ax = plt.gca()
    # plot_Fragos13b(ax)
    # plot_Lehmer22(ax,smooth,lum_fac=conv_lum)
    # plt.plot(ce,phE_tot_avg)
    plt.xscale("log")
    plt.yscale("log")
    # plt.axis([.5,10.,1e-7,5e-4])
    # plt.axis([.5,2.,1e38,5e40])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:g}'))
    ax.set_xticks([.5,1,3,7,10])
    # ax.set_xticks([.5,1])
    # plt.ylabel(r"$\mathrm{keV}^2(\mathrm{photons}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}\,\mathrm{keV}^{-1}$",fontsize=13)
    plt.ylabel(r"$E\,L_E\,[\mathrm{erg\,s^{-1}}]$",fontsize=13)
    plt.xlabel(r"$E$ [keV]",fontsize=13)
    ax.grid()
    plt.legend(loc="upper right",fontsize=9)
    plt.show()
    # return
    nph_arr = gald.sfr_arr
    ssfr_arr = sfr_arr/mstar_arr

    sfr_edges = np.array([.01,.5,1.5,8.,200])
    sfr_edges = np.array([.4,1.,3,8.,200])
    sfr_mask1 = (sfr_arr>sfr_edges[0]) & (sfr_arr<=sfr_edges[1])
    sfr_mask2 = (sfr_arr>sfr_edges[1]) & (sfr_arr<=sfr_edges[2])
    sfr_mask3 = (sfr_arr>sfr_edges[2]) & (sfr_arr<=sfr_edges[3])
    sfr_mask4 = (sfr_arr>sfr_edges[3])
    sfr_masks = [sfr_mask1,sfr_mask2,sfr_mask3,sfr_mask4]

    mstar_edges = np.array([9e9,10**10.3,10**10.7,10**11.2,10**13])
    mstar_mask1 = (mstar_arr>mstar_edges[0]) & (mstar_arr<=mstar_edges[1])
    mstar_mask2 = (mstar_arr>mstar_edges[1]) & (mstar_arr<=mstar_edges[2])
    mstar_mask3 = (mstar_arr>mstar_edges[2]) & (mstar_arr<=mstar_edges[3])
    mstar_mask4 = (mstar_arr>mstar_edges[3])

    # ssfr_edges = np.array([-13,-11.5,-10.6,-10,-9])
    # ssfr_mask1 = (ssfr_arr>ssfr_edges[0]) & (ssfr_arr<=ssfr_edges[1])
    # ssfr_mask2 = (ssfr_arr>ssfr_edges[1]) & (ssfr_arr<=ssfr_edges[2])
    # ssfr_mask3 = (ssfr_arr>ssfr_edges[2]) & (ssfr_arr<=ssfr_edges[3])
    # ssfr_mask4 = (ssfr_arr>ssfr_edges[3])

    f = plt.figure(figsize=(10,10),dpi=500)
    gs1 = f.add_gridspec(nrows=2,ncols=1,left=0.08,right=0.47,bottom=0.55,top=0.98,hspace=0,height_ratios=[2,1])
    gs2 = f.add_gridspec(nrows=2,ncols=1,left=0.58,right=0.97,bottom=0.55,top=0.98,hspace=0,height_ratios=[2,1])
    gs3 = f.add_gridspec(nrows=2,ncols=1,left=0.08,right=0.47,bottom=0.05,top=0.48,hspace=0,height_ratios=[2,1])   
    gs4 = f.add_gridspec(nrows=2,ncols=1,left=0.58,right=0.97,bottom=0.05,top=0.48,hspace=0,height_ratios=[2,1])

    axs = []
    axs.append( f.add_subplot(gs1[0,0]) )
    axs.append( f.add_subplot(gs1[1,0]) )
    axs.append( f.add_subplot(gs2[0,0]) )
    axs.append( f.add_subplot(gs2[1,0]) )
    axs.append( f.add_subplot(gs3[0,0]) )
    axs.append( f.add_subplot(gs3[1,0]) )
    axs.append( f.add_subplot(gs4[0,0]) )
    axs.append( f.add_subplot(gs4[1,0]) )

    for ax in axs[::2]:
        ax.axis([.5,2.,4e38,8e40])
        ax.loglog()
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:g}'))
        ax.set_xticks([.5,1,3,7,10])
        # ax.set_xticks([.5,1])

        # ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:.0e}'))
        ax.tick_params(labelbottom=False,bottom=False)
        ax.grid()
        ax.set_ylabel(r"$E\,L_E\,/\mathrm{SFR}\,[\mathrm{erg\,s^{-1}\,(M_{\odot}\,yr^{-1})^{-1}}]$",fontsize=13)
    
    for ax in axs[1::2]:
        ax.axis([.5,2.,-0.02,1.02])
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:g}'))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:g}'))
        ax.set_xticks([.5,1,3,7,10])
        # ax.set_xticks([.5,1])

        ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
        ax.grid()
        ax.set_xlabel('$E_{ph}$ [keV]',fontsize=13)
        ax.set_ylabel(r'$\frac{c_i}{c_{tot}}$',fontsize=16)

    n = (mstar_mask1|mstar_mask2|mstar_mask3|mstar_mask4)
    phE_tot_avg_sfr = [ np.mean((phE_hist_tot[m&n].T/nph_arr[m&n]).T,axis=0) for m in sfr_masks ]
    # sig_tot_sfr = [ qs.log_err(phE_tot_avg_sfr[i],np.std((phE_hist_tot[m&n].T/nph_arr[m&n]).T,axis=0) ) for i,m in enumerate(sfr_masks) ]
    phE_xrb_avg_sfr = [ np.mean((phE_hist_xrb[m&n].T/nph_arr[m&n]).T,axis=0) for m in sfr_masks ]
    phE_hxb_avg_sfr = [ np.mean((phEh_hist[m&n].T/nph_arr[m&n]).T,axis=0) for m in sfr_masks ]
    phE_lxb_avg_sfr = [ np.mean((phEl_hist[m&n].T/nph_arr[m&n]).T,axis=0) for m in sfr_masks ]
    phE_gas_avg_sfr = [ np.mean((phEg_hist[m&n].T/nph_arr[m&n]).T,axis=0) for m in sfr_masks ]

    lsfr1 = f"{sfr_edges[0]:.2f}"+r"$<\mathrm{SFR}<$"+f"{sfr_edges[1]:.1f}, "+r"$N={:d}$".format(len(sfr_arr[sfr_mask1&n]))
    lsfr2 = f"{sfr_edges[1]:.1f}"+r"$<\mathrm{SFR}<$"+f"{sfr_edges[2]:.1f}, "+r"$N={:d}$".format(len(sfr_arr[sfr_mask2&n]))
    lsfr3 = f"{sfr_edges[2]:.1f}"+r"$<\mathrm{SFR}<$"+f"{sfr_edges[3]:.1f}, "+r"$N={:d}$".format(len(sfr_arr[sfr_mask3&n]))
    lsfr4 = f"{sfr_edges[3]:.1f}"+r"$<\mathrm{SFR}<$"+f"{sfr_edges[4]:.0f}, "+r"$N={:d}$".format(len(sfr_arr[sfr_mask4&n]))
    lsfr = [lsfr1,lsfr2,lsfr3,lsfr4]

    for i in range(4):
        axs[2*i].text(.1,.9,lsfr[i],fontsize=10,bbox=dict(facecolor='grey', alpha=0.5),transform=axs[2*i].transAxes)
        axs[2*i].step(ce[mce],convolve(phE_tot_avg_sfr[i][mce],smooth),lw=0.9,c="k",where="mid",label="comb.")
        # axs[2*i].fill_between(ce,10**(np.log10(phE_tot_avg_sfr[i])+sig_tot_sfr[i]),10**(np.log10(phE_tot_avg_sfr[i])-sig_tot_sfr[i]),color="k",alpha=.15,edgecolor="none")
        axs[2*i].step(ce[mce],convolve(phE_xrb_avg_sfr[i][mce],smooth),lw=0.9,c="tab:orange",where="mid",label="XRB",ls="--")
        axs[2*i].step(ce[mce],convolve(phE_hxb_avg_sfr[i][mce],smooth),lw=0.9,c=clrHXB,where="mid",label="HXB",ls=":")
        axs[2*i].step(ce[mce],convolve(phE_lxb_avg_sfr[i][mce],smooth),lw=0.9,c=clrLXB,where="mid",label="LXB",ls=":")
        axs[2*i].step(ce[mce],convolve(phE_gas_avg_sfr[i][mce],smooth),lw=0.9,c=clrGAS,where="mid",label="GAS",ls=":")
        axs[2*i].legend(loc="upper right",fontsize=9)
        axs[2*i+1].plot(ce[mce],convolve(phE_xrb_avg_sfr[i][mce],smooth)/convolve(phE_tot_avg_sfr[i][mce],smooth),lw=0.9,c="tab:orange",ls="--")
        axs[2*i+1].plot(ce[mce],convolve(phE_hxb_avg_sfr[i][mce],smooth)/convolve(phE_tot_avg_sfr[i][mce],smooth),lw=0.9,c=clrHXB,ls=":")
        axs[2*i+1].plot(ce[mce],convolve(phE_lxb_avg_sfr[i][mce],smooth)/convolve(phE_tot_avg_sfr[i][mce],smooth),lw=0.9,c=clrLXB,ls=":")
        axs[2*i+1].plot(ce[mce],convolve(phE_gas_avg_sfr[i][mce],smooth)/convolve(phE_tot_avg_sfr[i][mce],smooth),lw=0.9,c=clrGAS,ls=":")

    plt.show()

def plot_SFR_Mstar(gd: dict):
    """
    plot SFR Mstar relation of dict galaxies
    """
    mstar_arr = np.linspace(9.,13.,100)
    sfr_arr3 = 10**(-11)*10**mstar_arr
    sfr_arr1 = 10**(-9)*10**mstar_arr
    sfr_arr2 = 10**(-10)*10**mstar_arr
    sfr_arr4 = 10**(-12)*10**mstar_arr

    ssfr_samp = gd.sfr_arr/gd.mstar_arr
    m = (ssfr_samp > 10**(-11.6))
    n = (ssfr_samp <= 10**(-11.6)) # Weinmann cut
    fig = plt.figure(figsize=(4.5,4.5),dpi=500)
    plt.yscale("log")
    plt.axis([9.5,12.5,.001,200])
    plt.ylabel(r"SFR $[M_{\odot}\,\mathrm{yr}^{-1}]$")
    plt.xlabel(r"$\log(M_*\,[M_{\odot}])$")
    plt.plot(mstar_arr,np.array([sfr_arr1,sfr_arr2,sfr_arr3,sfr_arr4]).T,ls="--",lw=.9,c="grey")
    msk = (gd.bVal_arr>-4.35)
    plt.plot(np.log10(gd.mstar_arr[msk]),gd.sfr_arr[msk],lw=0.,marker=".",ms=4.5,c="tab:blue",alpha=.5,mec="none")
    msk = (gd.bVal_arr<-4.35)&(gd.bVal_arr>-4.7)
    plt.plot(np.log10(gd.mstar_arr[msk]),gd.sfr_arr[msk],lw=0.,marker=".",ms=4.5,c="tab:green",alpha=.5,mec="none")
    msk = (gd.bVal_arr<-4.7)
    plt.plot(np.log10(gd.mstar_arr[msk]),gd.sfr_arr[msk],lw=0.,marker=".",ms=4.5,c="tab:red",alpha=.5,mec="none")
    # plt.scatter(np.log10(gd.mstar_arr),gd.sfr_arr,marker=".",s=13,c=gd.bVal_arr,cmap="jet_r",alpha=.7,vmin=-6.5,vmax=-3.,edgecolor="none")
    plt.show()

def plot_Mgas_Mstar(gd: dict):
    """
    plot SFR Mstar relation of dict galaxies
    """
    fig = plt.figure(figsize=(4.5,4.5),dpi=500)
    # plt.yscale("log")
    plt.axis([9.,12.5,9,12.5])
    plt.ylabel(r"$\log(M_{gas}(<R_{2500}) \,[M_{\odot}])$")
    plt.xlabel(r"$\log(M_*\,[M_{\odot}])$")
    plt.plot(np.linspace(9,13,100),np.linspace(9,13,100),ls="--",lw=.9,c="grey")
    plt.scatter(np.log10(gd.mstar_arr),np.log10(gd.mgas_arr*1e10/.704),marker=".",s=13,c=gd.bVal_arr,cmap="jet_r",alpha=.7,vmin=-6.5,vmax=-3,edgecolor="none")
    plt.colorbar()
    plt.show()


class GalaxyData:
    """
    Load galaxy data from file path
    """
    def __init__(self, fp: str, start: int = 0, stop: int = 2000, babs: str = "fits/", gdp: str = None) -> None:
        phxf        = phE_file(fp+"phoxdir_136/GASphotonsE_136.0.dat")
        dist        = phxf.Da*(1+phxf.zz_obs)**2 # in cm
        if gdp is not None:
            gal_dict    = Galaxy.gal_dict_from_npy(gdp)
        else:
            gal_dict    = Galaxy.gal_dict_from_npy("/home/moon/vladutescu/Projects/XRBs/gal_data.npy")
        
        if stop > len(gal_dict):
            stop = len(gal_dict)
        nums = stop - start

        sfr_arr     = np.zeros(nums)
        mstar_arr   = np.zeros(nums)
        mgas_arr    = np.zeros(nums)
        LxZ_arr     = np.zeros(nums)
        LxL_arr     = np.zeros(nums)
        LxH_arr     = np.zeros(nums)
        LxG_arr     = np.zeros(nums)
        LxO7_arr    = np.zeros(nums)
        LxO8_arr    = np.zeros(nums)
        oh_arr      = np.zeros(nums)
        bVal_arr    = np.zeros(nums)
        oh_arr_g    = np.zeros(nums)

        # bins        = np.logspace(np.log10(phxf.e_min),np.log10(phxf.e_max),phxf.n_chan)
        bins        = np.linspace(phxf.e_min,phxf.e_max,phxf.n_chan)
        phEh_hist   = np.zeros((nums,len(bins)-1))
        phEz_hist,phEl_hist,phEg_hist = np.zeros_like(phEh_hist),np.zeros_like(phEh_hist),np.zeros_like(phEh_hist)
        phEh_len    = np.zeros((nums,3))
        phEz_len,phEl_len,phEg_len = np.zeros_like(phEh_len),np.zeros_like(phEh_len),np.zeros_like(phEh_len)

        lumarr      = xm.Mineo12S().lumarr
        xlf_h       = np.zeros((nums,len(lumarr)))
        xlf_l,xlf_z = np.zeros_like(xlf_h), np.zeros_like(xlf_h)

        ffs         = fp+babs
        # if babs:
        #     ffsa    = fp+"fits_abs/"
        # else:
        #     ffsa    = ffs

        pbar = tqdm(enumerate(gal_dict.items(), start=-start),total=stop,desc="[GalaxyData]: Loading data ...",bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',initial=start)

        for num, (key, gal) in enumerate(gal_dict.items(), start=-start):
            if num < 0:
                continue
                
            if num == stop-start:
                break
            # num = num-start
            fpZ = fp+f"fits/gal{key}HZB.fits"
            fpH = fp+f"fits/gal{key}HXB.fits"
            fpL = fp+f"fits/gal{key}LXB.fits"
            fpG = ffs+f"gal{key}GAS.fits"
            gal.snapbase = "/HydroSims/Magneticum/Box4/uhr_test/snapdir_136/snap_136"
            gal.groupbase = "/HydroSims/Magneticum/Box4/uhr_test/groups_136/sub_136"

            mstar_arr[num] = gal.Mstar
            # mgas_arr[num] = np.sum(gal.get_gas()["MASS"])
            # Weinmann et al. 2010 suggest setting the sSFR value of
            # simulated galaxies with SFR = 0 uniformly in range
            # -12.4 <= sSFR <= -11.6

            # Rhea suggested calculating a lower limit for star-formation
            # based on 1 star particle formed in the last 100 Myr
            if gal.SFR > 0.:
                sfr_arr[num] = gal.SFR
            else:
                sfr_arr[num] = 10**np.random.uniform(-12.4,-11.6)*gal.Mstar
                # sfr_arr[num] = 0.023 #np.mean(gal.get_stars()["iM  "])*1e2
            
            oh_arr[num] = gal.logOH12_s
            bVal_arr[num] = gal.bVal
            # oh_arr_g[num] = gal.logOH_12_g

            phEz = yield_photon_energy(fpZ)
            phEh = yield_photon_energy(fpH)
            phEg = yield_photon_energy(fpG)
            phEl = yield_photon_energy(fpL)

            xlf_h[num] = xm.XRB().XLF(lumarr,get_num_XRB(phEh,phxf.time,phxf.area,dist)[1])/sfr_arr[num]
            xlf_l[num] = xm.XRB().XLF(lumarr,get_num_XRB(phEl,phxf.time,phxf.area,dist)[1])/gal.Mstar*1e11
            xlf_z[num] = xm.XRB().XLF(lumarr,get_num_XRB(phEz,phxf.time,phxf.area,dist)[1])/sfr_arr[num]

            phEh_len[num] = [ len(phEh[(phEh>=.5)&(phEh<=8.)]), len(phEh[(phEh>=.5)&(phEh<=2.)]), len(phEh[(phEh>=2.)&(phEh<=8.)]) ]
            phEz_len[num] = [ len(phEz[(phEz>=.5)&(phEz<=8.)]), len(phEz[(phEz>=.5)&(phEz<=2.)]), len(phEz[(phEz>=2.)&(phEz<=8.)]) ]
            phEl_len[num] = [ len(phEl[(phEl>=.5)&(phEl<=8.)]), len(phEl[(phEl>=.5)&(phEl<=2.)]), len(phEl[(phEl>=2.)&(phEl<=8.)]) ]
            phEg_len[num] = [ len(phEg[(phEg>=.5)&(phEg<=8.)]), len(phEg[(phEg>=.5)&(phEg<=2.)]), len(phEg[(phEg>=2.)&(phEg<=8.)]) ]

            phEh_hist[num] = calc_flux_per_bin(bins,np.histogram(phEh,bins=bins)[0],phxf.area,phxf.time)
            phEz_hist[num] = calc_flux_per_bin(bins,np.histogram(phEz,bins=bins)[0],phxf.area,phxf.time)
            phEl_hist[num] = calc_flux_per_bin(bins,np.histogram(phEl,bins=bins)[0],phxf.area,phxf.time)
            phEg_hist[num] = calc_flux_per_bin(bins,np.histogram(phEg,bins=bins)[0],phxf.area,phxf.time)
            
            LxZ_arr[num]    = calc_lum_cgs(phEz,phxf.area,phxf.time,dist)
            LxH_arr[num]    = calc_lum_cgs(phEh,phxf.area,phxf.time,dist)
            LxG_arr[num]    = calc_lum_cgs(phEg,phxf.area,phxf.time,dist,Emax=2.)
            LxL_arr[num]    = calc_lum_cgs(phEl,phxf.area,phxf.time,dist)
            LxO7_arr[num]   = calc_lum_cgs(phEg,phxf.area,phxf.time,dist,Emin=.569,Emax=0.579)
            LxO8_arr[num]   = calc_lum_cgs(phEg,phxf.area,phxf.time,dist,Emin=.649,Emax=0.659)

            pbar.update()

        pbar.close()

        self.sfr_arr        = sfr_arr[:num]
        self.mstar_arr      = mstar_arr[:num]
        self.mgas_arr       = mgas_arr[:num]
        self.oh_arr         = oh_arr[:num]
        self.bVal_arr       = bVal_arr[:num]
        self.phEh_len       = phEh_len[:num]
        self.phEz_len       = phEz_len[:num]
        self.phEl_len       = phEl_len[:num]
        self.phEg_len       = phEg_len[:num]
        self.phEh_hist      = phEh_hist[:num]
        self.phEz_hist      = phEz_hist[:num]
        self.phEl_hist      = phEl_hist[:num]
        self.phEg_hist      = phEg_hist[:num]
        self.bins           = bins
        self.LxZ_arr        = LxZ_arr[:num]
        self.LxH_arr        = LxH_arr[:num]
        self.LxG_arr        = LxG_arr[:num]
        self.LxL_arr        = LxL_arr[:num]
        self.LxO7_arr       = LxO7_arr[:num]
        self.LxO8_arr       = LxO8_arr[:num]
        self.xlf_h          = xlf_h[:num]
        self.xlf_l          = xlf_l[:num]
        self.xlf_z          = xlf_z[:num]
        self.lumarr         = lumarr
        self.area           = phxf.area
        self.time           = phxf.time
        self.dist           = dist

if __name__ == '__main__':
    pass
    
    
