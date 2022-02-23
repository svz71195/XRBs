import numpy as np
import helper
import xrb_main as xm
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter



def IMF_m(m,a=.241367,b=.241367,c=.497056):

    # a,b,c = (.241367,.241367,.497056)
    # a=b=c=1/3.6631098624

    if .1 <= m <= .3:
        res = a*( 1-100**(-.3) )/.3 + b*( 1-.3**(.2) )/.2 + c*( .3**(.8)-m**(.8) )/.8
    elif .3 < m <= 1.:
        res = a*( 1-100**(-.3) )/.3 + b*( 1-m**(.2) )/.2
    elif 1. < m <= 100.:
        res = a*( m**(-.3)-100**(-.3) )/.3
    else:
        res = 0

    return res


def IMF_N(m,a=.241367,b=.241367,c=.497056):
    """
    returns number of stars with mass m
    """
    # a,b,c = (.241367,.241367,.497056)
    # a=b=c=1/3.6631098624

    if .1 <= m <= .3:
        res = c*( m**(-1.2) )
    elif .3 < m <= 1.:
        res = b*( m**(-1.8) )
    elif 1. < m <= 100.:
        # res = a*( m**(-1.3)-100**(-1.3) )/1.3
        res = a*( m**(-2.3) )
    else:
        res = 0

    return res

def dm_dt_MM(m,t):
    """
    't': time in [Gyr],
    'm': mass in [Msun]
    """

    if(m <= 1.3):
        return -m / t / 0.6545
    if(m > 1.3 and m <= 3):
        return -m / t / 3.7
    if(m > 3 and m <= 7):
        return -m / t / 2.51
    if(m > 7 and m <= 15):
        return -m / t / 1.78
    if(m > 15 and m <= 53.054):
        return -m / t / 0.86
    if(m > 53.054):
        return -0.54054054054 * m / (t - 0.003)

def dm_dt_PM(m,t):
    if(t > 0.039765318659064693):
        return -m / t * (1.338 - 0.1116 * (9 + np.log10(t)))
    else:
        return -0.45045045045 * m / (t - 0.003)


def lifetime_MM(mass1: float):
    """
    Stellar lifetime function of Maeder & Maynet (1989) extrapolated by Chiappini, Matteucci & Gratton (1997).
    See also Tornatore (2007) for use case in stellar chemical enrichment.
    -----
    tau: stellar age (one star) [Gyr]
    mass: stellar mass (one star) [Msun]
    """
    tau: float = 0

    if mass1 <= 1.3:
        tau = mass1**(-.6545) * 10.
    elif 1.3 < mass1 <= 3.:
        tau = mass1**(-3.7) * 10**1.351
    elif 3. < mass1 <= 7.:
        tau = mass1**(-2.51) * 10**.77
    elif 7. < mass1 <= 15.:
        tau = mass1**(-1.78) * 10**.17
    elif 15. < mass1 <= 53.054:
        # discontinuity to previous step
        tau = mass1**(-.86) * 10**(-.94)
    else:
        # mass > 53
        tau = 1.2*mass1**(-1.85)+.003

    return tau

def lifetime_PM(mass):
    if(mass <= 6.6):
        return pow(10, ((1.338 - np.sqrt(1.790 - 0.2232 * (7.764 - np.log10(mass)))) / 0.1116) - 9)
    else:
        return 1.2 * pow(mass, -1.85) + 0.003


def inverse_lifetime_MM(tau: float):
    """
    Inverse stellar lifetime function of Maeder & Maynet (1989) extrapolated by Chiappini, Matteucci & Gratton (1997).
    See also Tornatore (2007) for use case in stellar chemical enrichment.
    -----
    tau: stellar age (one star)
    mass: stellar mass (one star)
    """
    if(tau >= 8.4221714076):
        return pow(10, (1 - np.log10(tau)) / 0.6545)
    if(tau < 8.4221714076 and tau >= 0.38428316376):
        return pow(10, (1.35 - np.log10(tau)) / 3.7)
    if(tau < 0.38428316376 and tau >= 0.044545508363):
        return pow(10, (0.77 - np.log10(tau)) / 2.51)
    if(tau < 0.044545508363 and tau >= 0.01192772338):
        return pow(10, (0.17 - np.log10(tau)) / 1.78)
    if(tau < 0.01192772338 and tau >= 0.0037734864318):
        return pow(10, -(0.94 + np.log10(tau)) / 0.86)
    if(tau < 0.0037734864318 and tau > .003001):
        return pow((tau - 0.003) / 1.2, -0.54054054)
    if(tau <= 0.003001):
        return 100


def inverse_lifetime_PM(tau):
    if(tau > 0.039765318659064693):
        return pow(10, 7.764 - (1.79 - pow(1.338 - 0.1116 * (9 + np.log10(tau)), 2)) / 0.2232)
    elif(tau > .003001):
        return pow((tau - 0.003) / 1.2, -1.0 / 1.85)
    else:
        return 100.

def LT_SNrate(time, bPM: bool = True):
    """
    time: in [Gyr]
    bPM: boolean controlling the lifetime-function type
    returns SNII-rate in [#/Gyr] since IMF_N ~ dN/dm with dm/dt ->>> dN/dt
    for a SSP of 1 Msun
    """
    if bPM:
        mass_SNR = inverse_lifetime_PM(time)
        return (IMF_N(mass_SNR) * (-dm_dt_PM(mass_SNR, time)))
    else:
        mass_SNR = inverse_lifetime_MM(time)
        return (IMF_N(mass_SNR) * (-dm_dt_MM(mass_SNR, time)))

def Shty07(Ntime: np.ndarray, mas: np.ndarray):

    part1 = ((Ntime > .004)&(Ntime < .009))*.229*mas*1.e-5
    part2 = ((Ntime > .009)&(Ntime < .02))*5*.229*mas*1.e-5
    part3 = ((Ntime > .02)&(Ntime < .05))*6.5*.229*mas*1.e-5
    part4 = ((Ntime > .05)&(Ntime < .1))*1.5*.229*mas*1.e-5

    full = part1+part2+part3+part4

    return full

if __name__ == '__main__':
    time_arr = np.logspace(np.log10(4e-3), -1, 10000)
    mass_arr = np.vectorize(inverse_lifetime_MM)(time_arr[::-1])
    mass_diff = np.diff(mass_arr)
    print(time_arr, mass_arr, inverse_lifetime_MM(.1))
    imf_fac = helper.Integrate.Riemann_log(np.vectorize(IMF_N),8,100,10000)
    print(IMF_m(.1), imf_fac, inverse_lifetime_PM(.003))

    from galaxies import Galaxy
    import g3read as g3
    groupbase = "/HydroSims/Magneticum/Box4/uhr_test/groups_136/sub_136"
    snapbase = "/HydroSims/Magneticum/Box4/uhr_test/snapdir_136/snap_136"
    phbase = "/ptmp2/vladutescu/paper21/seed/919/fits/"


    head = g3.GadgetFile(snapbase+".0").header
    h = head.HubbleParam
    zz = head.redshift

    gal_dict = Galaxy.gal_dict_from_npy("gal_data.npy")
    for key in gal_dict.keys():
        # if int(key) == 13633:
        if int(key) == 12623:
            x = gal_dict[key]
            x.snapbase = snapbase
            x.groupbase = groupbase
            stars = x.get_stars()
            pos = x.pos_to_phys(stars["POS "] - x.center)
            age = stars["AGE "]
            rad = g3.to_spherical(pos,[0,0,0]).T[0]
            R25K = x.pos_to_phys(x.R25K)
            # print(R25K,x.center)
            # print(f"{x.Mstar:.2e}")
            mask = (rad < R25K*1) & (x.age_part(age)<100) & (x.age_part(age)>4) 
            xpos = pos[:,0][mask]
            ypos = pos[:,1][mask]
            pos = pos[mask]
            age = x.age_part(age[mask])
            mass = x.mass_to_phys(stars["MASS"])[mask]
            iM = x.mass_to_phys(stars["iM  "])[mask]
            asfr = np.sum(mass)/1e8
            print("aSFR ",asfr)
            

    Npop = len(age)
    Mtot = mass 
    ran_stars = age/1e3 #np.random.uniform(.004,.1,Npop)
    LT = np.vectorize(LT_SNrate)
    f_x = xm.Mineo12S(Lmin=35).model()[0] / imf_fac * 1.e-5
    print(f_x)
    eff = 1.e-4*mass*f_x
    
    SFR = asfr #np.sum(Mtot) / 1.e8

    NHXB_LT = np.sum(LT(ran_stars[age<30],False)*1.e-5*Mtot[age<30]*.18*7)
    NHXB_th = 135*SFR

    NHXB_LT2 = np.sum(LT(ran_stars[age<30])*eff[age<30])
    NHXB_th2 = xm.Mineo12S().model()[0]*SFR

    print(f"SFR = {SFR:.1f}\nNHXB_LT = {NHXB_LT:.1f}\nNHXB_th = {NHXB_th:.1f}")
    print(f"SFR = {SFR:.1f}\nNHXB_LT2 = {NHXB_LT2:.1f}\nNHXB_th2 = {NHXB_th2:.1f}")
    print(LT(0.02202))

    plt.plot(np.log10(time_arr*1e3), np.log10(LT(time_arr,False)),label='MM89 lifetime')
    plt.plot(np.log10(time_arr*1e3), np.log10(LT(time_arr)),label='PM93 lifetime')
    plt.xlabel(r"$\log(\mathrm{\tau_M\,[Myr]})$",fontsize=13)
    plt.ylabel(r"$\log(R_{\mathrm{SNII}}\,\, [\mathrm{Gyr^{-1}}])$",fontsize=13)
    plt.vlines(np.log10(4),-1,0.,color='k',ls="--",lw=2.,label=r"age cut")
    plt.vlines(np.log10(30),-1,0.,color='k',ls="--",lw=2.)
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

    N=2000
    nhxb = np.zeros(N)
    nhxb2 = np.zeros(N)
    nhxb3 = np.zeros(N)
    nhxb4 = np.zeros(N)
    sfr = np.zeros(N)
    for i,key in enumerate(gal_dict.keys()):
        if i >= 2000:
            break
        x = gal_dict[key]
        x.snapbase = snapbase
        x.groupbase = groupbase
        if x.SFR < 0.005:
            continue
        stars = x.get_stars()
        pos = x.pos_to_phys(stars["POS "] - x.center)
        age = stars["AGE "]
        rad = g3.to_spherical(pos,[0,0,0]).T[0]
        R25K = x.pos_to_phys(x.R25K)
        mask = (rad < R25K*1) & (x.age_part(age)<100) & (x.age_part(age)>4) 
        age = x.age_part(age[mask])/1e3
        if len(age)==0:
            continue
        mass = x.mass_to_phys(stars["MASS"])[mask]
        iM = x.mass_to_phys(stars["iM  "])[mask]
        asfr = np.sum(mass)/1e8
        
        mass = mass[age<.03]
        age = age[age<.03]
        
        if len(age)==0:
            continue
        # nhxb[i] = np.sum(LT(age)*1.e-5*mass*f_x*8.21*binary_frac)/asfr
        nhxb[i] = np.sum(LT(age,False)*1.e-4*mass*f_x)/asfr
        # nhxb4[i] = np.sum(LT2(age)*1.e-5*mass*f_x*8.21*binary_frac)/asfr
        nhxb4[i] = np.sum(LT(age)*1.e-4*mass*f_x)/asfr
        if (nhxb4[i] < 40) or (nhxb4[i] > 3*187):
            print(i, nhxb4[i], asfr, age*1e3)
        sfr[i] = asfr

    dfg = (nhxb > 0)
    nhxba = nhxb[dfg]
    sfr2 = sfr[dfg]
    # nhxb2 = nhxb2[nhxb2>0]
    nhxb4a = nhxb4[dfg]

    xlf_list = []
    sampl_list = []
    
    for nh,sf in zip(nhxb4a,sfr2):
        sampl = xm.Mineo12S(Lmin=35).sample(int(nh*sf))
        sampl_list.append(sampl)
        xlf = xm.XRB(Lmin=35).XLF(sampl)/sf
        xlf_list.append(xlf)
        plt.plot(xm.XRB(Lmin=35).lumarr[xlf>0], xlf[xlf>0],c='m',lw=.9,alpha=.2)

    plt.plot(xm.Mineo12S(Lmin=35).lumarr, xm.Mineo12S(Lmin=35).model(),c='k',ls='--',lw=1.5)        
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"$L_X\,[\mathrm{erg\,s^{-1}}]$",fontsize=13)
    plt.ylabel(r"$N(>L)\times\mathrm{SFR^{-1}}$",fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

    medi, medu, medl = np.zeros_like(xm.Mineo12S(Lmin=35).lumarr), np.zeros_like(xm.Mineo12S(Lmin=35).lumarr), np.zeros_like(xm.Mineo12S(Lmin=35).lumarr)
    meni, stdi = np.zeros_like(xm.Mineo12S(Lmin=35).lumarr), np.zeros_like(xm.Mineo12S(Lmin=35).lumarr)
    for i, lum in enumerate(xm.Mineo12S(Lmin=35).lumarr):
        spread = np.array([xx[i] for xx in xlf_list])# if xx[i]>0]
        meni[i] = np.mean(spread)
        stdi[i] = np.std(spread)
        if len(spread[spread>0]) == 0:
            continue
        medi[i] = np.median(spread[spread>0])
        medu[i] = np.percentile(spread[spread>0],84)
        medl[i] = np.percentile(spread[spread>0],16)

    scale = []
    for samp in sampl_list:
        scale.append(np.sum(samp))
    
    plt.plot(sfr2,scale,"bo",markeredgecolor="none",alpha=.3)
    plt.plot(np.linspace(0,200,1000),xm.Mineo12S().model(bLum=True)[0]*1.e38*np.linspace(0,200,1000),c='k',lw=2.)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"$\mathrm{SFR}\,[\mathrm{M_{\odot}\,\mathrm{yr}^{-1}}]$",fontsize=13)
    plt.ylabel(r"$L_{\mathrm{tot}}\,[\mathrm{erg\,s^{-1}}]$",fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
        

    
    plt.fill_between(xm.Mineo12S(Lmin=35).lumarr, medi,medu,color='b',alpha=.4)
    plt.fill_between(xm.Mineo12S(Lmin=35).lumarr, medi,medl,color='b',alpha=.4)
    plt.plot(xm.Mineo12S(Lmin=35).lumarr, medi,c='b',ls='-',lw=1.4)
    # plt.fill_between(xm.Mineo12S(Lmin=35).lumarr, meni,meni+stdi,color='c',alpha=.4)
    # plt.fill_between(xm.Mineo12S(Lmin=35).lumarr, meni,meni-stdi,color='c',alpha=.4)
    plt.plot(xm.Mineo12S(Lmin=35).lumarr, meni,c='orange',ls='-',lw=1.4)
    plt.plot(xm.Mineo12S(Lmin=35).lumarr, xm.Mineo12S(Lmin=35).model(),c='k',ls='--',lw=1.6)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"$L_X\,[\mathrm{erg\,s^{-1}}]$",fontsize=13)
    plt.ylabel(r"$N(>L)\times\mathrm{SFR^{-1}}$",fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
        
        
    bins = np.logspace(-2,2,9)
    cens = bins[:-1]+np.diff(bins)/2.
    hist = np.zeros(len(bins)-1)
    hist_u = np.zeros(len(bins)-1)
    hist_d = np.zeros(len(bins)-1)
    hist2 = np.zeros(len(bins)-1)
    hist2_u = np.zeros(len(bins)-1)
    hist2_d = np.zeros(len(bins)-1)
    hist4 = np.zeros(len(bins)-1)
    hist4_u = np.zeros(len(bins)-1)
    hist4_d = np.zeros(len(bins)-1)

    for i in range(len(bins)-1):
        hist[i] = np.median(nhxb[(sfr>bins[i])&(sfr<bins[i+1])])
        hist_u[i] = np.percentile(nhxb[(sfr>bins[i])&(sfr<bins[i+1])],84)
        hist_d[i] = np.percentile(nhxb[(sfr>bins[i])&(sfr<bins[i+1])],16)
        hist2[i] = np.median(nhxb2[(sfr>bins[i])&(sfr<bins[i+1])])
        hist2_u[i] = np.percentile(nhxb2[(sfr>bins[i])&(sfr<bins[i+1])],84)
        hist2_d[i] = np.percentile(nhxb2[(sfr>bins[i])&(sfr<bins[i+1])],16)
        # print(nhxb2[(sfr>bins[i])&(sfr<bins[i+1])])
        hist4[i] = np.median(nhxb4[(sfr>bins[i])&(sfr<bins[i+1])])
        hist4_u[i] = np.percentile(nhxb4[(sfr>bins[i])&(sfr<bins[i+1])],84)
        hist4_d[i] = np.percentile(nhxb4[(sfr>bins[i])&(sfr<bins[i+1])],16)

    plt.hlines(xm.Mineo12S().model()[0],.01,200.,color='k',label=r"$\log L_{\mathrm{min}}=35$")
    plt.hlines(xm.Mineo12S(Lmin=34).model()[0],.01,200.,color='k', ls="--", label=r"$\log L_{\mathrm{min}}=34$")
    # plt.errorbar(cens,hist,(hist-hist_d,hist_u-hist),ms=8.,capsize=3.,capthick=1.,c='b',fmt='x',label=r"$R^{\mathrm{MM}}_{\mathrm{SNII}}$")
    # plt.plot(sfr,nhxb4,'go',markeredgecolor="none",alpha=.3)
    # plt.plot(sfr,nhxb2,'ms',markeredgecolor="none",alpha=.3)
    plt.errorbar(cens*1.1,hist2,(hist2-hist2_d,hist2_u-hist2),ms=8.,capsize=3.,capthick=1.,c='m',fmt='x',label=r"$\eta_{\mathrm{HXB}}$")
    plt.errorbar(cens,hist4,(hist4-hist4_d,hist4_u-hist4),ms=8.,capsize=3.,capthick=1.,c='g',fmt='x',label=r"$R^{\mathrm{PM}}_{\mathrm{SNII}}$")
    plt.xlabel(r"$\mathrm{SFR}$",fontsize=13)
    plt.ylabel(r"$N_{\mathrm{HXB}}\times\mathrm{SFR}^{-1}$",fontsize=13)
    plt.legend(loc=0,fontsize=12,ncol=2)
    plt.xscale("log")
    plt.yscale("log")
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}')) # No decimal places
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
    print(np.average(nhxb,weights=sfr))
    print(np.average(nhxb2,weights=sfr))
    # print(np.average(nhxb3,weights=sfr))
    print(np.average(nhxb4,weights=sfr))
    plt.show()

    # plt.plot(sfr,nhxb*sfr,'gx',alpha=.3,label="SNII rate 34")
    # plt.plot(sfr,nhxb4*sfr,'mx',alpha=.3,label=r"$R^{\mathrm{PM}}_{\mathrm{SNII}}$")
    # plt.errorbar(cens,hist*cens,((hist-hist_d)*cens,(hist_u-hist)*cens),ms=8.,capsize=3.,capthick=1.,c='k',fmt='x',label=r"$R^{\mathrm{MM}}_{\mathrm{SNII}}$")
    plt.errorbar(cens,hist4*cens,((hist4-hist4_d)*cens,(hist4_u-hist4)*cens),ms=8.,capsize=3.,capthick=1.,c='g',fmt='x',label=r"$R^{\mathrm{PM}}_{\mathrm{SNII}}$")
    plt.plot(np.logspace(-2,2.3),np.logspace(-2,2.3)*187)
    plt.xlabel("SFR",fontsize=13)
    plt.ylabel(r"$N_{HXB}$",fontsize=13)
    plt.legend()
    plt.show()
    


    