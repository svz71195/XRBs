"""
Constants for the XRB models
"""

##--- LMXB constants, Zhang+2012 ---##

#--- slopes with error ---#
alpha1: float           = 1.02  #+0.07 -0.08
alpha2: float           = 2.06  #+0.06 -0.05
alpha3: float           = 3.63  #+0.67 -0.49
sig_a1: float           = 0.075
sig_a2: float           = 0.055
sig_a3: float           = 0.58
#--- luminosity breaks in units of 1.e36 erg/s---#
Lb1: float              = 54.6  #+4.3 -3.7
Lb2: float              = 599.  #+95 -67
Lcut_L: float           = 5.e4
sig_Lb1: float          = 4.e-2
sig_Lb2: float          = (95+67)/2
#--- normalization in units of 1.e11 Msol ---#
norm1: float            = 1.01  #+-0.28; per 10^11 solar masses
sig_K1: float           = 0.28


##--- HMXB constants, Mineo+12, single power law ---##

Lcut_Hs: float          = 1.e3  # in units of 1.e38 erg/s
gamma_s: float          = 1.59  #+-0.25 (rms, Mineo 2012)
xi_s: float             = 1.88  #*/ 10^(0.34) (rms=0.34 dex, Mineo 2012)

log_xi_s: float         = 0.27415 #log10
sig_gam_s: float        = 0.25
log_sig_xi_s: float     = 0.34


##--- HMXB constants, Mineo+12, broken power law ---##

#--- luminosity breaks in units of 1.e38 erg/s
Lcut_Hb: float          = 5.e3
LbH: float              = 110.

gamma1_b: float         = 1.58
gamma2_b: float         = 2.73
xi2_b: float            = 1.49

sig_LbH: float          = (57+34)/2
sig_g1: float           = 0.02
sig_g2: float           = (1.58+0.54)/2
sig_xi2: float          = 0.07