"""
Constants for the XRB models
"""

##--- LMXB constants, Gilfanov04 ---##
#--- lumionosities in units of 1.e38 erg/s ---#

normG                   = 440.4
Lb1G                    = .19
Lb2G                    = 5.
a1G                     = 1.
a2G                     = 1.86
a3G                     = 4.8
LcutG                   = 500


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


##--- LMXB constants, Lehmer+19 ---##
#--- lumionosities in units of 1.e38 erg/s ---#

norm2: float            = 33.8
alph1: float            = 1.28
alph2: float            = 2.33
bre: float              = 1.48
cut: float              = 10**2.7 # (40.7 -38)

sig_norm2: float        = 5.
sig_alph1: float        = 0.06
sig_alph2: float        = 0.24
sig_bre: float          = 0.68
sig_cut: float          = 0.3

##--- LMXB constants, Lehmer+20, combination of GC LMXBs and field (in-situ + seeded) ---##
#--- luminositis in units of 1.e38 ---#

# GC
K_GC                    = 8.08
gamma_GC                = 1.08
cut_GC                  = 10**.61 #(38.61 - 38)

# field (no priors)
K_field                 = 42.4
a1_field                = 0.98
Lb_field                = 0.45
a2_field                = 2.43
cut_field               = 100

K_seed                  = 5.
gamma_seed              = 1.21
cut_seed                = 10**.66 #(38.66-38)


##--- HMXB constants, Grimm+03 ---##
#--- luminosities in units of 1.e38 ---#

norm_Gr                 = 3.3
gamma_Gr                = 1.61
Lcut_Gr                 = 240

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


##--- HMXB constants, Lehmer+19 ---##
#--- luminosity breaks in units of 1.e38 erg/s ---#
norm3: float            = 1.96
gam: float              = 1.65
cut                     = cut #as in LMXB model from Lehmer+19

sig_norm3: float        = 0.14
sig_gam: float          = 0.025

##--- HMXB constants, Lehmer+21, metallicity ---##

A_h: float              = 1.29
g1_h: float             = 1.74
g2_h: float             = 1.16
g2_logZ: float          = 1.34
logLb: float            = 38.54
logLc: float            = 39.98
logLc_logZ: float       = 0.6

sig_Ah: float           = 0.185
sig_g1h: float          = 0.04
sig_g2h: float          = 0.17
sig_g2logZ: float       = 0.5
sig_logLb: float        = 0.2
sig_logLc: float        = 0.24
sig_logLcZ: float       = 0.3
