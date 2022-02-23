import numpy as np

CM_PER_MPC = 3.085678e24

phox_file = input("give name of phox package file to see Header INFO: \n")

with open(phox_file,"rb") as f:
    e_min       = np.fromfile(f,np.float64,count=1)
    e_max       = np.fromfile(f,np.float64,count=1)
    n_chan      = np.fromfile(f,np.int32,count=1)
    temp_min    = np.fromfile(f,np.float32,count=1)
    delta_temp  = np.fromfile(f,np.float32,count=1)
    area        = np.fromfile(f,np.float64,count=1)
    time        = np.fromfile(f,np.float64,count=1)
    Da          = np.fromfile(f,np.float64,count=1)
    zz          = np.fromfile(f,np.float64,count=1)
    zz_obs      = np.fromfile(f,np.float64,count=1)
    om0M        = np.fromfile(f,np.float64,count=1)
    om0L        = np.fromfile(f,np.float64,count=1)
    h0          = np.fromfile(f,np.float64,count=1)
    nph         = np.fromfile(f,np.ulonglong,count=1)
    active_part = np.fromfile(f,np.ulonglong,count=1)
    
print("e_min = ",e_min)
print("e_max = ",e_max)
print("n_chan = ",n_chan)
print("temp_min = ",temp_min)
print("delta_temp = ",delta_temp)
print("Aeff = ",area)
print("exp_time = ",time)
print("Da = ",Da/CM_PER_MPC)
print("Redshift = ",zz)
print("ObservedRedshift = ",zz_obs)
print("om0M = ",om0M)
print("om0L = ",om0L)
print("h0 = ",h0)
print("nph = ",nph)
print("active_part = ",active_part)
