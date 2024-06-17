import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys
import warnings
import h5py
import datetime

#### TO EDIT ####
bins = 640
x_range = (20,80)
vel_bins =  np.linspace(-400000,400000,65)
proj_ax = 2 #for 0_0
name = 'm2e4_m11_'
####

savepath = '/u/kneralwar/ptmp_link/starforge_cubes/M2e4a2_fits_ff_mask_64x640x640_2024_06_17/'

snap = int(sys.argv[1])

x_name = name + str(snap).zfill(3) + '_90_90' 
y_name = name + str(snap).zfill(3) + '_90_0' 
z_name = name + str(snap).zfill(3) + '_0_0' 

fileEnding = 'KSter.fits'

hd_path = '/u/kneralwar/ptmp_link/starforge/M2e4_alpha2_fiducial/'
f = h5py.File(hd_path + 'snapshot_' + str(snap).zfill(3) +'.hdf5', 'r')
data=f['PartType0']

coord = data['Coordinates'][()]
vel=data['Velocities'][()]
mass = data['Masses'][()]
mass_m11 = data['Masses'][()] * data['Metallicity'][()][:,11]
mass_m12 = data['Masses'][()] * data['Metallicity'][()][:,12]
mass_m13 = data['Masses'][()] * data['Metallicity'][()][:,13]

f.close()

x = coord[:,0]
y = coord[:,1]
z = coord[:,2]

vx1 = vel[:,0]
vy1 = vel[:,1]
vz1 = vel[:,2]

big_m11_z2 = np.zeros([len(vel_bins[:-1]), bins, bins])

# for vel1, vel2 in zip(vel_bins[:-1], vel_bins[1:]):
for i, (vel1, vel2) in enumerate(zip(vel_bins[:-1], vel_bins[1:])):
    mass_vel = mass.copy()
    mass_m11_vel = mass_m11.copy()
    mass_vel = np.where((vel[:,proj_ax] > vel1) & (vel[:,proj_ax] < vel2), mass_vel, 0)
    mass_m11_vel = np.where((vel[:,proj_ax] > vel1) & (vel[:,proj_ax] < vel2), mass_m11_vel, 0)
    mass_g, edges = np.histogramdd((x,y,z), bins = bins, range = (x_range,x_range,x_range), weights = mass_vel)
    mass_m11_g, edges = np.histogramdd((x,y,z), bins = bins, range = (x_range,x_range,x_range), weights = mass_m11_vel)
    mass_mean_z2 = np.nansum(mass_g, axis = proj_ax)
    mass_m11_mean_z2 = np.nansum(mass_m11_g, axis = proj_ax)
    m11_mean_z2 = np.divide(mass_m11_mean_z2, mass_mean_z2, out=np.zeros_like(mass_mean_z2), where=mass_mean_z2==np.nan)
    big_m11_z2[i] = m11_mean_z2
    
big_m11_z2 = np.array(big_m11_z2)
big_m11_z = np.swapaxes(big_m11_z2,1,2)

hdu = fits.PrimaryHDU(np.array(big_m11_z))
hdu.header['BITPIX'] = -32
hdu.header['NAXIS'] = 3
hdu.header['NAXIS1'] = big_m11_z.shape[0] 
hdu.header['NAXIS2'] = big_m11_z.shape[1]
hdu.header['NAXIS3'] = big_m11_z.shape[2]
hdu.header['BUNIT'] = 'Msun'
hdu.header['CTYPE0'] = 'velocity'
hdu.header['CDELT0'] = vel_bins[1] - vel_bins[0]
hdu.header['CTYPE1'] = 'y'
hdu.header['CDELT1'] = (x_range[1] - x_range[0]) / bins
hdu.header['CTYPE2'] = 'x'
hdu.header['CDELT2'] = (x_range[1] - x_range[0]) / bins
hdu.header['DATE'] = str(datetime.datetime.now().date())
hdu.header['software'] = 'Python'
hdu.header['creator'] = 'k-neralwar'
hdu.header['method'] = 'numpy_hist'


hdu.data = big_m11_z

fits.writeto('%s%s%s' % (savepath,z_name, fileEnding), np.float32(hdu.data), hdu.header, overwrite=True)
