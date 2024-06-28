#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import matplotlib.pylab as plt


import time
from tqdm import tqdm
from tqdm._tqdm import trange  #### tqdm for visualization


folder = '/lcrc/project/cosmo_ai/dongx/512-data/'
print("testing path folder")
plt.figure()
plt.savefig(folder + 'testfigure.png')
plt.clf()
print("test is over.")

def whitenoise3d(N):    
    return np.random.normal(0,1,size=(N,N,N))

def whitenoise(N):    
    return np.random.normal(0,1,size=(N,N))


def removeNaN(ar):
    """Remove infinities and NaNs from an array"""
    ar[ar!=ar]=0
    ar[ar==np.inf]=0



def apply_powerlaw_power_spectrum(f, n=-1.0,min_freq=2.0,max_freq=200.0):
    f_fourier = np.fft.fft2(f)
    freqs = np.fft.fftfreq(f.shape[0])
    freqs_2 = np.sqrt(freqs[:,np.newaxis]**2+freqs[np.newaxis,:]**2)
    f_fourier[freqs_2<min_freq/f.shape[0]]=0
    f_fourier[freqs_2>max_freq/f.shape[0]]=0
    freqs_2**=n
    removeNaN(freqs_2)
    f_fourier*=freqs_2
    return np.fft.ifft2(f_fourier).real

def apply_powerlaw_power_spectrum3d(f, n=-1.0,min_freq=2.0,max_freq=200.0):
    f_fourier = np.fft.fftn(f)
    freqs = np.fft.fftfreq(f.shape[0])
    freqs_2 = np.sqrt(freqs[:,np.newaxis,np.newaxis]**2+freqs[np.newaxis,:,np.newaxis]**2+freqs[np.newaxis,np.newaxis,:]**2)
    f_fourier[freqs_2<min_freq/f.shape[0]]=0
    f_fourier[freqs_2>max_freq/f.shape[0]]=0
    freqs_2**=n
    removeNaN(freqs_2)
    f_fourier*=freqs_2
    return np.fft.ifftn(f_fourier).real



def get_potential_gradients(den_real):
    """Starting from a density field in 2D, get the potential gradients i.e.
    returns the two components of grad (grad^-2 den_real)"""
    den = np.fft.fft2(den_real)
    freqs = np.fft.fftfreq(den.shape[0])
    del_sq_operator = -(freqs[:,np.newaxis]**2+freqs[np.newaxis,:]**2)

    grad_x_operator = -1.j*np.fft.fftfreq(den.shape[0])[:,np.newaxis]
    grad_y_operator = -1.j*np.fft.fftfreq(den.shape[0])[np.newaxis,:]

    phi = den/del_sq_operator
    removeNaN(phi)

    grad_phi_x = grad_x_operator*phi
    grad_phi_y = grad_y_operator*phi

    grad_phi_x_real = np.fft.ifft2(grad_phi_x).real
    grad_phi_y_real = np.fft.ifft2(grad_phi_y).real

    return grad_phi_x_real, grad_phi_y_real



def get_potential_gradients3d(den_real):
    """Starting from a density field in 2D, get the potential gradients i.e.
    returns the two components of grad (grad^-2 den_real)"""
    den = np.fft.fftn(den_real)
    freqs = np.fft.fftfreq(den.shape[0])
    del_sq_operator = -(freqs[:,np.newaxis,np.newaxis]**2+freqs[np.newaxis,:,np.newaxis]**2+freqs[np.newaxis,np.newaxis,:]**2)

    grad_x_operator = -1.j*np.fft.fftfreq(den.shape[0])[:,np.newaxis,np.newaxis]
    grad_y_operator = -1.j*np.fft.fftfreq(den.shape[0])[np.newaxis,:,np.newaxis]
    grad_z_operator = -1.j*np.fft.fftfreq(den.shape[0])[np.newaxis,np.newaxis,:]
    phi = den/del_sq_operator
    removeNaN(phi)
    
    grad_phi_x = grad_x_operator*phi
    grad_phi_y = grad_y_operator*phi
    grad_phi_z = grad_z_operator*phi
    
    grad_phi_x_real = np.fft.ifftn(grad_phi_x).real
    grad_phi_y_real = np.fft.ifftn(grad_phi_y).real
    grad_phi_z_real = np.fft.ifftn(grad_phi_z).real
    
    return grad_phi_x_real, grad_phi_y_real, grad_phi_z_real



def evolved_particle_positions(den, time=0.025):
    """Generate a grid of particles, one for each cell of the density field,
    then displace those particles along gradient of potential implied by
    the density field."""
    N = len(den)
    x,y = np.mgrid[0.:N,0.:N]
    grad_x, grad_y = get_potential_gradients(den)
    x+=time*grad_x
    y+=time*grad_y
    x[x>N]-=N
    y[y>N]-=N
    x[x<0]+=N
    y[y<0]+=N
    return x.flatten(),y.flatten()

def evolved_particle_positions3d(den, time=0.025):
    """Generate a grid of particles, one for each cell of the density field,
    then displace those particles along gradient of potential implied by
    the density field."""
    N = resolution
    x,y,z = np.mgrid[0.:N,0.:N,0.:N]
    grad_x, grad_y, grad_z = get_potential_gradients3d(den)
    
    x+=time*grad_x
    y+=time*grad_y
    z+=time*grad_z
    
    x[x>N]-=N
    y[y>N]-=N
    z[z>N]-=N
    
    x[x<0]+=N
    y[y<0]+=N
    z[z<0]+=N
    
    return x.flatten(),y.flatten(),z.flatten()


def evolved_particle_positions3d_nonflat(den, time):
    """Generate a grid of particles, one for each cell of the density field,
    then displace those particles along gradient of potential implied by
    the density field."""
    N = resolution
    x,y,z = np.mgrid[0.:N,0.:N,0.:N]
    grad_x, grad_y, grad_z = get_potential_gradients3d(den)
    
    x+=time*grad_x
    y+=time*grad_y
    z+=time*grad_z
    
    
    while((np.amax(x)>N ) | (np.amax(y)>N) | (np.amax(z)>N ) | (np.amin(x)<0) | (np.amin(y)<0) | (np.amin(z)<0)): 
      x[x>N]-=N
      y[y>N]-=N
      z[z>N]-=N
    
      x[x<0]+=N
      y[y<0]+=N
      z[z<0]+=N
    
    return x,y,z


def evolved_particle_positions_nonflat(den, time):
    """Generate a grid of particles, one for each cell of the density field,
    then displace those particles along gradient of potential implied by
    the density field."""
    N = resolution
    x,y = np.mgrid[0.:N,0.:N]
    grad_x, grad_y = get_potential_gradients(den)
    x+=time*grad_x
    y+=time*grad_y
    
    
    while( (np.amax(x)>N) | (np.amax(y)>N) | (np.amin(x)<0) | (np.amin(y)<0)):
      x[x>N]-=N
      y[y>N]-=N
      x[x<0]+=N
      y[y<0]+=N
    return x,y


def densityCIC(x,y):   #  0 <  x, y, z < nGr in 1D 
    #x = (x-x_l)*Nx/(x_h - x_l)
    #y = (y-y_l)*Ny/(y_h - y_l) 
    Np = np.size(x)
    macro = np.zeros([nGrid, nGrid])
    for particle in range(Np):
        i = int(x[particle]) 
        j = int(y[particle]) 
        dx = dy = 1
        
        a1 = np.around(x[particle], decimals = 4) - i*dx
        b1 = np.around(y[particle], decimals = 4) - j*dy   
        
        wx1 = a1/dx
        wx2 = (dx - a1)/dx
        wy1 = b1/dy
        wy2 = (dy - b1)/dy        
        
        macro[i, j] += (wx1 * wy1)
        macro[np.mod(i+1,nGrid), j] += (wx2 * wy1)
        macro[i, np.mod(j+1,nGrid)] += (wx1 * wy2)
        macro[np.mod(i+1,nGrid), np.mod(j+1,nGrid)] += (wx2 * wy2 )
    return macro


def densityCIC3d(x,y,z):   #  0 <  x, y, z < nGr in 1D 
    #x = (x-x_l)*Nx/(x_h - x_l)
    #y = (y-y_l)*Ny/(y_h - y_l) 
    Np = np.size(x)
    macro = np.zeros([nGrid, nGrid, nGrid])
    for particle in range(Np):
        i = int(x[particle]) 
        j = int(y[particle]) 
        k = int(z[particle])
        
        dx = dy = dz = 1
        
        a1 = np.around(x[particle], decimals = 4) - i*dx
        b1 = np.around(y[particle], decimals = 4) - j*dy   
        c1 = np.around(z[particle], decimals = 4) - k*dz
        
        wx1 = a1/dx
        wx2 = (dx - a1)/dx
        
        
        wy1 = b1/dy
        wy2 = (dy - b1)/dy        
        
        
        wz1 = c1/dz
        wz2 = (dz - c1)/dz
        
        
        
        macro[i, j, k] += (wx1 * wy1 * wz1)
        
        macro[np.mod(i+1,nGrid), j, k ] +=(wx2 * wy1 * wz1)
        
        macro[i, np.mod(j+1,nGrid), k] += (wx1 * wy2 * wx1)
        
        macro[i, j , np.mod(k+1,nGrid)] += (wx1 * wy1* wz2)
        
        macro[np.mod(i+1,nGrid), np.mod(j+1,nGrid), k] += (wx2 * wy2 * wz1)
        
        macro[np.mod(i+1,nGrid), j, np.mod(k+1,nGrid)] += (wx2 * wy1 * wz2)
        
        macro[i, np.mod(j+1,nGrid), np.mod(k+1,nGrid)] += (wx1 * wy2 * wz2)
        
        macro[np.mod(i+1,nGrid), np.mod(j+1,nGrid), np.mod(k+1,nGrid)] += (wx2 * wy2 * wz2)
        
        
        
      #  macro[np.mod(i+1,nGrid), j] += (wx2 * wy1)
      #  macro[i, np.mod(j+1,nGrid)] += (wx1 * wy2)
      #  macro[np.mod(i+1,nGrid), np.mod(j+1,nGrid)] += (wx2 * wy2 )
    return macro


#### from mpl_toolkits import mplot3d

resolution = 512 #256 #128 #64
nGrid = resolution
num_samples = 1000  ## not used anymore

#a = np.logspace(np.log10(1),np.log10(0.1),10)
#print(a)
#times=np.logspace(np.log10(0.0001),np.log10(0.08),10)
#times=np.linspace(0.0001,0.08,10)  #### This is what generated num_xxxx simulations!!!!#####

#### timesteps 

times=np.logspace(np.log10(0.01),np.log10(1),10)
print(times)  #### [times] array is the series of times that we generate snapshots. Simply change it and range of j correspondingly.

#linear_field = apply_powerlaw_power_spectrum3d(whitenoise3d(resolution))    
#times = 1/(1+a)
#print(times)

train_number = 160
test_number = 200
val_number = 200

times = np.array([0.04641589, 0.21544347])

num_timesteps = times.shape[0]
print("number of timesteps generated is:" + str(num_timesteps))

N_Grid_x, N_Grid_y, N_Grid_z = np.mgrid[0.:nGrid,0.:nGrid,0.:nGrid]

for j in [range(num_timesteps)]:
  #datamatrix = np.zeros((num_samples,resolution,resolution))  
  lagr_coord = np.zeros((train_number,nGrid,nGrid,nGrid,3))
  for i in trange(train_number):
    np.random.seed(i)
    
    linear_field = apply_powerlaw_power_spectrum3d(whitenoise3d(resolution))

    #plt.imshow(linear_field)
    #x0, y0 = evolved_particle_positions_nonflat(linear_field, time=t0)
    #x00,y00= evolved_particle_positions_nonflat(linear_field, time=t0)
    x1, y1, z1 = evolved_particle_positions3d_nonflat(linear_field, time=times[j])
    
    #print(N_Grid_x.shape)
    ##### print("time: " + str(j) +" , number: " + str(i))
    #print(np.amax(y1))
    lagr_coord[i,:,:,:,0] = x1-N_Grid_x
    lagr_coord[i,:,:,:,1] = y1-N_Grid_y
    lagr_coord[i,:,:,:,2] = z1-N_Grid_z
    

    lagr_coord[lagr_coord>nGrid/2]-=nGrid
    lagr_coord[lagr_coord<-nGrid/2]+=nGrid
    
  np.save(folder +'lagr_'+str(j)+'_train_160', lagr_coord)
  print("mission accomplished for training timestep "+str(j))




for j in range(num_timesteps):
  #datamatrix = np.zeros((num_samples,resolution,resolution))  
  lagr_coord = np.zeros((test_number,nGrid,nGrid,nGrid,3))
  for i in trange(train_number, train_number+test_number):
    np.random.seed(i)
    
    linear_field = apply_powerlaw_power_spectrum3d(whitenoise3d(resolution))

    #plt.imshow(linear_field)
    #x0, y0 = evolved_particle_positions_nonflat(linear_field, time=t0)
    #x00,y00= evolved_particle_positions_nonflat(linear_field, time=t0)
    x1, y1, z1 = evolved_particle_positions3d_nonflat(linear_field, time=times[j])
    #print(N_Grid_x.shape)
    
    #print("time: " + str(j) +" , number: " + str(i))
    ii = i - train_number
    #print(np.amax(y1))
    lagr_coord[ii,:,:,:,0] = x1-N_Grid_x
    lagr_coord[ii,:,:,:,1] = y1-N_Grid_y
    lagr_coord[ii,:,:,:,2] = z1-N_Grid_z
    

    lagr_coord[lagr_coord>nGrid/2]-=nGrid
    lagr_coord[lagr_coord<-nGrid/2]+=nGrid
    
  np.save(folder +'lagr_'+str(j)+'_test_160', lagr_coord)
  print("mission accomplished for test timestep "+str(j))


for j in range(num_timesteps):
  #datamatrix = np.zeros((num_samples,resolution,resolution))  
  lagr_coord = np.zeros((val_number,nGrid,nGrid,nGrid,3))
  for i in trange(train_number+test_number, train_number+test_number+val_number):
    np.random.seed(i)
    
    linear_field = apply_powerlaw_power_spectrum3d(whitenoise3d(resolution))

    #plt.imshow(linear_field)
    #x0, y0 = evolved_particle_positions_nonflat(linear_field, time=t0)
    #x00,y00= evolved_particle_positions_nonflat(linear_field, time=t0)
    x1, y1, z1 = evolved_particle_positions3d_nonflat(linear_field, time=times[j])
    #print(N_Grid_x.shape)
    #print(np.amax(y1))

    #print("time: " + str(j) +" , number: " + str(i))
    ii = i - train_number - test_number
    
    lagr_coord[ii,:,:,:,0] = x1-N_Grid_x
    lagr_coord[ii,:,:,:,1] = y1-N_Grid_y
    lagr_coord[ii,:,:,:,2] = z1-N_Grid_z
    

    lagr_coord[lagr_coord>nGrid/2]-=nGrid
    lagr_coord[lagr_coord<-nGrid/2]+=nGrid
    
  np.save(folder +'lagr_'+str(j)+'_val', lagr_coord)
  print("mission accomplished for val timestep "+str(j))




from datetime import datetime
from datetime import date
now = datetime.now()

current_time = now.strftime("%H:%M:%S")
today = date.today()


print("mission finally accomplished!")
print(today)

##### runlog

with open(folder+"Data_log_test.txt", "w") as f:   # Opens file and casts as f 
    f.write("The ZA dataset is generated at:" + current_time + " Central Time , " +  str(today) + " by XFD. \n") 
    f.write("It contains: \n " + str(train_number)+ " training samples \n")  
    f.write("And  " + str(test_number)+ " test samples\n") 
    f.write("And  " + str(val_number)+ " test samples\n")
    f.write("Data dimension:  "+ str(resolution)+ "^3 \n")
    f.write("The total number of timesteps is: 10 \n "  )
    f.write("And the corresponding times are: " + str(times))
    f.write("------------------------------------ Mission Accomplished ----------------------------")      # Writing




