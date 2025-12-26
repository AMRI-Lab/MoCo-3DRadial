import torch
import utils
import numpy as np
import h5py
import torchkbnufft as tkbn
import sigpy as sp
import sigpy.mri as mr
from numpy import fft
import cupy as cp
from tqdm import tqdm
import os

cp.cuda.Device(0)
plane_num = 240
stage_num = 48
plane_per_stage = 5
mot = 5
size_x = 224
size_z = 180
spoke_size = size_x * 2
grid_z = size_z
nChannel = 24
voxel_size = [1.0, 1.0, 1.0]
root = './data/'
kdata_path = '{}/kdata.h5'.format(root, mot)
angle_path = '{}/rotAngle.mat'.format(root)
out_path = '{}/recon/'.format(root)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

kdata = h5py.File(kdata_path, 'r')['kdata'][:].transpose(1,2,3,0)  # [ch,pe,ro,spe]->[pe,ro,spe,ch]
kdata = torch.tensor(kdata).to(torch.complex64).to(device)
print(kdata.shape)

mot_path = out_path
rot_motion_x = np.loadtxt('{}/rot_motion_x_est_{}.txt'.format(mot_path, plane_num)) # rad
rot_motion_y = np.loadtxt('{}/rot_motion_y_est_{}.txt'.format(mot_path, plane_num))
rot_motion_z = np.loadtxt('{}/rot_motion_z_est_{}.txt'.format(mot_path, plane_num))
shift_x = np.loadtxt('{}/shift_x_est_{}.txt'.format(mot_path, plane_num)) / voxel_size[0] # voxel (240)
shift_y = np.loadtxt('{}/shift_y_est_{}.txt'.format(mot_path, plane_num)) / voxel_size[1]
shift_z = np.loadtxt('{}/shift_z_est_{}.txt'.format(mot_path, plane_num)) / voxel_size[2]

with h5py.File(angle_path, 'r') as rotAngle:
    angle = rotAngle['rotAngle'][:]
    angle = np.deg2rad(angle[:,0])
print(angle.shape)
del rotAngle

# correct kspace
ktraj = utils.get_traj3d_stackofstar_plane_per_stage(omega=angle, spoke_size=spoke_size, grid_z=grid_z, 
                                         rot_motion_x=rot_motion_x, rot_motion_y=rot_motion_y, rot_motion_z=rot_motion_z).reshape(plane_num, spoke_size, grid_z, 3).to(torch.float32).to(device)

for i in range(stage_num):
    kdata[i*plane_per_stage:(i+1)*plane_per_stage,:,:,:] = kdata[i*plane_per_stage:(i+1)*plane_per_stage,:,:,:] \
                                                        / torch.exp(1j * (shift_x[i]*ktraj[i*plane_per_stage:(i+1)*plane_per_stage,:,:,0].unsqueeze(-1) + shift_y[i]*ktraj[i*plane_per_stage:(i+1)*plane_per_stage,:,:,1].unsqueeze(-1) + shift_z[i]*ktraj[i*plane_per_stage:(i+1)*plane_per_stage,:,:,2].unsqueeze(-1)))
print('correct kspace finished!')

# calc dcomp
dcomp = tkbn.calc_density_compensation_function(ktraj.view(1, -1, 3).permute(0, 2, 1), 
                                                im_size=(size_x, size_x, size_z),
                                                grid_size=(spoke_size, spoke_size, grid_z)).to(torch.float32).to(device)

# nufft
recon = []
adjnufft_ob = tkbn.KbNufftAdjoint(im_size=(size_x, size_x, size_z), 
                                  grid_size=(spoke_size, spoke_size, grid_z)).to(torch.complex64).to(device)
for i in tqdm(range(nChannel)):
    recon.append(adjnufft_ob(kdata[:,:,:,i].reshape(1,1,-1)*dcomp, 
                             ktraj.reshape(1, -1, 3).permute(0, 2, 1)).squeeze().cpu().numpy())
recon = np.array(recon)
nChannel, size_x, size_y, size_z = recon.shape
print('nufft finished!')

# calc smap
kdata_grid = fft.ifftshift(fft.fftn(fft.fftshift(recon, axes=(1,2)),axes=(1,2)),axes=(1,2)) # [ch, kx, ky, z]
smap = cp.zeros((nChannel, size_x, size_y, size_z), dtype=np.complex64)
for i in tqdm(range(size_z)):
    kdata_grid_slice = cp.asarray(kdata_grid[..., i])
    smap[..., i] = mr.app.EspiritCalib(kdata_grid_slice, device=sp.get_device(kdata_grid_slice), show_pbar=False).run()
smap = smap.get()
del kdata_grid
print('calc smap finished!')

# coil combine
recon_combine = np.sum(smap.conj() * recon, axis=0, keepdims=False)
recon_combine[np.isinf(recon_combine)] = 0
recon_combine[np.isnan(recon_combine)] = 0
print('coil combine finished!')

utils.save_nii(np.abs(recon_combine), os.path.join(out_path,'recon_combine_mag.nii'), voxel_size=voxel_size)
utils.save_nii(np.angle(recon_combine), os.path.join(out_path,'recon_combine_phi.nii'), voxel_size=voxel_size)
utils.save_nii(np.abs(smap.transpose(1,2,3,0)), os.path.join(out_path,'smap_mag.nii'), voxel_size=voxel_size)
utils.save_nii(np.angle(smap.transpose(1,2,3,0)), os.path.join(out_path,'smap_phi.nii'), voxel_size=voxel_size)