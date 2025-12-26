import torch
import torch.nn as nn
from torch.utils import data
import yaml
import numpy as np
import nibabel as nib
import h5py

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def save_nii(arr,path,voxel_size=[1,1,1]):
    affine=np.array([[voxel_size[0],0,0,0],
                     [0,voxel_size[1],0,0],
                     [0,0,voxel_size[2],0],
                     [0,0,0,1]])
    nib.Nifti1Image(arr,affine).to_filename(path)
    return None

def angle_to_rot_z(theta): 
    # theta: (n, ), torch tensor
    rot_z = torch.zeros(size=(theta.shape[0], 3, 3)).to(theta.device)
    rot_z[:, 0, 0] = torch.cos(theta)
    rot_z[:, 0, 1] = torch.sin(theta)
    rot_z[:, 0, 2] = 0
    rot_z[:, 1, 0] = -torch.sin(theta)
    rot_z[:, 1, 1] = torch.cos(theta)
    rot_z[:, 1, 2] = 0
    rot_z[:, 2, 0] = 0
    rot_z[:, 2, 1] = 0
    rot_z[:, 2, 2] = 1
    return rot_z   # [n, 3, 3]


def angle_to_rot_y(theta):
    # theta: (n, ), torch tensor
    rot_y = torch.zeros(size=(theta.shape[0], 3, 3)).to(theta.device)
    rot_y[:, 0, 0] = torch.cos(theta)
    rot_y[:, 0, 1] = 0
    rot_y[:, 0, 2] = -torch.sin(theta)
    rot_y[:, 1, 0] = 0
    rot_y[:, 1, 1] = 1
    rot_y[:, 1, 2] = 0
    rot_y[:, 2, 0] = torch.sin(theta)
    rot_y[:, 2, 1] = 0
    rot_y[:, 2, 2] = torch.cos(theta)
    return rot_y   # [n, 3, 3]

def angle_to_rot_x(theta): 
    rot_x = torch.zeros(size=(theta.shape[0], 3, 3)).to(theta.device)
    rot_x[:, 0, 0] = 1
    rot_x[:, 0, 1] = 0
    rot_x[:, 0, 2] = 0
    rot_x[:, 1, 0] = 0
    rot_x[:, 1, 1] = torch.cos(theta)
    rot_x[:, 1, 2] = torch.sin(theta)
    rot_x[:, 2, 0] = 0
    rot_x[:, 2, 1] = -torch.sin(theta)
    rot_x[:, 2, 2] = torch.cos(theta)
    return rot_x   # [n, 3, 3]


def grid_coordinate(h, w, d):
    x = np.linspace(-1, 1, h, endpoint=True)
    y = np.linspace(-1, 1, w, endpoint=True)
    z = np.linspace(-1, 1, d, endpoint=True)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')  # (h, w, d)
    xyz = np.stack([x, y, z], -1)  # (h, w, d, 3)
    return xyz
    
class MYTVLoss(torch.nn.Module):
    def __init__(self):
        super(MYTVLoss, self).__init__()

    def forward(self, x):
        L_RO, L_PE, L_SPE = x.shape
        tv_loss = ( torch.sum(torch.abs(x[1:, :, :] - x[:L_RO-1, :, :])) + torch.sum(torch.abs(x[:,1:,:] - x[:,:L_PE-1,:])) + torch.sum(torch.abs(x[:,:,1:] - x[:,:,:L_SPE-1])) )/ ((L_RO-1)*(L_PE-1)*(L_SPE-1))
        return tv_loss
    
class PoseRefiner(nn.Module):
    def __init__(self, rot_motion_x, rot_motion_y, rot_motion_z, offset, rot_is_train, offset_is_train):
        super(PoseRefiner, self).__init__()
        # dim [num_stage, 1, 1]
        self.rot_motion_x = nn.Parameter(torch.tensor(rot_motion_x, dtype=torch.float), requires_grad=rot_is_train)
        # dim [num_stage, 1, 1]
        self.rot_motion_y = nn.Parameter(torch.tensor(rot_motion_y, dtype=torch.float), requires_grad=rot_is_train)
        # dim [num_stage, 1, 1]
        self.rot_motion_z = nn.Parameter(torch.tensor(rot_motion_z, dtype=torch.float), requires_grad=rot_is_train)
        # dim [num_stage, 1, 3]
        self.offset = nn.Parameter(torch.tensor(offset, dtype=torch.float), requires_grad=offset_is_train)

    def forward(self, stage_id, xyz):
        # batch_size, 3, 3, -> batch_size, 1, 1, 3, 3
        rot_mat_z = angle_to_rot_z(theta=self.rot_motion_z[stage_id].view(-1, )).unsqueeze(1).unsqueeze(1)
        rot_mat_y = angle_to_rot_y(theta=self.rot_motion_y[stage_id].view(-1, )).unsqueeze(1).unsqueeze(1)
        rot_mat_x = angle_to_rot_x(theta=self.rot_motion_x[stage_id].view(-1, )).unsqueeze(1).unsqueeze(1)
        offset_vec = self.offset[stage_id].unsqueeze(1).unsqueeze(1)
        return torch.matmul(torch.matmul(torch.matmul(xyz.unsqueeze(0), rot_mat_x), rot_mat_y), rot_mat_z) + offset_vec  

class TestData(data.Dataset):
    def __init__(self, h, w, d):
        self.xyz = grid_coordinate(h=h, w=w, d=d)  # (x,y,z,3)

    def __len__(self):
        return self.xyz.shape[2]

    # xyz.shape=[x*y*z, 3]，先变z再变y最后变x，取spoke_size//2 * grid_size个点，刚好是一个x对应的yz平面
    def __getitem__(self, item):
        xyz = self.xyz[:,:,item,:]
        return xyz
    

class TrainData(data.Dataset):  
    def __init__(self, kdata_path, angle_path, stage_num):
        self.stage_num = stage_num
        # load kdata
        self.kdata = h5py.File(kdata_path, 'r')['kdata'][:].transpose(3,1,0,2)
        self.grid_z, self.plane_num, self.nChannel, self.spoke_size = self.kdata.shape
        self.grid_z = self.grid_z//2
        self.spoke_size = self.spoke_size//2
        self.kdata = self.kdata[self.grid_z//2:self.grid_z//2*3, :, :, self.spoke_size//2:self.spoke_size//2*3]
        self.kdata = self.kdata / np.max(np.abs(self.kdata)) * 1000
        print(self.kdata.shape)
        self.plane_per_stage = int(self.plane_num/self.stage_num)
        
        with h5py.File(angle_path, 'r') as rotAngle:
            angle = rotAngle['rotAngle'][:]
            angle = np.deg2rad(angle[:,0])
        print(angle.shape)
        del rotAngle

        # calc ktraj
        rot_z = angle_to_rot_z(torch.tensor(angle))   
        kx = np.linspace(-np.pi, np.pi, self.spoke_size, endpoint=True)
        kz = np.linspace(-np.pi, np.pi, self.grid_z, endpoint=True)
        kx, kz = np.meshgrid(kx, kz, indexing='ij')  
        ky = np.zeros_like(kx)  
        kxyz_init = np.stack([kx, ky, kz], -1).reshape(1, -1, 3)
        self.ktraj = torch.matmul(torch.tensor(kxyz_init).float(), rot_z).reshape(self.plane_num, self.spoke_size, self.grid_z, 3)
        print('calc traj finished!')

        self.rays = grid_coordinate(h=self.spoke_size//2, w=self.spoke_size//2, d=self.grid_z) # [x,y,z,3]

    def __len__(self):
        return self.stage_num

    def __getitem__(self, item):
        kdata_sample = self.kdata[:, item*self.plane_per_stage : (item+1)*self.plane_per_stage, :, :].transpose(2,1,3,0) # [ch,plane_per_stage,spoke_size,grid_z]
        ktraj_sample = self.ktraj[item*self.plane_per_stage : (item+1)*self.plane_per_stage, :, :, :].reshape(1, -1, 3).permute(0,2,1) # (1, 3, plane_per_stage*spoke_size*grid_z)
        return item, self.rays, kdata_sample, ktraj_sample 
    
def get_traj3d_stackofstar_plane_per_stage(omega, spoke_size, grid_z, rot_motion_x, rot_motion_y, rot_motion_z):
    rot_z = angle_to_rot_z(torch.tensor(omega))   # plane_num, 3, 3  # k线逆时针转

    kx = np.linspace(-np.pi, np.pi, spoke_size, endpoint=True)
    kz = np.linspace(-np.pi, np.pi, grid_z, endpoint=True)
    kx, kz = np.meshgrid(kx, kz, indexing='ij')  # (spoke_size, size_z)
    ky = np.zeros_like(kx)  # (spoke_size, size_z)

    kxyz_init = np.stack([kx, ky, kz], -1).reshape(1, -1, 3)  # (1, spoke_size*size_z, 3)
    kxyz = torch.matmul(torch.tensor(kxyz_init).float(), rot_z)     # (plane_num, spoke_size*size_z, 3)

    plane_per_stage=int(omega.shape[0]/rot_motion_z.shape[0])

    rot_motion_z = angle_to_rot_z(torch.tensor(rot_motion_z))   # stage_num, 3, 3  #头顺时针转
    rot_motion_y = angle_to_rot_y(torch.tensor(rot_motion_y))   # stage_num, 3, 3
    rot_motion_x = angle_to_rot_x(torch.tensor(rot_motion_x))   # stage_num, 3, 3
    rot_motion_z = torch.repeat_interleave(rot_motion_z, repeats=plane_per_stage, dim=0) # plane_num, 3, 3
    rot_motion_y = torch.repeat_interleave(rot_motion_y, repeats=plane_per_stage, dim=0) # plane_num, 3, 3
    rot_motion_x = torch.repeat_interleave(rot_motion_x, repeats=plane_per_stage, dim=0) # plane_num, 3, 3

    kxyz_motion = torch.matmul(torch.matmul(torch.matmul(kxyz, rot_motion_x), rot_motion_y), rot_motion_z)  # matmul将最后两个维度相乘，第一个维度为batch，自动调整至相同大小。[spoke_size*size_z, 3] * [3,3]
    return kxyz_motion