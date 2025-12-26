import os
import numpy as np
import torch
import tinycudann as tcnn
import torchkbnufft as tkbn
from torch.utils import data
from torch.optim import lr_scheduler
from tqdm import tqdm
import utils

def train(config):   

    # file
    # -----------------------
    in_path = config["file"]["in_dir"]
    out_dir = config["file"]["out_dir"]
    kdata_path = os.path.join(in_path, 'kdata.h5')   # 这里也要改
    angle_path = os.path.join(in_path, 'rotAngle.mat')
    plane_num = config["file"]["plane_num"]
    stage_num = config["file"]["stage_num"]
    nChannel = config["file"]["nChannel"]
    plane_per_stage = int(plane_num/stage_num)
    spoke_size = config["file"]["spoke_size"]
    grid_z = config["file"]["grid_z"]
    voxel_size = np.array(config["file"]["voxel_size"])

    # parameter
    # -----------------------
    lr = config["train"]["lr"]
    gpu = config["train"]["gpu"]
    epoch = config["train"]["epoch"]
    save_epoch = config["train"]["save_epoch"]
    lr_decay_epoch = config["train"]["lr_decay_epoch"]
    lr_decay_rate = config["train"]["lr_decay_rate"]
    batch_size = config["train"]["batch_size"]
    device = torch.device('cuda:{}'.format(str(gpu) if torch.cuda.is_available() else 'cpu'))

    # data
    # -----------------------
    train_loader = data.DataLoader(
        dataset=utils.TrainData(kdata_path=kdata_path, angle_path=angle_path, stage_num=stage_num),
        batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(
        dataset=utils.TestData(h=spoke_size//2, w=spoke_size//2, d=grid_z), batch_size=1, shuffle=False)
    rays = torch.tensor(utils.grid_coordinate(h=spoke_size//2, w=spoke_size//2, d=grid_z)).to(device).float() # [x,y,z,3]
    
    # model & optimizer
    # -----------------------
    nufft_ob = tkbn.KbNufft(im_size=(spoke_size//2, spoke_size//2, grid_z),
                            grid_size=(spoke_size, spoke_size, grid_z)).to(torch.complex64).to(device)
    DC_loss = torch.nn.L1Loss().to(device)
    TV_loss_function = utils.MYTVLoss().to(device)

    network = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=2, encoding_config=config["encoding"],
                                            network_config=config["mlp"]).to(device)
    optimizer_network = torch.optim.Adam(params=network.parameters(), lr=lr)
    scheduler_network = lr_scheduler.StepLR(optimizer_network, step_size=lr_decay_epoch, gamma=lr_decay_rate)

    # csm
    network_csm = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=2*nChannel, 
                                              encoding_config=config["encoding_csm"], network_config=config["mlp_csm"]).to(device)
    optimizer_network_csm = torch.optim.Adam(params=network_csm.parameters(), lr=lr)
    scheduler_network_csm = lr_scheduler.StepLR(optimizer_network_csm, step_size=lr_decay_epoch, gamma=lr_decay_rate)

    # initial pose
    rot_motion_x = np.zeros(shape=(stage_num, 1, 1), dtype=float)
    rot_motion_y = np.zeros(shape=(stage_num, 1, 1), dtype=float)
    rot_motion_z = np.zeros(shape=(stage_num, 1, 1), dtype=float)
    offset = np.zeros(shape=(stage_num, 1, 3), dtype=float)

    pose_refiner = utils.PoseRefiner(rot_motion_x=rot_motion_x, rot_motion_y=rot_motion_y, rot_motion_z=rot_motion_z, offset=offset, 
                                           rot_is_train=True, offset_is_train=True).to(device)
    optimizer_pose = torch.optim.Adam([{'params': pose_refiner.rot_motion_x, 'lr': lr * 5},
                                       {'params': pose_refiner.rot_motion_y, 'lr': lr * 5},
                                       {'params': pose_refiner.rot_motion_z, 'lr': lr * 5},
                                       {'params': pose_refiner.offset, 'lr': lr}])
    scheduler_pose = lr_scheduler.StepLR(optimizer_pose, step_size=lr_decay_epoch, gamma=lr_decay_rate)

    for e in range(epoch):
        network.train()
        network_csm.train()
        pose_refiner.train()
        loss_log = 0
        epoch_loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
        epoch_loop.set_description("[Train] [Epoch {}/{}] [Lr:{}]".format(e + 1, epoch, scheduler_network.get_last_lr()[0]))
        for i, (stage_id, ray, kdata, ktraj) in epoch_loop:  # 从train_loader中采样得到的ray已经旋转过角度
            ray = ray.to(device).float()  # [batch_size,x,y,z,3]
            kdata = kdata.to(device).to(torch.complex64)  # [batch_size, ch, plane_per_stage, spoke_size, grid_z]
            ktraj = ktraj.to(device).float().reshape(-1,3,plane_per_stage*spoke_size*grid_z)  # (batch_size, 3, plane_per_stage*spoke_size*grid_z)
            stage_id = np.array(stage_id)

            # smap
            tstCsm_tensor = network_csm(ray.view(-1, 3)).float().view(-1, spoke_size//2, spoke_size//2, grid_z, 2*nChannel)
            tstCsm_tensor = torch.complex(tstCsm_tensor[:,:,:,:,:nChannel],tstCsm_tensor[:,:,:,:,nChannel:])  #实部，虚部
            tstCsm_tensor_norm = torch.sqrt(torch.sum(tstCsm_tensor.conj()*tstCsm_tensor,dim=-1)).unsqueeze(-1)
            tstCsm_tensor = tstCsm_tensor / (tstCsm_tensor_norm+1e-8) # [batch_size, x, y, z, ch]
            # tstCsm_tensor = tstCsm_tensor / torch.tile((tstCsm_tensor[:,:,:,0]/torch.abs(tstCsm_tensor[:,:,:,0])).unsqueeze(-1),(1,1,1,nChannel)) # 除第一个coil的相位
            tstCsm_tensor = tstCsm_tensor.permute(0,4,1,2,3) # [batch_size,ch,x,y,z]

            # pose correction
            ray = pose_refiner(stage_id, ray).view(-1, 3)  # [batch_size*x*y*z,3]
            intensity_pre = network(ray).float().view(-1, spoke_size//2, spoke_size//2, grid_z, 2)
            intensity_pre = torch.complex(intensity_pre[:,:,:,:,0],intensity_pre[:,:,:,:,1]).reshape(-1,1,spoke_size//2,spoke_size//2, grid_z) # [batch_size,1,x,y,z]

            # forward
            kdata_pre = nufft_ob(intensity_pre*tstCsm_tensor, ktraj).reshape(-1, nChannel, plane_per_stage, spoke_size, grid_z)
            if torch.isnan(intensity_pre).any():
                print('intensity_pre has nan')
            if torch.isnan(tstCsm_tensor).any():
                print('tstCsm_tensor has nan')
            if torch.isnan(kdata_pre).any():
                print('kdata_pre has nan')
            # compute loss
            dc_loss = DC_loss(torch.view_as_real(kdata_pre).float(), torch.view_as_real(kdata).float())
            img_pre = network(rays.view(-1,3)).reshape(spoke_size//2, spoke_size//2, grid_z, 2)
            img_pre = torch.complex(img_pre[:, :, :, 0], img_pre[:, :, :, 1])
            TV_loss = TV_loss_function(img_pre.real) + TV_loss_function(img_pre.imag)
            if e<=10:
                loss = dc_loss
            else:
                loss = dc_loss + 0.4*TV_loss
            # backward
            optimizer_network.zero_grad()
            optimizer_network_csm.zero_grad()
            optimizer_pose.zero_grad()
            loss.backward()
            optimizer_network.step()
            optimizer_network_csm.step()
            optimizer_pose.step()
            # record and print loss
            loss_log += loss.item()
            epoch_loop.set_postfix(loss='{:.4f}'.format(loss_log/(i+1)))

        scheduler_network.step()
        scheduler_network_csm.step()
        scheduler_pose.step()
        # model save & reconstruction
        if (e + 1) % save_epoch == 0:
            with torch.no_grad():
                network.eval()
                network_csm.eval()
                pose_refiner.eval()
                img_pre = []
                # torch.save(network.state_dict(), '{}/model_{}.pkl'.format(model_path, rot))
                for i, (ray) in enumerate(test_loader):
                    ray = ray.to(device).float().view(-1, 3)  # (x*y, 3)
                    # forward
                    temp_pre = network(ray).float().view(spoke_size//2, spoke_size//2, 2)
                    temp_pre = torch.complex(temp_pre[:,:,0],temp_pre[:,:,1]).cpu().detach().numpy()
                    img_pre.append(temp_pre)
                img_pre = np.array(img_pre).transpose(1,2,0)
                print(img_pre.shape)
                csm_pre = []
                for i, (ray) in enumerate(test_loader):
                    ray = ray.to(device).float().view(-1, 3)  # (x*y, 3)
                    # forward
                    temp_pre = network_csm(ray).float().view(spoke_size//2, spoke_size//2, 2*nChannel)
                    temp_pre = torch.complex(temp_pre[:,:,:nChannel],temp_pre[:,:,nChannel:])  #实部，虚部
                    temp_pre_norm = torch.sqrt(torch.sum(temp_pre.conj()*temp_pre,dim=-1)).unsqueeze(-1)
                    temp_pre = temp_pre / (temp_pre_norm+1e-8) # [x, y, ch]
                    # temp_pre = temp_pre / torch.tile((temp_pre[:,:,0]/torch.abs(temp_pre[:,:,0])).unsqueeze(-1),(1,1,nChannel))
                    temp_pre = temp_pre.cpu().detach().numpy()
                    csm_pre.append(temp_pre)
                csm_pre = np.array(csm_pre).transpose(1,2,0,3)
                print(csm_pre.shape)
                # save
                rot_motion_x_corrected = pose_refiner.rot_motion_x.view(-1, 1).float().cpu().detach().numpy()
                rot_motion_y_corrected = pose_refiner.rot_motion_y.view(-1, 1).float().cpu().detach().numpy()
                rot_motion_z_corrected = pose_refiner.rot_motion_z.view(-1, 1).float().cpu().detach().numpy()
                x_corrected = pose_refiner.offset.view(-1, 3)[:, 0].float().cpu().detach().numpy() * (spoke_size//4) * voxel_size[0]
                y_corrected = pose_refiner.offset.view(-1, 3)[:, 1].float().cpu().detach().numpy() * (spoke_size//4) * voxel_size[1]
                z_corrected = pose_refiner.offset.view(-1, 3)[:, 2].float().cpu().detach().numpy() * (grid_z//2) * voxel_size[2]
                np.savetxt('{}/rot_motion_x_est_{}.txt'.format(out_dir, plane_num), rot_motion_x_corrected)
                np.savetxt('{}/rot_motion_y_est_{}.txt'.format(out_dir, plane_num), rot_motion_y_corrected)
                np.savetxt('{}/rot_motion_z_est_{}.txt'.format(out_dir, plane_num), rot_motion_z_corrected)
                np.savetxt('{}/shift_x_est_{}.txt'.format(out_dir, plane_num), x_corrected)
                np.savetxt('{}/shift_y_est_{}.txt'.format(out_dir, plane_num), y_corrected)
                np.savetxt('{}/shift_z_est_{}.txt'.format(out_dir, plane_num), z_corrected)
                utils.save_nii(np.abs(img_pre), '{}/recon_mag_{}.nii'.format(out_dir, plane_num), voxel_size=voxel_size)
                utils.save_nii(np.angle(img_pre), '{}/recon_phase_{}.nii'.format(out_dir, plane_num), voxel_size=voxel_size)
                utils.save_nii(np.abs(csm_pre), '{}/csm_mag_{}.nii'.format(out_dir, plane_num), voxel_size=voxel_size)
                utils.save_nii(np.angle(csm_pre), '{}/csm_phase_{}.nii'.format(out_dir, plane_num), voxel_size=voxel_size)