import os
#os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
import numpy as np
import pyngpmesh
import trimesh
from tqdm import tqdm
from network import Implicit_Segment
#import argparse
from config_load import get_config, save_config
import time
from torch.utils.data import Dataset,DataLoader
from dataset import mesh_pc_dataset
from diff_emc import extract_mesh

BCE_fn=torch.nn.BCELoss()
args=get_config().parse_args()
if args.time_folder>0:
    args.ckpt_path=os.path.join(args.ckpt_path,time.strftime("%m_%d_%H_%M_%S", time.localtime()))


def log_string(out_str):
    global LOG_FOUT
    LOG_FOUT.write(out_str)
    LOG_FOUT.flush()


os.makedirs(args.ckpt_path,exist_ok=True)

save_config(os.path.join(args.ckpt_path,'config.txt'),args)

global LOG_FOUT
LOG_FOUT = open(os.path.join(args.ckpt_path, 'log.txt'), 'w')
'''LOG_FOUT.write(str(datetime.now()) + '\n')
LOG_FOUT.write(os.path.abspath(__file__) + '\n')
LOG_FOUT.write(str(arg) + '\n')'''


dataset=mesh_pc_dataset(args.data_path,args.len_dataset,args.num_points,args.num_sample,args.max_noise_std)

for i in range(len(dataset)):
    data=dataset[i]
    points=data['points']
    os.makedirs(os.path.join(args.ckpt_path,'train_points'),exist_ok=True)
    np.savetxt(os.path.join(args.ckpt_path,'train_points','%06d.xyz'%i),points[:dataset.num_points].cpu().numpy())

dataloader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True)

net=Implicit_Segment(args).cuda()

if args.optimizer=='ADAM':
    optimizer=torch.optim.Adam(net.parameters(),lr=args.lr)
elif args.optimizer=='ADAMW':
    optimizer=torch.optim.AdamW(net.parameters(),lr=args.lr)
else:
    assert False

if args.lr_schedual=='cos':
    schedualer=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,args.lr_min)
elif args.lr_schedual=='step':
    schedualer=torch.optim.lr_scheduler.StepLR(optimizer,int(args.epochs*0.34),0.1)

net.train()

for epoch in range(1,args.epochs+1):
    for data in tqdm(dataloader,desc='%04d'%epoch):
        optimizer.zero_grad()
        v1=data['v1'].cuda()
        v2=data['v2'].cuda()
        gt_o=data['o'].cuda()#.reshape(-1)
        gt_s=data['s'].cuda()#.reshape(-1)

        points=data['points'].cuda()

        #print(points.size())

        output=net(torch.cat((v1,v2),dim=-1),points)

        pred_o=output[...,0]#.reshape(-1)
        #pred_s=torch.clamp(output[:,1],0,1)
        pred_s=output[...,1]#.reshape(-1)

        assert torch.sum(gt_o)>0

        loss_s=torch.sum(torch.abs(pred_s-gt_s)*gt_o)/torch.sum(gt_o)

        if args.occ_loss=='MAE':
            loss_o=torch.mean(torch.abs(pred_o-gt_o))
        elif args.occ_loss=='BCE':
            loss_o=torch.mean(-gt_o*torch.log(pred_o+1e-10)-(1-gt_o)*torch.log(1-pred_o+1e-10))
        loss=10*loss_s+args.occ_weight*loss_o
        loss.backward()
        optimizer.step()
    if args.schedular:
        schedualer.step()
    print('acc of O-0.8: %0.3f, acc of O-0.5: %0.3f, acc of O-0.1: %0.3f, acc of O-1e-3: %0.3f, '%( torch.mean((gt_o==(pred_o>0.8).float()).float()).item(),
                                                                                                            torch.mean((gt_o==(pred_o>0.5).float()).float()).item(),
                                                                                                            torch.mean((gt_o==(pred_o>0.1).float()).float()).item(),
                                                                                                            torch.mean((gt_o==(pred_o>1e-3).float()).float()).item()
                                                                                                            ),)
    log_string('epoch: %d, acc of O-0.8: %0.3f, acc of O-0.5: %0.3f, acc of O-0.1: %0.3f, acc of O-1e-3: %0.3f, \n'%(epoch, torch.mean((gt_o==(pred_o>0.8).float()).float()).item(),
                                                                                                            torch.mean((gt_o==(pred_o>0.5).float()).float()).item(),
                                                                                                            torch.mean((gt_o==(pred_o>0.1).float()).float()).item(),
                                                                                                            torch.mean((gt_o==(pred_o>1e-3).float()).float()).item()
                                                                                                            ),)
    if epoch%50==0:
        torch.save(net.state_dict(),os.path.join(args.ckpt_path,'model_%d.pt'%epoch))

torch.save(net.state_dict(),os.path.join(args.ckpt_path,'model_final.pt'))

