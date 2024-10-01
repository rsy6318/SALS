import os
#os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
import numpy as np
import pyngpmesh
import trimesh
from diff_emc import extract_mesh
from tqdm import tqdm
from network import MLPNet, MLP  #,parse_options
#import argparse
from config_load import get_config, save_config
import time
from torch.utils.data import Dataset,DataLoader

BCE_fn=torch.nn.BCELoss()

args=get_config().parse_args()

if args.time_folder>0:
    args.ckpt_path=os.path.join(args.ckpt_path,time.strftime("%m_%d_%H_%M_%S", time.localtime()))


os.makedirs(args.ckpt_path,exist_ok=True)

save_config(os.path.join(args.ckpt_path,'config.txt'),args)

filename=args.filename

gt_mesh=trimesh.load_mesh(filename)
gt_mesh.export(os.path.join(args.ckpt_path,'gt_mesh.obj'))

num_sample=args.num_sample #4000000

'''ratio_on_surface=0.5
ratio_near_surface=0.3
ratio_far_surface=0.1
ratio_space=0.05
num_on_surface=int(ratio_on_surface*num_sample)
num_near_surface=int(ratio_near_surface*num_sample)
num_far_surface=int(ratio_far_surface*num_sample)
num_space=int(ratio_space*num_sample)

samples1=np.tile(gt_mesh.sample(num_on_surface),(1,2))+np.random.randn(num_on_surface,6)*0.005
samples2=np.tile(gt_mesh.sample(num_near_surface),(1,2))+np.random.randn(num_near_surface,6)*0.015
samples3=np.tile(gt_mesh.sample(num_far_surface),(1,2))+np.random.randn(num_far_surface,6)*0.03
samples4=np.random.rand(num_space,6)*2-1 + np.random.randn(num_space,6)*0.03
sample_pairs=np.concatenate([samples1,samples2,samples3,samples4],axis=0)
'''



surface_sample_scales = [float(scale1) for scale1 in args.surface_sample_scales.split('_')]    #[0.005, 0.015, 0.02]
surface_sample_ratios = [float(ratio1) for ratio1 in args.surface_sample_ratios.split('_')]        #[0.4, 0.3, 0.1]    #[0.5, 0.3, 0.1]        # sum: 0.9
bbox_space_sample_ratio=[float(ratio2) for ratio2 in args.bbox_space_sample_ratio.split('_')]

bbox_sample_scale, bbox_sample_ratio, bbox_padding = args.space_sample_scale, bbox_space_sample_ratio[0], 0.02
space_sample_scale, space_sample_ratio, space_size = args.space_sample_scale, bbox_space_sample_ratio[1], 1

surface_pairs=[]
for sample_ratio, sample_scale in zip(surface_sample_ratios, surface_sample_scales):
    sample_points_num = int(num_sample * sample_ratio)
    surface_points = gt_mesh.sample(sample_points_num) #* 4
    random_pairs = np.tile(surface_points, (1, 2))
    assert random_pairs.shape[1] == 6  # shape: N x 6
    random_pairs = random_pairs + np.random.randn(*random_pairs.shape) * sample_scale
    surface_pairs.append(random_pairs)
surface_pairs = np.concatenate(surface_pairs, axis=0)

bbox_points_num = int(num_sample * bbox_sample_ratio)
extents, transform = trimesh.bounds.to_extents(gt_mesh.bounds)
padding_extents = extents + bbox_padding
bbox_points = trimesh.sample.volume_rectangular(padding_extents, bbox_points_num , transform=transform) #* 6
bbox_pairs = np.tile(bbox_points, (1, 2))
bbox_pairs = bbox_pairs + np.random.randn(*bbox_pairs.shape) * bbox_sample_scale

# sample in space
space_points_num = int(num_sample * space_sample_ratio)
space_points = (np.random.rand(int(space_points_num ), 3) * 2 - 1) * space_size #* 2
space_pairs0 = np.tile(space_points, (1, 2))
space_pairs0 = space_pairs0 + np.random.randn(*space_pairs0.shape) * space_sample_scale
space_pairs=space_pairs0

# sample points in bbox and space
"""extents, transform = trimesh.bounds.to_extents(gt_mesh.bounds)
bbox_points = trimesh.sample.volume_rectangular(extents, space_points_num , transform=transform) #* 2
space_points = (np.random.rand(int(space_points_num ), 3) * 2 - 1) * space_size   #* 2
space_pairs1 = np.concatenate([bbox_points, space_points], axis=1)
space_pairs = np.concatenate([space_pairs0, space_pairs1], axis=0)"""

sample_pairs = np.concatenate([surface_pairs, bbox_pairs, space_pairs], axis=0)    #bbox_pairs,

v1=sample_pairs[:,0:3]
v2=sample_pairs[:,3:]
#v2v1=v2-v1
dir=v2-v1
dir=dir/np.sqrt(np.sum(dir**2,axis=-1,keepdims=True))   #(N,3)

renderer=pyngpmesh.NGPMesh(gt_mesh.triangles)

p=np.array(renderer.trace(v1,dir))

#udf_v1=np.array(renderer.unsigned_distance(v1))
#p[udf_v1==0]=v1[udf_v1==0]

if args.occ_argu>=0:
    udf_v1=np.array(renderer.unsigned_distance(v1))
    udf_v2=np.array(renderer.unsigned_distance(v2))
    p[udf_v1<args.occ_argu]=v1[udf_v1<args.occ_argu]
    p[udf_v2<args.occ_argu]=v2[udf_v2<args.occ_argu]

ratio=np.sum((p-v1)*dir,axis=-1)/np.sum((v2-v1)*dir,axis=-1)

#ratio=np.max(np.abs(p-v1),axis=-1)/np.max(np.abs(v2-v1),axis=-1)
all_o=(ratio<=1) & (ratio>=0)
all_s=1-ratio
all_s=np.clip(all_s,0,1)    #np.sigmoid(all_s)

max_iter= args.max_iter # 50000 #          
batch_size= args.batch_size #50000             #

net=MLP(d=args.d_pe,act=args.mlp_act,o_act=args.o_act,s_act=args.s_act,mlp_dim=args.mlp_dim).cuda()
#net=MLPNet(args).cuda()
#print(net)

if args.optimizer=='ADAM':
    optimizer=torch.optim.Adam(net.parameters(),lr=args.lr)
elif args.optimizer=='ADAMW':
    optimizer=torch.optim.AdamW(net.parameters(),lr=args.lr)
else:
    assert False

if args.lr_schedual=='cos':
    schedualer=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,max_iter,args.lr_min)
elif args.lr_schedual=='step':
    schedualer=torch.optim.lr_scheduler.StepLR(optimizer,int(max_iter*0.34),0.1)


for iter in tqdm(range(max_iter)):
    if 1:
        
        optimizer.zero_grad()
        sel_index=np.random.randint(v1.shape[0],size=batch_size)
        sel_v1=v1[sel_index]
        sel_v2=v2[sel_index]
        sel_o=all_o[sel_index]
        sel_s=all_s[sel_index]



        sel_v1=torch.from_numpy(sel_v1).float().cuda()
        sel_v2=torch.from_numpy(sel_v2).float().cuda()
        sel_o=torch.from_numpy(sel_o).float().cuda()
        sel_s=torch.from_numpy(sel_s).float().cuda()

        output=net(torch.cat((sel_v1,sel_v2),dim=-1))
        pred_o=output[:,0]
        
        pred_s=output[:,1]

        assert torch.sum(sel_o)>0

        loss_s=torch.sum(torch.abs(pred_s-sel_s)*sel_o)/torch.sum(sel_o)
        
        if args.occ_loss=='MAE':
            loss_o=torch.mean(torch.abs(pred_o-sel_o))
            
        elif args.occ_loss=='BCE':
            
            loss_o=torch.mean(-sel_o*torch.log(pred_o+1e-10)-(1-sel_o)*torch.log(1-pred_o+1e-10))
            
        loss_reg=0

        loss=10*loss_s+args.occ_weight*loss_o

        loss.backward()
        optimizer.step()
        schedualer.step()

os.makedirs(args.ckpt_path,exist_ok=True)
torch.save(net.state_dict(),os.path.join(args.ckpt_path,'model.pt'))

loaded_data = torch.load('cube_res_127_new.pth',map_location=torch.device('cpu'))

# 访问加载的数据
verts_unique  = loaded_data['verts_unique']
cubes  = loaded_data['cubes']
cubes_index  = loaded_data['cubes_index']
cube_edge_index = loaded_data['cubes_edge_index']
num_cubes  = loaded_data['num_cubes']
num_unique_edges  = loaded_data['num_unique_edges']
unique_edges  = loaded_data['unique_edges']             #(num_unique_edges,2)
unique_edges_map  = loaded_data['unique_edges_map']
unique_edges_idx  = loaded_data['unique_edges_idx']
unique_edges_count  = loaded_data['unique_edges_count']
unique_edges_cube_index  = loaded_data['unique_edges_cube_index']
all_edges  = loaded_data['all_edges']
all_edges_idx  = loaded_data['all_edges_idx']
all_edges_count  = loaded_data['all_edges_count']

verts_unique=verts_unique.cuda().float()
cubes=cubes.cuda().long()
cube_edge_index=cube_edge_index.cuda().long()
unique_edges=unique_edges.cuda().long()
unique_edges_count=unique_edges_count.cuda().long()
unique_edges_cube_index=unique_edges_cube_index.cuda().long()

unique_edges_verts=verts_unique[unique_edges.reshape(-1)].reshape(-1,2,3).cpu().numpy()

v1=unique_edges_verts[:,0,:]
v2=unique_edges_verts[:,1,:]

v1=torch.from_numpy(v1).float() #.cuda()
v2=torch.from_numpy(v2).float() #.cuda()


v1_set=v1.chunk(50)
v2_set=v2.chunk(50)

pred_o_set=[]
pred_s_set=[]
pred_n_set=[]

for v1,v2 in zip(v1_set,v2_set):
    v1=v1.cuda()
    v2=v2.cuda()
    v1.requires_grad=True
    v2.requires_grad=True
    output=net(torch.cat((v1,v2),dim=-1))
    pred_o=output[:,0]
    #pred_s=torch.clamp(output[:,1],0,1)
    pred_s=output[:,1]
    
    #pred_n2=v1.grad.detach()
    #print(pred_o)
    #print(pred_s.size())
    
    #pred_n=cal_grad(v1,pred_s)
    pred_o_set.append(pred_o)
    pred_s_set.append(pred_s)
    pred_s.sum().backward()
    pred_n1=v1.grad.detach()
    pred_n1=pred_n1/(torch.sqrt(torch.sum(pred_n1**2,dim=-1,keepdim=True))+1e-10)
    pred_n2=v2.grad.detach()
    pred_n2=pred_n2/(torch.sqrt(torch.sum(pred_n2**2,dim=-1,keepdim=True))+1e-10)
    pred_n=torch.where(pred_s.unsqueeze(-1).repeat(1,3)>0.5,pred_n1,pred_n2)
    pred_n_set.append(pred_n)
    #print(pred_n.size())

pred_o=torch.cat(pred_o_set,dim=0)
pred_n=torch.cat(pred_n_set,dim=0)
pred_s=torch.cat(pred_s_set,dim=0)



for occ_threshold in [1e-1]:
    qef_threshold=0.1
    

    vertices,faces=extract_mesh(verts_unique,cubes,cube_edge_index,
                            unique_edges,unique_edges_count,unique_edges_cube_index,
                            (pred_o>occ_threshold).float(),pred_s,pred_n,None,False,qef_threshold)

    trimesh_np=trimesh.Trimesh(vertices.detach().cpu().numpy(),faces.detach().cpu().numpy())
    trimesh_np.export(os.path.join(args.ckpt_path,'result.obj'))