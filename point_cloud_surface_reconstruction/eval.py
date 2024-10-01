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
#from torch.utils.data import Dataset,DataLoader
#from dataset import mesh_pc_dataset
from diff_emc import extract_mesh
import pytorch3d.ops

args=get_config().parse_args()

test_data_path= 'test_data_abc2'            #os.path.join(args.ckpt_path,'train_points')  #'logs/09_02_16_19_53/train_points'

save_path='rec2_result'

#test_data_filename_list=os.listdir(test_data_path)

ff=open('abc2_list.txt','r')
mesh_list=ff.readlines()
test_data_filename_list=[meshname.replace('\n','')+'.xyz' for meshname in mesh_list]


net=Implicit_Segment(args).cuda()
net.load_state_dict(torch.load(os.path.join(args.ckpt_path,'model_final.pt')))

net.eval()

for test_data_filename in test_data_filename_list:

    if os.path.exists(os.path.join(args.ckpt_path,save_path,test_data_filename[:-4],'occ_0.500.obj')):
        continue

    points=np.loadtxt(os.path.join(test_data_path,test_data_filename))
    points=torch.from_numpy(points).float().cuda()

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
    #v1.requires_grad=True
    #v2.requires_grad=True
    #print(v1)

    mid_point=(v1+v2)/2

    def cal_nn_dist(input,pointcloud):
        #input (B,3)
        #pointcloud (N,3)
        with torch.no_grad():
            all_nn_dist=[]
            input_set=input.split(2**16)
            for data in input_set:
                dists,_,_=pytorch3d.ops.knn_points(data.unsqueeze(0),pointcloud.unsqueeze(0),K=1,return_nn=True,return_sorted=False)
                dists=dists.squeeze(2).squeeze(0)   #(M)
                all_nn_dist.append(torch.sqrt(dists))
        all_nn_dist=torch.cat(all_nn_dist,dim=0)
        return all_nn_dist


    all_midpoint_dist=cal_nn_dist(mid_point.cuda(),points)

    near_surf_thres=2/127*4

    near_surf_mask=(all_midpoint_dist<near_surf_thres)
    #print(all_midpoint_dist.size())
    #print(near_surf_mask.size())
    

    v1_near=v1[near_surf_mask]
    v2_near=v2[near_surf_mask]


    v1_near_set=v1_near.split(5000)
    v2_near_set=v2_near.split(5000)

    #v1_set=v1.split(15000)  #    chunk(500)
    #v2_set=v2.split(15000)  #    chunk(500)

    pred_o_set_near=[]
    pred_s_set_near=[]
    pred_n_set_near=[]

    for v1,v2 in zip(v1_near_set,v2_near_set):
        v1=v1.cuda()    #.unsqueeze(0)
        v2=v2.cuda()    #.unsqueeze(0)
        v1.requires_grad=True
        v2.requires_grad=True
        output=net(torch.cat((v1,v2),dim=-1).unsqueeze(0),points.unsqueeze(0))
        pred_o=output[0,:,0]
        #pred_s=torch.clamp(output[:,1],0,1)
        pred_s=output[0,:,1]
        pred_o_set_near.append(pred_o)
        pred_s_set_near.append(pred_s)
        pred_s.sum().backward()
        pred_n1=v1.grad.detach()
        pred_n1=pred_n1/(torch.sqrt(torch.sum(pred_n1**2,dim=-1,keepdim=True))+1e-10)
        pred_n2=v2.grad.detach()
        pred_n2=pred_n2/(torch.sqrt(torch.sum(pred_n2**2,dim=-1,keepdim=True))+1e-10)
        pred_n=torch.where(pred_s.unsqueeze(-1).repeat(1,3)>0.5,pred_n1,pred_n2)
        pred_n_set_near.append(pred_n)
        #print(pred_n.size())

    pred_o_near=torch.cat(pred_o_set_near,dim=0)
    pred_s_near=torch.cat(pred_s_set_near,dim=0)
    pred_n_near=torch.cat(pred_n_set_near,dim=0)
    
    pred_o=torch.zeros(unique_edges.size(0),).to(v1)
    pred_s=torch.zeros(unique_edges.size(0),).to(v1)
    pred_n=torch.zeros(unique_edges.size(0),3).to(v1)

    #print(pred_o.size())

    #assert False

    pred_o[near_surf_mask]=pred_o_near
    pred_s[near_surf_mask]=pred_s_near
    pred_n[near_surf_mask]=pred_n_near

    qef_threshold=0.1

    os.makedirs(os.path.join(args.ckpt_path,save_path,test_data_filename[:-4]),exist_ok=True)

    for occ_thres in [0.3]:

        vertices,faces=extract_mesh(verts_unique,cubes,cube_edge_index,
                                unique_edges,unique_edges_count,unique_edges_cube_index,
                                (pred_o>occ_thres).float(),pred_s,pred_n,None,False,qef_threshold)

        trimesh_np=trimesh.Trimesh(vertices.detach().cpu().numpy(),faces.detach().cpu().numpy())
        os.makedirs(os.path.join(args.ckpt_path,save_path),exist_ok=True)
        trimesh_np.export(os.path.join(args.ckpt_path,save_path,test_data_filename[:-4]+'.obj'))
        

