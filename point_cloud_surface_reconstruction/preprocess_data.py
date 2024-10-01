import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import numpy as np
import torch
import trimesh
import random
import pyngpmesh
from tqdm import tqdm

from multiprocessing import Pool

base_path='path to data'     
save_path='path to save'    
mesh_name_list=os.listdir(base_path)

random.shuffle(mesh_name_list)

num_sample=1000000


idx_list=list(range(len(mesh_name_list)))

#=all_file_path[:len_dataset]

#for filepath in tqdm(all_file_path):
    
def process_one(index):    
    filepath=os.path.join(base_path,mesh_name_list[index])
    mesh=trimesh.load_mesh(filepath)
    v=mesh.vertices

    v_max=np.max(v,axis=0)
    v_min=np.min(v,axis=0)

    center=(v_max+v_min)/2
    scale=np.max(v_max-v_min)
    v=(v-center.reshape(1,3))/scale*1.99
    
    
    mesh.vertices=v
    
    os.makedirs(os.path.join(save_path,mesh_name_list[index][:-4]))

    mesh.export(os.path.join(save_path,mesh_name_list[index][:-4],'scaled_mesh.obj'))
    
    gt_mesh=trimesh.load_mesh(os.path.join(save_path,mesh_name_list[index][:-4],'scaled_mesh.obj'))
    renderer=pyngpmesh.NGPMesh(gt_mesh.triangles)

    surface_sample_scales = [0.005, 0.01, 0.03]    #[0.005, 0.015, 0.02]
    surface_sample_ratios = [0.5,0.3,0.1]        #[0.4, 0.3, 0.1]    #[0.5, 0.3, 0.1]        # sum: 0.9
    bbox_space_sample_ratio=[0.08,0.02]

    bbox_sample_scale, bbox_sample_ratio, bbox_padding = 0.08, 0.08, 0.02
    space_sample_scale, space_sample_ratio, space_size = 0.08, 0.02, 1

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


    sample_pairs = np.concatenate([surface_pairs, bbox_pairs, space_pairs], axis=0)    #bbox_pairs,

    v1=sample_pairs[:,0:3]
    v2=sample_pairs[:,3:]
    dir=v2-v1
    dir=dir/np.sqrt(np.sum(dir**2,axis=-1,keepdims=True))   #(N,3)
    p=np.array(renderer.trace(v1,dir))
    ratio=np.sum((p-v1)*dir,axis=-1)/np.sum((v2-v1)*dir,axis=-1)
    #ratio=np.max(np.abs(p-v1),axis=-1)/np.max(np.abs(v2-v1),axis=-1)
    all_o=(ratio<=1) & (ratio>=0)
    all_s=1-ratio
    all_s=np.clip(all_s,0,1)    #np.sigmoid(all_s)

    points,faces=gt_mesh.sample(1000000,return_index=True)
    normals=gt_mesh.face_normals[faces]

    np.savez(os.path.join(save_path,mesh_name_list[index][:-4],'data.npz'),points=points,normals=normals,v1=v1.astype(np.float32),v2=v2.astype(np.float32),o=all_o.astype(np.float32),s=all_s.astype(np.float32))


def multiprocess(func):
    p = Pool(10)
    p.map(func, idx_list)
    p.close()
    p.join()

if __name__=='__main__':
    multiprocess(process_one)