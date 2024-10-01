import os
import torch
from tqdm import tqdm


cube_corners = torch.tensor([[0, 0, 0], 
                             [1, 0, 0], 
                             [0, 1, 0], 
                             [1, 1, 0], 
                             [0, 0, 1], 
                             [1, 0, 1], 
                             [0, 1, 1], 
                             [1, 1, 1]], dtype=torch.float, )

cube_edges=torch.tensor([0,1,
                         1,5,
                         4,5,
                         0,4,
                         2,3,
                         3,7,
                         6,7,
                         2,6,
                         0,2,
                         1,3,
                         5,7,
                         4,6], dtype=torch.float).long()

def vector_pad(input):
    #pad vector less than 4 with -1 to make their length become 4
    if input.size(0)==4:
        return input
    else:
        input_len=input.size(0)
        output=(torch.ones(4,)*(-1)).to(input)
        output[:input_len]=input
        return output


def check_matrix_inv(input,threshold=1e-3):
    #input: (*,3,3)
    #L,V=torch.linalg.eig(input)
    #L=L.real
    #V=V.real
    L,V=torch.linalg.eigh(input)
    V_inv=V.transpose(-1,-2)
    L_inv=torch.where(L<threshold,torch.zeros_like(L),1/L)
    L_inv_matrix=torch.diag_embed(L_inv)
    result=V@L_inv_matrix@V_inv
    return result

def my_construct_voxel_grid(res: int,device:str='cuda'):
    base_cube_f=torch.arange(8)
    voxel_grid_template=torch.ones((res,res,res)).to(device)
    coords=torch.nonzero(voxel_grid_template).long()    #N,3
    
    verts=(cube_corners.long().unsqueeze(0).to(device)+coords.unsqueeze(1)).reshape(-1,3)         # (1,8,3) + (N,1,3) -> (N,8,3) -> (N*8,3)
    cubes=(base_cube_f.unsqueeze(0)+
           torch.arange(coords.shape[0]).unsqueeze(1)*8)

    verts_unique,inverse_indices=torch.unique(verts,dim=0,return_inverse=True)
    cubes=inverse_indices[cubes.reshape(-1)].reshape(-1,8)
    verts_unique=verts_unique.float()/(res)*2-1

    cubes_index=torch.arange(cubes.size(0)).to(device).long()
    num_cubes=cubes.size(0)
    all_edges=cubes[:,cube_edges].reshape(-1,2)
    unique_edges,unique_edges_map,unique_edges_count=torch.unique(all_edges,dim=0,return_inverse=True,return_counts=True)
    num_unique_edges=unique_edges.size(0)
    unique_edges_idx=torch.arange(unique_edges.size(0)).long()  #(num_unique_edges)

    all_edges_idx=unique_edges_idx[unique_edges_map]
    all_edges_count=unique_edges_count[unique_edges_map]

    all_edges_idx_reshapes=all_edges_idx.reshape(-1,12).to(device)     #(num_cubes, 12)

    print(all_edges_idx_reshapes.size())
    print(cubes.size())

    #unique_edges_cube_index=[]
    #import edge_cube_index
    #import cube_edge_index

    unique_edges_cube_index=[]

    for i in tqdm(range(num_unique_edges),desc='constructing edge_cube_index'):
        quad_index=[0,1,3,2]
        this_edge_cube_index=cubes_index[torch.sum((all_edges_idx_reshapes==i).float(),dim=-1)>0]
        this_edge_cube_index=vector_pad(this_edge_cube_index)[quad_index]

        #print(this_edge_cube_index)
        
        unique_edges_cube_index.append(this_edge_cube_index.long().to(device))
    
    unique_edges_cube_index=torch.stack(unique_edges_cube_index,dim=0)

    
    cube_edge_index=[]

    for i in tqdm(range(cubes.size(0)),desc='constructing cube_edge_index'):
        for j in range(12):
            this_edge_index=unique_edges_idx[(unique_edges==all_edges.reshape(-1,12,2)[i,j].unsqueeze(0)).float().sum(dim=-1)==2]
            cube_edge_index.append(this_edge_index.reshape(-1))
    cube_edge_index=torch.tensor(cube_edge_index).to(all_edges).long().reshape(-1,12)

    
    print(cube_edge_index.size())

    #return 2*(verts_unique.to(device) - 0.5), cubes.to(device), num_unique_edges
    return {
        'verts_unique':verts_unique.to(device),
        'cubes':cubes.to(device),
        'cubes_index':cubes_index.to(device),
        'num_cubes':num_cubes,
        'cubes_edge_index':cube_edge_index,

        'num_unique_edges':num_unique_edges,
        'unique_edges':unique_edges.to(device),
        'unique_edges_map':unique_edges_map.to(device),
        'unique_edges_idx':unique_edges_idx.to(device),
        'unique_edges_count':unique_edges_count.to(device),
        'unique_edges_cube_index':unique_edges_cube_index.to(device),

        'all_edges':all_edges.to(device).long(),
        'all_edges_idx':all_edges_idx.to(device).long(),
        'all_edges_count':all_edges_count.to(device).long()
    }

def construct_voxel_grid(res: int,device:str='cuda'):
    """
    res: resolution of cubes (not grid points !!!)
    return: grid points in [-1,1], cube indes [numcubes, 8], num_edges
    """
    
    base_cube_f = torch.arange(8)

    #num_unique_edges=(res+1)**3*3-(res+1)**2*3

    if isinstance(res, int):
        res = (res, res, res)

    voxel_grid_template = torch.ones(res).to(device)
    res = torch.tensor([res], dtype=torch.float).to(device)
    coords = torch.nonzero(voxel_grid_template).float() / res  # N, 3

    verts = (cube_corners.unsqueeze(0).to(device) / res + coords.unsqueeze(1)).reshape(-1, 3)
    cubes = (base_cube_f.unsqueeze(0) +
                 torch.arange(coords.shape[0]).unsqueeze(1) * 8).reshape(-1)

    verts_rounded = torch.round(verts * 10**5) / (10**5)
    verts_unique, inverse_indices = torch.unique(verts_rounded, dim=0, return_inverse=True)
    cubes = inverse_indices[cubes.reshape(-1)].reshape(-1, 8)

    cubes_index=torch.arange(cubes.size(0)).to(device).long()
    num_cubes=cubes.size(0)

    all_edges=cubes[:,cube_edges].reshape(-1,2)
    unique_edges,unique_edges_map,unique_edges_count=torch.unique(all_edges,dim=0,return_inverse=True,return_counts=True)
    num_unique_edges=unique_edges.size(0)

    # unique_edges -> (num_unique_edges,2)

    unique_edges_idx=torch.arange(unique_edges.size(0)).long()  #(num_unique_edges)
    
    all_edges_idx=unique_edges_idx[unique_edges_map]
    all_edges_count=unique_edges_count[unique_edges_map]


    all_edges_idx_reshapes=all_edges_idx.reshape(-1,12).to(device)     #(num_cubes, 12)

    print(all_edges_idx_reshapes.size())
    print(cubes.size())

    unique_edges_cube_index=[]

    for i in tqdm(range(num_unique_edges),desc='constructing edge_cube_index'):
        quad_index=[0,1,3,2]
        this_edge_cube_index=cubes_index[torch.sum((all_edges_idx_reshapes==i).float(),dim=-1)>0]
        this_edge_cube_index=vector_pad(this_edge_cube_index)[quad_index]

        #print(this_edge_cube_index)

        unique_edges_cube_index.append(this_edge_cube_index.long().to(device))
    
    unique_edges_cube_index=torch.stack(unique_edges_cube_index,dim=0)

    cube_edge_index=[]

    for i in tqdm(range(cubes.size(0)),desc='constructing cube_edge_index'):
        for j in range(12):
            #print(unique_edges.size())
            #print(all_edges.reshape(-1,12,2)[i,j].unsqueeze(0).size())
            #print((unique_edges==all_edges.reshape(-1,12,2)[i,j].unsqueeze(0)).float().sum(dim=-1)==2)
            #assert False
            this_edge_index=unique_edges_idx[(unique_edges==all_edges.reshape(-1,12,2)[i,j].unsqueeze(0)).float().sum(dim=-1)==2]
            cube_edge_index.append(this_edge_index.reshape(-1))
    cube_edge_index=torch.tensor(cube_edge_index).to(all_edges).long().reshape(-1,12)

    print(cube_edge_index.size())

    #return 2*(verts_unique.to(device) - 0.5), cubes.to(device), num_unique_edges
    return {
        'verts_unique':2*(verts_unique.to(device)-0.5),
        'cubes':cubes.to(device),
        'cubes_index':cubes_index.to(device),
        'num_cubes':num_cubes,
        'cubes_edge_index':cube_edge_index,

        'num_unique_edges':num_unique_edges,
        'unique_edges':unique_edges.to(device),
        'unique_edges_map':unique_edges_map.to(device),
        'unique_edges_idx':unique_edges_idx.to(device),
        'unique_edges_count':unique_edges_count.to(device),
        'unique_edges_cube_index':unique_edges_cube_index.to(device),

        'all_edges':all_edges.to(device).long(),
        'all_edges_idx':all_edges_idx.to(device).long(),
        'all_edges_count':all_edges_count.to(device).long()
    }



def solve_qef(sel_edge_o,sel_edge_p,sel_edge_n,sel_vertices,inv_thres=1e-3):
    #sel_edge_o:        (N,12)
    #sel_edge_p:    (N,12,3)
    #sel_edge_n:    (N,12,3)
    #sel_vertices:  (N,8,3)
    #qef_reg_scale=1e-3

    #v0=sel_vertices[:,0,:]  #(N,3)
    v0=(sel_edge_p*sel_edge_o.unsqueeze(-1)).sum(1)/sel_edge_o.unsqueeze(-1).sum(1)   #(N,3)
    #return center_p
    
    #center_p_v0=center_p-v0             #(N,3)
    sel_edge_p_v0=sel_edge_p-v0.unsqueeze(1)    #(N,12,3)

    A=sel_edge_o.unsqueeze(-1)*sel_edge_n       #(N,12,3)
    #B=sel_edge_o.unsqueeze(-1)*torch.sum(sel_edge_n*sel_edge_p,dim=-1,keepdim=True)  #(N,12,1)
    B=sel_edge_o.unsqueeze(-1)*torch.sum(sel_edge_n*sel_edge_p_v0,dim=-1,keepdim=True)  #(N,12,1)
    
    ATA=torch.matmul(A.transpose(-2,-1),A)   #(N,3,3)
    ATB=torch.matmul(A.transpose(-2,-1),B)   #(N,3,1)
    
    ATA_inv=check_matrix_inv(ATA,inv_thres)
    #ATA_inv=torch.pinverse(ATA)
    #ATA_inv=torch.inverse(ATA)
    
    x=torch.matmul(ATA_inv,ATB).squeeze(-1)
    x=x+v0

    #x=diff_clamp(x,sel_vertices[:,0,:],sel_vertices[:,-1,:])

    #print(sel_vertices[0,0,:],sel_vertices[0,-1,:])

    return x


def solve_qef_no_normal(sel_edge_o,sel_edge_p,sel_edge_n,sel_vertices,inv_thres=1e-3):
    #sel_edge_o:        (N,12)
    #sel_edge_p:    (N,12,3)
    #sel_edge_n:    (N,12,3)
    #sel_vertices:  (N,8,3)
    #qef_reg_scale=1e-3

    #v0=sel_vertices[:,0,:]  #(N,3)
    v0=(sel_edge_p*sel_edge_o.unsqueeze(-1)).sum(1)/sel_edge_o.unsqueeze(-1).sum(1)   #(N,3)
    #return center_p
    
    #center_p_v0=center_p-v0             #(N,3)
    sel_edge_p_v0=sel_edge_p-v0.unsqueeze(1)    #(N,12,3)

    A=sel_edge_o.unsqueeze(-1)*sel_edge_n       #(N,12,3)
    #B=sel_edge_o.unsqueeze(-1)*torch.sum(sel_edge_n*sel_edge_p,dim=-1,keepdim=True)  #(N,12,1)
    B=sel_edge_o.unsqueeze(-1)*torch.sum(sel_edge_n*sel_edge_p_v0,dim=-1,keepdim=True)  #(N,12,1)
    
    ATA=torch.matmul(A.transpose(-2,-1),A)   #(N,3,3)
    ATB=torch.matmul(A.transpose(-2,-1),B)   #(N,3,1)
    
    ATA_inv=check_matrix_inv(ATA,inv_thres)
    #ATA_inv=torch.pinverse(ATA)
    #ATA_inv=torch.inverse(ATA)
    
    x=torch.matmul(ATA_inv,ATB).squeeze(-1)
    x=x+v0

    #x=diff_clamp(x,sel_vertices[:,0,:],sel_vertices[:,-1,:])

    #print(sel_vertices[0,0,:],sel_vertices[0,-1,:])

    return v0


def extract_mesh(verts,cubes,cube_edge_index,
                 unique_edges,unique_edges_count,unique_edges_cube_index,
                 #unique_edges_z,unique_edges_w):
                 unique_edges_o,unique_edges_s,unique_edges_n,unique_edges_w=None,return_tet=False,inv_thres=1e-3):
    #verts: grid coords
    #cubes_fx8: cube vertex index
    #unique_edges: (num_unique_edges,2)

    #unique_edges_o=diff_mask(unique_edges_z[:,0])               #(num_unique_edges,)
    #unique_edges_s=torch.sigmoid(unique_edges_z[:,1])           #(num_unique_edges,)
    #unique_edges_n=unique_edges_z[:,2:] /torch.sqrt(1e-10+torch.sum(unique_edges_z[:,2:]**2,dim=-1,keepdim=True))   #(num_unique_edges,3)
    #unique_edges_w=torch.softmax(unique_edges_w,dim=-1)         #(num_unique_edges,4)

    unique_edges_verts=verts[unique_edges.reshape(-1)].reshape(-1,2,3)
    unique_edges_intersection=unique_edges_verts[:,0,:]*unique_edges_s.unsqueeze(-1)+unique_edges_verts[:,1,:]*(1-unique_edges_s.unsqueeze(-1))
    
    sel_unique_edges_mask=(unique_edges_o>0.5)&(unique_edges_count==4)
    sel_unique_edges_cube_index=unique_edges_cube_index[sel_unique_edges_mask]  #(*, 4)
    sel_unique_edges_cube_index_flatten=sel_unique_edges_cube_index.reshape(-1) #(*,)           
    
    #the cubes of each selected edges (occ&count==4)
    unique_sel_unique_edges_cube_index_flatten,unique_sel_unique_edges_cube_index_map=torch.unique(sel_unique_edges_cube_index_flatten,dim=0,return_inverse=True)

    unique_sel_cube_edge_index=cube_edge_index[unique_sel_unique_edges_cube_index_flatten]            #(*,12)      edge index

    unique_sel_all_edges_o=unique_edges_o[unique_sel_cube_edge_index.reshape(-1)].reshape(-1,12)
    unique_sel_all_edges_n=unique_edges_n[unique_sel_cube_edge_index.reshape(-1)].reshape(-1,12,3)
    unique_sel_all_edges_intersection=unique_edges_intersection[unique_sel_cube_edge_index.reshape(-1)].reshape(-1,12,3)
    unique_sel_all_verts=verts[cubes[unique_sel_unique_edges_cube_index_flatten].reshape(-1)].reshape(-1,8,3)
    
    assert unique_sel_all_edges_intersection.size(0)==unique_sel_all_verts.size(0)

    unique_sel_x=solve_qef(unique_sel_all_edges_o,unique_sel_all_edges_intersection,unique_sel_all_edges_n,unique_sel_all_verts,inv_thres=inv_thres)    #vertices

    unique_sel_x_index=torch.arange(unique_sel_x.size(0))

    all_sel_x=unique_sel_x[unique_sel_unique_edges_cube_index_map].reshape(-1,4,3)
    all_sel_x_index=unique_sel_x_index[unique_sel_unique_edges_cube_index_map].reshape(-1,4)    #tet index
    
    if return_tet:
        return unique_sel_x,all_sel_x_index.to(unique_sel_x).long()
    
    #all_sel_w=unique_edges_w[sel_unique_edges_mask]     #.reshape(-1,4)

    #new_x=torch.sum(all_sel_x*all_sel_w.unsqueeze(-1),dim=1)    #(num_new_x, 3)
    new_x=unique_edges_intersection[sel_unique_edges_mask]

    new_x_index=torch.arange(new_x.size(0)).unsqueeze(-1)+unique_sel_x.size(0)
    vertices=torch.cat((unique_sel_x,new_x),dim=0)
    faces=torch.cat((all_sel_x_index[:,0:2],new_x_index,all_sel_x_index[:,1:3],new_x_index,all_sel_x_index[:,2:],new_x_index,all_sel_x_index[:,3:],all_sel_x_index[:,0:1],new_x_index,),dim=1).reshape(-1,3)


    return vertices,faces.to(vertices).long()








if __name__=='__main__':
    

    cube_res=127

    output_dict=my_construct_voxel_grid(cube_res)

    torch.save(output_dict, 'cube_res_%d_new.pth'%cube_res)

    