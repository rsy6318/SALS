import numpy as np
import os
import torch
import torch.utils.data as data
from tqdm import tqdm

class mesh_pc_dataset(data.Dataset):
    def __init__(self,data_path,len_data ,num_points,num_samples,max_noise_std=0.01,):
        super(mesh_pc_dataset,self).__init__()
        self.data_path=data_path
        self.len_data=len_data
        self.num_points=num_points
        self.num_samples=num_samples
        self.max_noise_std=max_noise_std

        filename_list=os.listdir(data_path)

        np.random.seed(1024)
        #index=np.random.permutation(len(filename_list))[:self.len_data]

        '''print(len(filename_list))
        assert False'''

        filename_list=np.random.choice(filename_list,self.len_data, replace=False)  ####filename_list[index]

        self.all_data=[]

        for filename in tqdm(filename_list):
            self.all_data.append(np.load(os.path.join(data_path,filename,'data.npz')))

    def __len__(self):
        return self.len_data
    
    def __getitem__(self, index):

        data=self.all_data[index]
        points=data['points']
        v1=data['v1']
        v2=data['v2']
        o=data['o']
        s=data['s']

        pc_index=np.random.permutation(points.shape[0])[0:self.num_points]

        points=points[pc_index]         #points[:self.num_points] #

        if self.max_noise_std:
            noise=np.random.randn(self.num_points,3)*np.random.rand()*self.max_noise_std
            points=points+noise

        imp_index=np.random.permutation(v1.shape[0])[0:self.num_samples]

        v1=v1[imp_index]
        v2=v2[imp_index]
        o=o[imp_index]
        s=s[imp_index]

        return {
            'v1':torch.from_numpy(v1).float(),
            'v2':torch.from_numpy(v2).float(),
            'o':torch.from_numpy(o).float(),
            's':torch.from_numpy(s).float(),
            'points':torch.from_numpy(points).float()
        }
    
if __name__=='__main__':
    dataset=mesh_pc_dataset('shapenet_xu_ori_processed',200,10000,2048,0.01)

    data=dataset[70]
    points=data['points']
    points=points.cpu().numpy()
    np.savetxt('points.xyz',points)