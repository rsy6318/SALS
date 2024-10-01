import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch3d.ops


class Implicit_Segment(nn.Module):
    def __init__(self,args):
        super(Implicit_Segment,self).__init__()
        
        fd=256

        knn=args.knn
        o_act=args.o_act
        s_act=args.s_act
        mlp_dim=args.mlp_dim
        use_dist=args.use_dist
        num_layers=args.num_layers

        use_bn=args.use_bn

        self.knn=knn
        self.use_dist=use_dist
        if self.use_dist:
            input_dim=4
        else:
            input_dim=3
        patch_feature_net1=[nn.Conv2d(input_dim,mlp_dim,1) ]
        if use_bn:
            patch_feature_net1.append(nn.BatchNorm2d(mlp_dim))
        patch_feature_net1.append(nn.LeakyReLU(negative_slope=0.2))
        for _ in range(num_layers):
            patch_feature_net1.append(nn.Conv2d(mlp_dim,mlp_dim,1))
            if use_bn:
                patch_feature_net1.append(nn.BatchNorm2d(mlp_dim))
            patch_feature_net1.append(nn.LeakyReLU(negative_slope=0.2))
        patch_feature_net1.append(nn.Conv2d(mlp_dim,fd,1))
        self.patch_feature_net1=nn.Sequential( *patch_feature_net1  )
        
        patch_feature_net2=[nn.Conv2d(input_dim,mlp_dim,1) ]
        if use_bn:
            patch_feature_net2.append(nn.BatchNorm2d(mlp_dim))
        patch_feature_net2.append(nn.LeakyReLU(negative_slope=0.2))
        for _ in range(num_layers):
            patch_feature_net2.append(nn.Conv2d(mlp_dim,mlp_dim,1))
            if use_bn:
                patch_feature_net2.append(nn.BatchNorm2d(mlp_dim))
            patch_feature_net2.append(nn.LeakyReLU(negative_slope=0.2))
        patch_feature_net2.append(nn.Conv2d(mlp_dim,fd,1))
        self.patch_feature_net2=nn.Sequential( *patch_feature_net2  )


        regress_net=[nn.Conv1d(fd+fd,mlp_dim,1) ]
        if use_bn:
            regress_net.append(nn.BatchNorm1d(mlp_dim))
        regress_net.append(nn.LeakyReLU(negative_slope=0.2),)
        for _ in range(num_layers):
            regress_net.append(nn.Conv1d(mlp_dim,mlp_dim,1))
            if use_bn:
                regress_net.append(nn.BatchNorm1d(mlp_dim))
            regress_net.append(nn.LeakyReLU(negative_slope=0.2))
        regress_net.append(nn.Conv1d(mlp_dim,2,1))
        self.regress_net=nn.Sequential(*regress_net)


        assert o_act in ['sigmoid','clamp','none','relu']
        if o_act=='sigmoid':
            self.o_act=F.sigmoid
        elif o_act=='clamp':
            self.o_act=lambda x: torch.clamp(x,0,1)
        elif o_act=='relu':
            self.o_act=F.relu
        elif o_act=='none':
            self.o_act=lambda x: x
        else:
            print('no o act !!!')
            assert False

        assert s_act in ['sigmoid','clamp']
        if s_act=='sigmoid':
            self.s_act=F.sigmoid
        elif s_act=='clamp':
            self.s_act=lambda x: torch.clamp(x,0,1)
        else:
            print('no s act !!!')
            assert False

    def forward(self,input,points):
        #input:     B,M,6
        #points:    B,N,3
        v1=input[:,:,:3]    #(B,M,3)
        v2=input[:,:,3:]    #(B,M,3)
        mid=(v1+v2)/2       #(B,M,3)


        #(B,M,K)   (B,M,K,3)
        dists,_,knn_points=pytorch3d.ops.knn_points(mid,points,K=self.knn,return_nn=True,return_sorted=False)


        knn_points1=knn_points-v1.unsqueeze(2)   #(B,M,K,3)
        knn_points2=knn_points-v2.unsqueeze(2)   #(B,M,K,3)
        
        if self.use_dist:
            #knn_points=torch.cat((knn_points,torch.sqrt(dists+1e-16).unsqueeze(-1)),dim=-1)
            knn_points1=torch.cat((knn_points1,torch.sqrt(torch.sum(knn_points1**2,dim=-1,keepdim=True)+1e-16)),dim=-1)
            knn_points2=torch.cat((knn_points2,torch.sqrt(torch.sum(knn_points2**2,dim=-1,keepdim=True)+1e-16)),dim=-1)

        knn_points1=knn_points1.permute(0,3,1,2)  #(B,3,M,K)
        knn_points2=knn_points2.permute(0,3,1,2)  #(B,3,M,K)
        patch_feature1=self.patch_feature_net1(knn_points1)    #(B,C,M,K)
        patch_feature1=torch.max(patch_feature1,dim=3)[0]     #(B,C,M,)
        
        patch_feature2=self.patch_feature_net2(knn_points2)
        patch_feature2=torch.max(patch_feature2,dim=3)[0]


        concat=torch.cat((patch_feature1,patch_feature2),dim=1)       #(B,C,M)v1-knn_points_center,v2-knn_points_center,
        y=self.regress_net(concat)          #(B,2,M)
        y=y.permute(0,2,1)

        pred_o=self.o_act(y[...,0:1])
        pred_s=self.s_act(y[...,1:])

        '''print(pred_o[0,sel_index])
        if pred_o[0,sel_index]>0.5:
            assert False'''

        return torch.cat((pred_o,pred_s),dim=-1)

if __name__=='__main__':
    net=Implicit_Segment().cuda()
    input=torch.rand(4,4096,6).cuda()
    points=torch.rand(4,10000,3).cuda()
    for i in range(10000):
        output=net(input,points)
    print(output.size())