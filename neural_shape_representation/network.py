import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pprint

"""def positional_encoding(x, d):
    pi = 3.1415927410125732

    dd = torch.arange(0,d).to(x.device)
    phase = torch.pow(2, dd)*pi
    base = torch.ones_like(x).unsqueeze(-1)
    dd = dd[None, None, :]
    dd = dd * base
    phase = torch.pow(2, dd)*pi
    phase = phase*x[...,None]

    sinp = torch.sin(phase)
    cosp = torch.cos(phase)
    pe = torch.stack([sinp, cosp], dim=-1)
    pe = pe.reshape(x.shape[0], d*6*2)

    return pe"""

"""def positional_encoding(x,d):
    #x: (N,6)
    n_channel=x.size(-1)
    num_points=x.size(0)
    xx=x.unsqueeze(-1)       #(N,6,1)
    dd=torch.arange(0,d).to(x).unsqueeze(0).unsqueeze(0)    #(1,1,d)
    dd=(2**dd)*torch.pi
    dd=xx*dd            #(N,6,d)
    sin=torch.sin(dd)   #(N,6,d)
    cos=torch.cos(dd)   #(N,6,d)

    sin=sin.reshape(num_points,-1)
    cos=cos.reshape(num_points,-1)

    output=torch.concat((sin,cos),dim=-1)

    return output

class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)"""



class MLP(nn.Module):
    def __init__(self,d=0,act='softplus',o_act='sigmoid',s_act='sigmoid',mlp_dim=512):
        super().__init__()
        self.d=d
        assert act in ['softplus','sine','gelu','leakrelu']
        if act=='softplus':
            self.layers=nn.Sequential(nn.Linear(2*(3+3*d*2),mlp_dim), nn.Softplus(100),   
                                    nn.Linear(mlp_dim,mlp_dim), nn.Softplus(100), 
                                    nn.Linear(mlp_dim,mlp_dim),nn.Softplus(100),  
                                    nn.Linear(mlp_dim,mlp_dim), nn.Softplus(100), 
                                    nn.Linear(mlp_dim,mlp_dim), nn.Softplus(100), 
                                    nn.Linear(mlp_dim,mlp_dim), nn.Softplus(100), 
                                    nn.Linear(mlp_dim,mlp_dim), nn.Softplus(100), 
                                    nn.Linear(mlp_dim,mlp_dim), nn.Softplus(100),
                                    nn.Linear(mlp_dim,2),)
        elif act=='sine':
            self.layers=nn.Sequential(nn.Linear(2*(3+3*d*2),mlp_dim), Sine(),  
                                    nn.Linear(mlp_dim,mlp_dim), Sine(), 
                                    nn.Linear(mlp_dim,mlp_dim),Sine(),  
                                    nn.Linear(mlp_dim,mlp_dim), Sine(), 
                                    nn.Linear(mlp_dim,mlp_dim), Sine(), 
                                    nn.Linear(mlp_dim,mlp_dim), Sine(), 
                                    nn.Linear(mlp_dim,mlp_dim), Sine(), 
                                    nn.Linear(mlp_dim,mlp_dim), Sine(),
                                    nn.Linear(mlp_dim,2),)
        elif act=='gelu':
            self.layers=nn.Sequential(nn.Linear(2*(3+3*d*2),mlp_dim), nn.GELU(),   
                                    nn.Linear(mlp_dim,mlp_dim), nn.GELU(),
                                    nn.Linear(mlp_dim,mlp_dim), nn.GELU(),
                                    nn.Linear(mlp_dim,mlp_dim), nn.GELU(),
                                    nn.Linear(mlp_dim,mlp_dim), nn.GELU(),
                                    nn.Linear(mlp_dim,mlp_dim), nn.GELU(),
                                    nn.Linear(mlp_dim,mlp_dim), nn.GELU(),
                                    nn.Linear(mlp_dim,mlp_dim), nn.GELU(),
                                    nn.Linear(mlp_dim,2),)
        elif act=='leakrelu':
            self.layers=nn.Sequential(nn.Linear(2*(3+3*d*2),mlp_dim), nn.LeakyReLU(0.2),   
                                    nn.Linear(mlp_dim,mlp_dim), nn.LeakyReLU(0.2),
                                    nn.Linear(mlp_dim,mlp_dim), nn.LeakyReLU(0.2),
                                    nn.Linear(mlp_dim,mlp_dim), nn.LeakyReLU(0.2),
                                    nn.Linear(mlp_dim,mlp_dim), nn.LeakyReLU(0.2),
                                    nn.Linear(mlp_dim,mlp_dim), nn.LeakyReLU(0.2),
                                    nn.Linear(mlp_dim,mlp_dim), nn.LeakyReLU(0.2),
                                    nn.Linear(mlp_dim,mlp_dim), nn.LeakyReLU(0.2),
                                    nn.Linear(mlp_dim,2),)
        else:
            print('no act func !!!')
            assert False
        
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

    def forward(self,x):
        if self.d>0:
            x=torch.cat((x,positional_encoding(x,self.d)),dim=-1)

        y=self.layers(x)

        pred_o=self.o_act(y[...,0:1])
        pred_s=self.s_act(y[...,1:])
        #y[...,0]=torch.sigmoid(y[...,0])        #o
        #y[...,1]=torch.clamp(y[...,1],0,1)      #s

        return torch.cat((pred_o,pred_s),dim=-1)
    

class MLP_UDF(nn.Module):
    def __init__(self,d=0,act='softplus',last_act='softplus',):
        super().__init__()
        self.d=d
        assert act in ['softplus','sine','gelu']
        if act=='softplus':
            self.layers=nn.Sequential(nn.utils.weight_norm(nn.Linear(1*(3+3*d*2),512)), nn.Softplus(100),   
                                    nn.utils.weight_norm(nn.Linear(512,512)), nn.Softplus(100), 
                                    nn.utils.weight_norm(nn.Linear(512,512)),nn.Softplus(100),  
                                    nn.utils.weight_norm(nn.Linear(512,512)), nn.Softplus(100), 
                                    nn.utils.weight_norm(nn.Linear(512,512)), nn.Softplus(100), 
                                    nn.utils.weight_norm(nn.Linear(512,512)), nn.Softplus(100), 
                                    nn.utils.weight_norm(nn.Linear(512,512)), nn.Softplus(100), 
                                    nn.utils.weight_norm(nn.Linear(512,512)), nn.Softplus(100),
                                    nn.utils.weight_norm(nn.Linear(512,1)),)
        elif act=='sine':
            self.layers=nn.Sequential(nn.Linear(1*(3+3*d*2),512), Sine(),  
                                    nn.Linear(512,512), Sine(), 
                                    nn.Linear(512,512),Sine(),  
                                    nn.Linear(512,512), Sine(), 
                                    nn.Linear(512,512), Sine(), 
                                    nn.Linear(512,512), Sine(), 
                                    nn.Linear(512,512), Sine(), 
                                    nn.Linear(512,512), Sine(),
                                    nn.Linear(512,1),)
        elif act=='gelu':
            self.layers=nn.Sequential(nn.Linear(1*(3+3*d*2),512), nn.GELU(),   
                                    nn.Linear(512,512), nn.GELU(),
                                    nn.Linear(512,512), nn.GELU(),
                                    nn.Linear(512,512), nn.GELU(),
                                    nn.Linear(512,512), nn.GELU(),
                                    nn.Linear(512,512), nn.GELU(),
                                    nn.Linear(512,512), nn.GELU(),
                                    nn.Linear(512,512), nn.GELU(),
                                    nn.Linear(512,1),)
        else:
            print('no act func !!!')
            assert False
        
        
        if last_act=='softplus':
            self.last_act=nn.Softplus(100)
        elif last_act=='clamp':
            self.last_act=lambda x: torch.clamp(x,0,1)
        else:
            print('no last act !!!')
            assert False

    def forward(self,x):
        if self.d>0:
            x=torch.cat((x,positional_encoding(x,self.d)),dim=-1)

        y=self.layers(x).squeeze(-1)

        y=self.last_act(y)
        #y[...,0]=torch.sigmoid(y[...,0])        #o
        #y[...,1]=torch.clamp(y[...,1],0,1)      #s

        return y
    

#----------------------------------------------------------------------------------------------------------------
class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)

def positional_encoding(x, d):

    pi = 3.1415927410125732

    dd = torch.arange(0,d).to(x.device)
    phase = torch.pow(2, dd)*pi
    base = torch.ones_like(x).unsqueeze(-1)
    dd = dd[None, None, :]
    dd = dd * base
    phase = torch.pow(2, dd)*pi
    phase = phase*x[...,None]

    sinp = torch.sin(phase)
    cosp = torch.cos(phase)
    pe = torch.stack([sinp, cosp], dim=-1)
    pe = pe.reshape(x.shape[0], -1)

    return pe


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def last_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)




"""def parse_options(return_parser=False):

    parser = argparse.ArgumentParser(description='Train deep implicit 3D geometry representations.')
    
    ## global information
    global_group = parser.add_argument_group('global')
    global_group.add_argument('--mesh_prefix', type = str, default =  'example/results/',
                              help='Path for saving the meshes')

    # Architecture for network
    net_group = parser.add_argument_group('net')
    net_group.add_argument('--init_dims', type = int, nargs = 8, default = [512, 512, 512, 512, 512, 512, 512, 512])
    net_group.add_argument('--output_dims', type = int, default = 2)
    net_group.add_argument('--dropout',type = int, nargs = 8, default = [])
    net_group.add_argument('--dropout_prob', type = float, default = 0)
    net_group.add_argument('--init_latent_in', type = int, nargs = 8, default = [])
    net_group.add_argument('--init_norm_layers', type = int, nargs = 8, default = [])
    net_group.add_argument('--weight_norm', action = 'store_true', help = 'Apply weight norm to the layers.')
    net_group.add_argument('--xyz_in_all', action = 'store_true')
    net_group.add_argument('--latent_dropout', action = 'store_true')
    net_group.add_argument('--use_pe', action = 'store_true')
    net_group.add_argument('--pe_dimen', type = int, default = 6)
    net_group.add_argument('--activation', type = str , default = 'sine', choices = ['sine', 'relu', 'softplus'])
    net_group.add_argument('--last_activation', type = str , default = 'softplus', choices = ['relu', 'softplus','sigmoid'])
    net_group.add_argument('--pretrained',type=str, default=None,
                            help = 'The checkpoint that we want to load.')

    # Parse and run
    if return_parser:
        return parser
    else:
        return argparse_to_str(parser)


def argparse_to_str(parser):

    args = parser.parse_args()

    args_dict = {}
    for group in parser._action_groups:
        group_dict = {a.dest:getattr(args, a.dest, None) for a in group._group_actions}
        args_dict[group.title] = vars(argparse.Namespace(**group_dict))

    pp = pprint.PrettyPrinter(indent=2)
    args_str = pp.pformat(args_dict)
    args_str = f'```{args_str}```'

    return args, args_str"""


class MLPNet(nn.Module):
    def __init__(
            self,
            args    #=parse_options(True).parse_args()
    ):
        super(MLPNet, self).__init__()

        self.args = args
        self.use_pe =self.args.pe_dimen>0 #self.args.use_pe
        if self.use_pe:
            self.xyz_dims = self.args.pe_dimen * 3 * 2 *2
        else:
            self.xyz_dims = 6

        # Init the network structure
        dims_init = [self.xyz_dims] + self.args.init_dims + [self.args.output_dims]
        self.num_layers_init = len(dims_init)
  
        for layer in range(0, self.num_layers_init - 1):
            in_dim = dims_init[layer]
            if layer + 1 in self.args.init_latent_in:
                out_dim = dims_init[layer + 1] - dims_init[0]
            else:
                out_dim = dims_init[layer + 1]
                if self.args.xyz_in_all and layer != self.num_layers_init - 2:
                    out_dim -= self.xyz_dims
            if self.args.weight_norm and layer in self.args.init_norm_layers:
                setattr(
                    self,
                    "lin_" + str(layer),
                    nn.utils.weight_norm(nn.Linear(in_dim, out_dim)),
                )
            else:
                setattr(self, "lin_" + str(layer), nn.Linear(in_dim, out_dim))
            if (
                    (not self.args.weight_norm)and self.args.init_norm_layers is not None and layer in self.args.init_norm_layers
            ):
                setattr(self, "bn_" + str(layer), nn.LayerNorm(out_dim))

        # Activate function
        if self.args.activation == "relu":
            self.activation = nn.ReLU()
        elif self.args.activation == "softplus":
            self.activation = nn.Softplus(beta=100)
        else:
            self.activation = Sine()
            for layer in range(0, self.num_layers_init - 1):
                lin = getattr(self, "lin_" + str(layer))
                if layer == 0:
                    first_layer_sine_init(lin)
                elif layer == self.num_layers_init - 2:
                    last_layer_sine_init(lin)
                else:
                    sine_init(lin)

        # Setup the switch for the last layer output
        self.sp = nn.Softplus(beta=100)
        self.relu = nn.ReLU()

        # Setup dropouts
        if self.args.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)


    # input: N x (L+3)
    def forward(self, input, write_debug=False):
        num_points=input.size(0)
        xyz = input #[..., -3:]

        x = xyz

        if self.use_pe:
            pe = positional_encoding(x, self.args.pe_dimen)
            x = pe

        # Forward the network
        for layer in range(0, self.num_layers_init - 1):
            lin = getattr(self, "lin_" + str(layer))
            if layer != 0 and self.args.xyz_in_all:
                x = torch.cat([x, xyz], -1)
            x = lin(x)
            # bn and activation
            if layer < self.num_layers_init - 2:
                if (
                        self.args.init_norm_layers is not None
                        and layer in self.args.init_norm_layers
                        and not self.args.weight_norm
                ):
                    bn = getattr(self, "bn_" + str(layer))
                    x = bn(x)
                x = self.activation(x)
                if self.args.dropout is not None and layer in self.args.dropout:
                    x = F.dropout(x, p=self.args.dropout_prob, training=self.training)
        if self.args.last_activation == 'softplus':
            x = self.sp(x)
        elif self.args.last_activation=='sigmoid':
            x=torch.sigmoid(x)
        else:
            assert False
            x = self.relu(x)
        return x

if __name__=='__main__':
    a=torch.rand(1000,6)
    b=positional_encoding(a,10)
    print(b.size())