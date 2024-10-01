import configargparse 
import json

def get_config(config_path=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config_path',type=str, default=config_path, is_config_file=True, 
                            help='config file path')
    
    #--- network config
    parser.add_argument('--filename', type=str,default='') 
    #parser.add_argument('--act_func',type=str,default='softplus')

    parser.add_argument('--d_pe',type=int,default=0)
    parser.add_argument('--ckpt_path',type=str,default='')
    
    parser.add_argument('--mlp_dim',type=int,default=512)

    parser.add_argument('--mlp_act',type=str,default='softplus')
    parser.add_argument('--o_act',type=str,default='sigmoid')
    parser.add_argument('--s_act',type=str,default='sigmoid')
    #parser.add_argument('--lr_schedule',type=str,default=None)

    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--lr_min',type=float,default=1e-5)
    parser.add_argument('--optimizer',type=str,default='ADAMW')

    parser.add_argument('--max_iter',type=int,default=100000)
    parser.add_argument('--batch_size',type=int,default=10000)

    parser.add_argument('--occ_argu',type=float,default=-1)
    parser.add_argument('--lr_schedual',type=str,default='cos')

    parser.add_argument('--occ_loss',type=str,default='BCE')
    parser.add_argument('--occ_weight',type=float,default=1)
    
    parser.add_argument('--self_reg',type=float,default=1)
    parser.add_argument('--cycle_loss',type=float,default=1)

    parser.add_argument('--time_folder',type=float,default=1)

    parser.add_argument('--num_sample',type=int,default='10000000')

    parser.add_argument('--surface_sample_scales',type=str,default='0.005_0.01_0.03')
    parser.add_argument('--surface_sample_ratios',type=str,default='0.5_0.3_0.1')
    parser.add_argument('--bbox_space_sample_ratio',type=str,default='0.08_0.02')
    parser.add_argument('--space_sample_scale',type=float,default=0.08)

    parser.add_argument('--surface_ratio',type=str,default='0.4')
    parser.add_argument('--near_surface_ratios',type=str,default='0.2_0.2')
    parser.add_argument('--near_surface_stds',type=str,default='0.005_0.015')
    parser.add_argument('--space_ratio',type=str,default='0.2')

    #------------------------------------------------------
    
    return parser

    #args = parser.parse_args()

def save_config(file_name, args):
    with open(file_name, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")


if __name__=='__main__':
    #args=get_config().parse_args()
    parser=get_config()
    args=parser.parse_args()
    #save_config('config.txt',parser)
    print(args)
    save_config('config.txt',args)
    