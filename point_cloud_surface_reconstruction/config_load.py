import configargparse 
import json

def get_config(config_path=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config_path',type=str, default=config_path, is_config_file=True, 
                            help='config file path')
    
    #--- network config
    #parser.add_argument('--act_func',type=str,default='softplus')

    parser.add_argument('--d_pe',type=int,default=0)
    parser.add_argument('--ckpt_path',type=str,default='logs')

    parser.add_argument('--mlp_dim',type=int,default=256)
    parser.add_argument('--knn',type=int,default=20)
    parser.add_argument('--use_dist',type=int,default=1)
    parser.add_argument('--num_layers',type=int,default=4)
    parser.add_argument('--use_bn',type=int,default=1)

    parser.add_argument('--o_act',type=str,default='sigmoid')
    parser.add_argument('--s_act',type=str,default='sigmoid')
    #parser.add_argument('--lr_schedule',type=str,default=None)

    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--lr_min',type=float,default=1e-5)
    parser.add_argument('--optimizer',type=str,default='ADAMW')

    parser.add_argument('--epochs',type=int,default=400)
    parser.add_argument('--batch_size',type=int,default=4)

    parser.add_argument('--occ_argu',type=float,default=-1)
    parser.add_argument('--lr_schedual',type=str,default='cos')

    parser.add_argument('--occ_loss',type=str,default='BCE')
    parser.add_argument('--occ_weight',type=float,default=1)
    

    parser.add_argument('--data_path',type=str,default='')
    parser.add_argument('--time_folder',type=float,default=1)
    parser.add_argument('--len_dataset',type=int,default=100)
    parser.add_argument('--num_sample',type=int,default=5120)
    parser.add_argument('--num_points',type=int,default=40000)

    parser.add_argument('--max_noise_std',type=float,default=0)

    parser.add_argument('--schedular',type=int,default=0)

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
    
    """if args.config_path:
        print('load config')
        args=load_config(args.config_path)
    print(args)
"""
    #args2=load_config('config.txt')
    #print(type(args2.channel_list))