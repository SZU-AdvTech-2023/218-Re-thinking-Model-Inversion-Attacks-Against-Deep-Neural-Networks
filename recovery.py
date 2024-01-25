from utils import *
from models.classify import *
from models.generator import *
from models.discri import *
import torch
import os
import numpy as np
from attack import inversion, dist_inversion
from argparse import  ArgumentParser

# 设置CPU生成随机数的种子：torch.manual_seed(...)
# 设置单一GPU生成随机数的种子：torch.cuda.manual_seed(...)
# 设置所有GPU生成随机数的种子：torch.cuda.manual_seed_all(...)

# 设置CPU生成随机数的种子，方便下次复现实验结果
torch.manual_seed(9)

# 命令行解析器（描述：Inversion）
parser = ArgumentParser(description='Inversion')
# 定义命令行参数
# configs参数，字符串类型
parser.add_argument('--configs', type=str, default='./config/celeba/attacking/celeba.json')
args = parser.parse_args()

# 参数加载
def init_attack_args(cfg):
    # gmi / kedmi
    # improved_flag：true为kedmi
    # clipz：true则使用torch.clamp对潜在向量z进行剪裁【限制在某一范围内】
    # num_seeds：优化得到的潜在向量个数/潜在向量分布
    if cfg["attack"]["method"] =='kedmi':
        args.improved_flag = True
        args.clipz = True
        args.num_seeds = 1
    else:
        args.improved_flag = False
        args.clipz = False
        args.num_seeds = 5
    # LOM：Lid → Llogit_id【改进优化目标】
    # MA：Lid → Laug_id【增强模型减轻MI过拟合】
    # LOMMA：Lid → Llogit_id + Laug_id
    # 使用LOM / LOMMA
    if cfg["attack"]["variant"] == 'L_logit' or cfg["attack"]["variant"] == 'ours':
        # 优化损失函数
        args.loss = 'logit_loss'
    else:
        # 原始损失函数【gmi/kedmi】
        args.loss = 'cel'
    # 使用MA / LOMMA
    if cfg["attack"]["variant"] == 'L_aug' or cfg["attack"]["variant"] == 'ours':
        # 目标模型 + 三个增强模型
        args.classid = '0,1,2,3'
    else:
        # 仅目标模型【原始损失函数】【gmi/kedmi】
        args.classid = '0'

if __name__ == "__main__":
    # 加载参数
    cfg = load_json(json_file=args.configs)
    init_attack_args(cfg=cfg)

    # 保存文件
    # ./attack_results/kedmi_300ids/
    if args.improved_flag == True:
        prefix = os.path.join(cfg["root_path"], "kedmi_300ids") 
    else:
        prefix = os.path.join(cfg["root_path"], "gmi_300ids")
    # celeba/ffhq
    # IR152/VGG16/FaceNet64
    # ours/L_logit/L_aug
    save_folder = os.path.join("{}_{}".format(cfg["dataset"]["name"], cfg["dataset"]["model_name"]), cfg["attack"]["variant"])
    # ./attack_results/kedmi_300ids/celeba_IR152/ours/
    prefix = os.path.join(prefix, save_folder)
    # ./attack_results/kedmi_300ids/celeba_IR152/ours/latent/
    save_dir = os.path.join(prefix, "latent")
    # ./attack_results/kedmi_300ids/celeba_IR152/ours/imgs_ours/
    save_img_dir = os.path.join(prefix, "imgs_{}".format(cfg["attack"]["variant"]))
    # ./attack_results/kedmi_300ids/celeba_IR152/ours/invertion_logs/
    args.log_path = os.path.join(prefix, "invertion_logs")

    os.makedirs(prefix, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # 加载模型【目标模型、三个增强模型、评估模型、生成器、鉴别器、采样p_reg对应分布的均值和标准差】
    targetnets, E, G, D, n_classes, fea_mean, fea_logvar = get_attack_model(args, cfg)
    N = 5
    bs = 60

    # Begin attacking
    for i in range(1):
        iden = torch.from_numpy(np.arange(bs))

        # 共攻击300个身份、每次攻击60个身份
        target_cosines = 0
        eval_cosines = 0
        for idx in range(5):
            iden = iden % n_classes
            print("--------------------- Attack batch [%s]------------------------------" % idx)
            print('Iden:{}'.format(iden))
            # ./attack_results/kedmi_300ids/celeba_IR152/latent/0_0/
            save_dir_z = '{}/{}_{}'.format(save_dir, i, idx)
            
            if args.improved_flag == True:
                # KEDMI
                print('kedmi')
                dist_inversion(G, D, targetnets, E, iden,  
                                        lr=cfg["attack"]["lr"], iter_times=cfg["attack"]["iters_mi"],
                                        momentum=0.9, lamda=100,  
                                        clip_range=1, improved=args.improved_flag, 
                                        num_seeds=args.num_seeds, 
                                        used_loss=args.loss,
                                        prefix=save_dir_z,
                                        save_img_dir=os.path.join(save_img_dir, '{}_'.format(idx)),
                                        fea_mean=fea_mean,
                                        fea_logvar=fea_logvar,
                                        lam=cfg["attack"]["lam"],
                                        clipz=args.clipz)
            else:
                # GMI
                print('gmi')
                if cfg["attack"]["same_z"] == '':
                    inversion(G, D, targetnets, E, iden,  
                                            lr=cfg["attack"]["lr"], iter_times=cfg["attack"]["iters_mi"], 
                                            momentum=0.9, lamda=100, 
                                            clip_range=1, improved=args.improved_flag,
                                            used_loss=args.loss,
                                            prefix=save_dir_z,
                                            save_img_dir=save_img_dir,
                                            num_seeds=args.num_seeds,                                        
                                            fea_mean=fea_mean,
                                            fea_logvar=fea_logvar,lam=cfg["attack"]["lam"],
                                            istart=0)
                else:
                    inversion(G, D, targetnets, E, iden,  
                                            lr=args.lr, iter_times=args.iters_mi, 
                                            momentum=0.9, lamda=100, 
                                            clip_range=1, improved=args.improved_flag,
                                            used_loss=args.loss,
                                            prefix=save_dir_z,
                                            save_img_dir=save_img_dir,
                                            num_seeds=args.num_seeds,                                        
                                            fea_mean=fea_mean,
                                            fea_logvar=fea_logvar,lam=cfg["attack"]["lam"],
                                            istart=0,
                                            same_z='{}/{}_{}'.format(args.same_z, i, idx))
            iden = iden + bs 

