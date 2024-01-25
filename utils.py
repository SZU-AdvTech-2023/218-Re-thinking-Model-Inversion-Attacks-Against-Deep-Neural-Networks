import torch.nn.init as init
import os, models.facenet as facenet, sys
import json, time, random, torch
from models import classify
from models.classify import *
from models.discri import *
from models.generator import *
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tvls
from torchvision import transforms
from datetime import datetime
import dataloader
from torch.autograd import grad

device = "cuda"

# 创建自定义的输出流
class Tee(object):
    def __init__(self, name, mode):
        # 保存文件对象【不同模式：w、r、a...】
        self.file = open(name, mode)
        # 保存原来的标准输出【控制台输出】
        self.stdout = sys.stdout
        # 将原来的标准输出重定向到文件
        sys.stdout = self
    def __del__(self):
        # 恢复标准输出【控制台输出】
        sys.stdout = self.stdout
        # 关闭文件
        self.file.close()
    def write(self, data):
        if not '...' in data:
            self.file.write(data)
        self.stdout.write(data)
        self.flush()
    def flush(self):
        self.file.flush()

# 数据集迭代器
def init_dataloader(args, file_path, batch_size=64, mode="gan", iterator=False):
    tf = time.time()

    # train、test、gan、attack、
    # 是否随机洗牌数据
    # 常在训练深度学习模型时使用，以确保模型在不同批次中看到的数据的顺序是随机的，从而帮助模型更好地泛化
    if mode == "attack":
        shuffle_flag = False
    else:
        shuffle_flag = True

    # 数据集对象
    data_set = dataloader.ImageFolder(args, file_path, mode)

    # 数据集迭代器对象DataLoader（可进行数据批处理、数据随机洗牌、数据多进程加载...）
    # dataset：数据集对象
    # batch_size：批量大小【默认为1，即数据一个个馈送入网络进行训练】
    # shuffle：是否随机洗牌数据【默认为false，即数据不进行随机洗牌】
    # drop_last：: 是否丢弃最后一个不足一个批次大小的批次的样本【默认为false，则不丢弃最后一个不足一个批次大小的批次的样本】
    # num_workers：设置数据加载过程中使用的子进程数【默认为0，即在主进程中进行数据加载，不使用额外的子进程】
    # pin_memory：是否将数据拷贝到CUDA存固定内中【若使用GPU训练模型，将数据加载到固定内存中可以提高数据传输的速度，但需要额外的显存】
    if iterator:
        # 使用next()函数获取数据集中的下一batch
        data_loader = torch.utils.data.DataLoader(dataset=data_set,
                                batch_size=batch_size,
                                shuffle=shuffle_flag,
                                drop_last=True,
                                num_workers=0,
                                pin_memory=True).__iter__()
    else:
        data_loader = torch.utils.data.DataLoader(dataset=data_set,
                                batch_size=batch_size,
                                shuffle=shuffle_flag,
                                drop_last=True,
                                num_workers=2,
                                pin_memory=True)
        interval = time.time() - tf
        print('Initializing data loader took %ds' % interval)
    
    return data_set, data_loader


def load_pretrain(self, state_dict):
    own_state = self.state_dict()
    for name, param in state_dict.items():
        if name.startswith("module.fc_layer"):
            continue
        if name not in own_state:
            print(name)
            continue
        own_state[name].copy_(param.data)

# 加载目标模型的预训练参数【backbone：特征提取部分的参数】
def load_state_dict(self, state_dict):
    # 获取当前模型的参数字典
    own_state = self.state_dict()
    # 遍历预训练模型的参数字典【键值对】
    for name, param in state_dict.items():
        if name not in own_state:
            print(name)
            continue
        # 预训练参数拷贝【下划线：原地操作，即修改调用它的张量而不创建新的张量】
        own_state[name].copy_(param.data)

def load_json(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    return data

def print_params(info, params, dataset=None):
    print('-----------------------------------------------------------------')
    print("Running time: %s" % datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    for i, (key, value) in enumerate(info.items()):
        print("%s: %s" % (key, str(value)))
    for i, (key, value) in enumerate(params.items()):
        print("%s: %s" % (key, str(value)))
    print('-----------------------------------------------------------------')

def save_tensor_images(images, filename, nrow = None, normalize = True):
    if not nrow:
        tvls.save_image(images, filename, normalize = normalize, padding=0)
    else:
        tvls.save_image(images, filename, normalize = normalize, nrow=nrow, padding=0)

# 图像处理器
def get_deprocessor():
    # resize 112,112
    proc = []
    # Resize操作：将输入图像的大小调整为112x112
    # ToTensor操作：将调整大小后的图像转换为PyTorch张量
    proc.append(transforms.Resize((112, 112)))
    proc.append(transforms.ToTensor())
    # 将两个操作组合成一个处理序列，并返回该序列作为图像处理器
    return transforms.Compose(proc)

# 增大图像分辨率：64×64 → 112×112
def low2high(img):
    # 0 and 1, 64 to 112
    bs = img.size(0)
    # 图像处理器
    proc = get_deprocessor()
    # 将图像张量转换为可处理的PIL图像，并将其大小调整为112x112
    img_tensor = img.detach().cpu().float()
    img = torch.zeros(bs, 3, 112, 112)
    for i in range(bs):
        img_i = transforms.ToPILImage()(img_tensor[i, :, :, :]).convert('RGB')
        img_i = proc(img_i)
        img[i, :, :, :] = img_i[:, :, :]
    
    img = img.cuda()
    return img

def get_model(attack_name, classes):
    if attack_name.startswith("VGG16"):
        T = classify.VGG16(classes)
    elif attack_name.startswith("IR50"):
        T = classify.IR50(classes)
    elif attack_name.startswith("IR152"):
        T = classify.IR152(classes)
    elif attack_name.startswith("FaceNet64"):
        T = facenet.FaceNet64(classes)
    else:
        print("Model doesn't exist")
        exit()

    T = torch.nn.DataParallel(T).cuda()
    return T

def get_augmodel(model_name, nclass, path_T=None, dataset='celeba'):
    if model_name=="VGG16":
        model = VGG16(nclass)   
    elif model_name=="FaceNet":
        model = FaceNet(nclass)
    elif model_name=="FaceNet64":
        model = FaceNet64(nclass)
    elif model_name=="IR152":
        model = IR152(nclass)
    elif model_name =="efficientnet_b0":
        model = classify.EfficientNet_b0(nclass)
    elif model_name =="efficientnet_b1":
        model = classify.EfficientNet_b1(nclass)   
    elif model_name =="efficientnet_b2":
        model = classify.EfficientNet_b2(nclass)  

    model = torch.nn.DataParallel(model).cuda()
    if path_T is not None:
        ckp_T = torch.load(path_T)        
        model.load_state_dict(ckp_T['state_dict'], strict=True)
    return model


def log_sum_exp(x, axis = 1):
    m = torch.max(x, dim = 1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim = axis))

# define "soft" cross-entropy with pytorch tensor operations
# "soft"交叉熵损失
def softXEnt (input, target):
    # 真实图像馈送到目标模型的分类情况【bs * 1000】e^x/sum(e^x)
    targetprobs = nn.functional.softmax(target, dim = 1)
    # 真实图像【有标签】在鉴别器中的分类情况【bs * 1000】log(softmax())
    logprobs = nn.functional.log_softmax(input, dim = 1)
    # -(targetprobs * logprobs).sum() / bs
    return  -(targetprobs * logprobs).sum() / input.shape[0]

# 熵正则化项【L_entropy】
class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        # x为生成图像输入鉴别器产生的分类情况
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b
    
def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False) 

def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)

# 梯度惩罚【确保鉴别器的Lipschitz连续性】
def gradient_penalty(x, y, DG):
    # interpolation【插值】
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).cuda()
    z = x + alpha * (y - x)
    z = z.cuda()
    z.requires_grad = True

    o = DG(z)
    # 计算梯度
    g = grad(o, z, grad_outputs = torch.ones(o.size()).cuda(), create_graph = True)[0].view(z.size(0), -1)
    # 计算梯度惩罚
    # (梯度按行求L2范数 - 1)^2，再求均值
    gp = ((g.norm(p = 2, dim = 1) - 1) ** 2).mean()

    return gp

# log(sum(exp()))
def log_sum_exp(x, axis = 1):
    # 1 * bs【每行最大值】
    m = torch.max(x, dim = 1)[0]
    # x【bs * 1000】- m.unsqueeze(1)【bs * 1】
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim = axis))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 获取GAN
def get_GAN(dataset, gan_type, gan_model_dir, n_classes, z_dim, target_model):
    # 生成器
    G = Generator(z_dim)
    # 鉴别器
    # kedmi - MinibatchDiscriminator
    # gmi - DGWGAN
    if gan_type == True:
        D = MinibatchDiscriminator(n_classes=n_classes)
    else:
        D = DGWGAN(3)

    # 获取kedmi/gmi对应的生成器和鉴别器
    if gan_type == True:
        path = os.path.join(os.path.join(gan_model_dir, dataset), target_model)
        path_G = os.path.join(path, "improved_{}_G.tar".format(dataset))
        path_D = os.path.join(path, "improved_{}_D.tar".format(dataset))
    else:
        path = os.path.join(os.path.join(gan_model_dir, dataset), "general_GAN")
        path_G = os.path.join(path, "{}_G.tar".format(dataset))
        path_D = os.path.join(path, "{}_D.tar".format(dataset)) 

    print('path_G',path_G)
    print('path_D',path_D)

    G = torch.nn.DataParallel(G).to(device)
    D = torch.nn.DataParallel(D).to(device)
    ckp_G = torch.load(path_G)
    G.load_state_dict(ckp_G['state_dict'], strict=True)
    ckp_D = torch.load(path_D)
    D.load_state_dict(ckp_D['state_dict'], strict=True)
  
    return G, D

# 加载模型和p_reg[均值、标准差]
def get_attack_model(args, args_json, eval_mode=False):
    now = datetime.now() # current date and time

    # 是否为评估模式
    if not eval_mode:
        # 攻击模式：invertion_logs_ours_时间.txt
        log_file = "invertion_logs_{}_{}.txt".format(args.loss, now.strftime("%m_%d_%Y_%H_%M_%S"))
        utils.Tee(os.path.join(args.log_path, log_file), 'w')

    # 1000
    n_classes=args_json['dataset']['n_classes']
    
    # 0,1,2,3 / 0
    model_types_ = args_json['train']['model_types'].split(',')
    # 已经训练好的模型
    checkpoints = args_json['train']['cls_ckpts'].split(',')

    # 生成器和鉴别器
    G, D = get_GAN(args_json['dataset']['name'], gan_type=args.improved_flag,
                    gan_model_dir=args_json['train']['gan_model_dir'],
                    n_classes=n_classes, z_dim=100, target_model=model_types_[0])

    # celeba/ffhq
    dataset = args_json['dataset']['name']
    # 目标模型/目标模型 + 增强模型
    cid = args.classid.split(',')
    # target and student classifiers
    for i in range(len(cid)):
        id_ = int(cid[i])
        # 移除字符串首尾空字符
        model_types_[id_] = model_types_[id_].strip()
        checkpoints[id_] = checkpoints[id_].strip()
        print('Load classifier {} at {}'.format(model_types_[id_], checkpoints[id_]))
        model = get_augmodel(model_types_[id_],n_classes,checkpoints[id_],dataset)
        model = model.to(device)
        model = model.eval()
        if i == 0:
            targetnets = [model]
        else:
            targetnets.append(model)
    
        # p_reg 
        if args.loss=='logit_loss':
            if model_types_[id_]=="IR152" or model_types_[id_]=="VGG16" or model_types_[id_]=="FaceNet64":
                # 目标模型
                p_reg = os.path.join(args_json["dataset"]["p_reg_path"], '{}_{}_p_reg.pt'.format(dataset,model_types_[id_])) #'./p_reg/{}_{}_p_reg.pt'.format(dataset,model_types_[id_])
            else:
                # 增强模型
                p_reg = os.path.join(args_json["dataset"]["p_reg_path"], '{}_{}_{}_p_reg.pt'.format(dataset,model_types_[0],model_types_[id_])) #'./p_reg/{}_{}_{}_p_reg.pt'.format(dataset,model_types_[0],model_types_[id_])
            # print('p_reg',p_reg)
            # p_reg路径不存在 —— 自己训练增强模型（获取计算p_reg的分布对应的均值和方差）
            if not os.path.exists(p_reg):
                # 数据集迭代器【随机洗牌、batch_size为50】
                _, dataloader_gan = init_dataloader(args_json, args_json['dataset']['gan_file_path'], 50, mode="gan")
                from attack import get_act_reg
                # 计算5000张图像在模型倒数第二层输出特征的按列计算的均值和标准差
                fea_mean_, fea_logvar_ = get_act_reg(dataloader_gan, model, device)
                torch.save({'fea_mean':fea_mean_, 'fea_logvar':fea_logvar_}, p_reg)
            else:
                fea_reg = torch.load(p_reg)
                fea_mean_ = fea_reg['fea_mean']
                fea_logvar_ = fea_reg['fea_logvar']
            if i == 0:
                fea_mean = [fea_mean_.to(device)]
                fea_logvar = [fea_logvar_.to(device)]
            else:
                fea_mean.append(fea_mean_)
                fea_logvar.append(fea_logvar_)
            # print('fea_logvar_',i,fea_logvar_.shape,fea_mean_.shape)
            
        else:
            fea_mean, fea_logvar = 0, 0

    # evaluation classifier
    E = get_augmodel(args_json['train']['eval_model'],n_classes,args_json['train']['eval_dir'])    
    E.eval()
    G.eval()
    D.eval()

    return targetnets, E, G, D, n_classes, fea_mean, fea_logvar
