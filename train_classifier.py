import torch, os, engine, utils
import torch.nn as nn
from argparse import ArgumentParser
from models import classify

### 指定GPU的方法
## 终端运行Python文件时设置环境变量
# CUDA_VISIBLE_DEVICES=1 python *.py
## 代码中设置环境变量
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


## 代码中指定GPU
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.to(device)

# 命令行解析器（描述：Train Classifier）
parser = ArgumentParser(description='Train Classifier')
# 定义命令行参数（configs参数，字符串类型，默认为classify.json文件）
parser.add_argument('--configs', type=str, default='./config/celeba/training_classifiers/classify.json')
# 解析命令行参数
args = parser.parse_args()

def main(args, model_name, trainloader, testloader):
    # 1000个类别
    n_classes = args["dataset"]["n_classes"]
    # reg【回归，监督学习】 / vib【变分信息瓶颈，无监督学习】
    mode = args["dataset"]["mode"]

    # 预训练模型参数文件加载路径
    resume_path = args[args["dataset"]["model_name"]]["resume"]
    # 加载目标模型【FaceNet、FaceNet64、IR152、VGG16】和预训练模型参数【backbone：特征提取部分的参数】
    net = classify.get_classifier(model_name=model_name, mode=mode, n_classes=n_classes, resume_path=resume_path)
    
    print(net)

    # SGD优化器【模型参数、学习率控制每次参数更新的步长、动量加速SGD在相关方向上前进，抑制震荡、权重衰减以防止模型过拟合】
    optimizer = torch.optim.SGD(params=net.parameters(),
							    lr=args[model_name]['lr'], 
            					momentum=args[model_name]['momentum'], 
            					weight_decay=args[model_name]['weight_decay'])
	# 交叉熵损失
    criterion = nn.CrossEntropyLoss().cuda()
    # 将模型移到指定的计算设备
    # 在多卡的GPU服务器上，当程序的迭代次数/epochs足够大时，可使用torch.nn.DataParallel支持在多GPU上进行并行处理，以加速模型训练
    # model = model.cuda()
    # device_ids = [0, 1]  # id为0和1的两块显卡
    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    # torch.nn.DataParallel参数：
    #    module：训练的模型
    #    device_ids：训练使用的GPU
    #    output_device：输出结果使用的GPU【一般省略不写，默认使用第一个GPU】
    net = torch.nn.DataParallel(net).to(args['dataset']['device'])

    # 训练次数（表示将训练数据集中的所有样本都过一遍的训练过程）
    n_epochs = args[model_name]['epochs']
    print("Start Training!")
	# 训练模式【监督/无监督】
    mode = args["dataset"]["mode"]
    if mode == "reg":
        best_model, best_acc = engine.train_reg(args, net, criterion, optimizer, trainloader, testloader, n_epochs)
    elif mode == "vib":
        best_model, best_acc = engine.train_vib(args, net, criterion, optimizer, trainloader, testloader, n_epochs)
	# 模型保存
    torch.save({'state_dict':best_model.state_dict()}, os.path.join(model_path, "{}_{:.2f}_mine.tar").format(model_name, best_acc[0]))


if __name__ == '__main__':
    # 加载JSON配置文件
    cfg = utils.load_json(json_file=args.configs)

    # ./checkpoints/target_model
    root_path = cfg["root_path"]
    # ./checkpoints/target_model/target_ckp
    model_path = os.path.join(root_path, "target_ckp")
    # ./checkpoints/target_model/target_logs
    log_path = os.path.join(root_path, "target_logs")
    # 创建文件夹
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # FaceNet、FaceNet64、IR152、VGG16
    model_name = cfg['dataset']['model_name']
    # 模型名.txt
    log_file = "{}.txt".format(model_name)
    # ./checkpoints/target_model/target_logs/模型名.txt
    # 创建一个自定义的输出流，将标准输出和标准错误重定向写入日志文件中
    utils.Tee(os.path.join(log_path, log_file), 'w')

    # TRAINING 模型名
    # -----------------------------------------------------------------
    # Running time: 时间（年-月-日_时-分-秒）
    # 打印数据集信息和目标模型信息
    # -----------------------------------------------------------------
    print("TRAINING %s" % model_name)
    utils.print_params(cfg["dataset"], cfg[model_name], dataset=cfg['dataset']['name'])

    # ./datasets/celeba/meta/trainset.txt
    train_file = cfg['dataset']['train_file_path']
    # ./datasets/celeba/meta/testset.txt
    test_file = cfg['dataset']['test_file_path']
    # 训练/测试数据迭代器对象（一批一批加载，最后一批不足批次大小则丢弃，每一epoch会进行数据随机洗牌）
    _, trainloader = utils.init_dataloader(cfg, train_file, mode="train")
    _, testloader = utils.init_dataloader(cfg, test_file, mode="test")

    main(cfg, model_name, trainloader, testloader)