import torch, os, time, utils
import torch.nn as nn
from copy import deepcopy
from utils import *
from models.discri import MinibatchDiscriminator, DGWGAN
from models.generator import Generator
from models.classify import *
from tensorboardX import SummaryWriter

# 测试模型训练效果【每训练完一次】
def test(model, criterion=None, dataloader=None, device='cuda'):
    tf = time.time()
    # 评估模式
    model.eval()
    loss, cnt, ACC, correct_top5 = 0.0, 0, 0,0
    # 禁用梯度计算
    with torch.no_grad():
        for i,(img, iden) in enumerate(dataloader):
            img, iden = img.to(device), iden.to(device)
            bs = img.size(0)
            iden = iden.view(-1)
            _, out_prob = model(img)
            out_iden = torch.argmax(out_prob, dim=1).view(-1)
            ACC += torch.sum(iden == out_iden).item()

            # 按行获取概率前5的样本索引
            _, top5 = torch.topk(out_prob,5, dim = 1)
            # 判断目标标签是否在样本索引列表中
            for ind, top5pred in enumerate(top5):
                if iden[ind] in top5pred:
                    correct_top5 += 1
        
            cnt += bs

    return ACC * 100.0/cnt, correct_top5 * 100.0/cnt

# 目标模型训练【监督学习】
def train_reg(args, model, criterion, optimizer, trainloader, testloader, n_epochs, device='cuda'):
    best_ACC = (0.0, 0.0)

    # 训练次数
    for epoch in range(n_epochs):
        tf = time.time()
        # 准确率、训练总样本数、训练总损失
        ACC, cnt, loss_tot = 0, 0, 0.0
        # 将模型设置为训练模式【使Batch Normalization和Dropout等层正常工作】
        # 在训练模式下，Batch Normalization层使用当前batch的统计信息（均值和方差）进行归一化，以加速训练过程；而Dropout层会随机丢弃一部分神经元，以防止过拟合
        # 在评估模式下【model.eval()】，Batch Normalization层使用所有训练数据的统计信息（均值和方差）进行归一化，以保证模型的稳定性；而Dropout层会保留所有神经元，以提高模型的准确性
        model.train()

		# 枚举函数enumerate：返回元素索引和元素【一batch的数据】
        for i, (img, iden) in enumerate(trainloader):
            # 把数据放到GPU
            img, iden = img.to(device), iden.to(device)
            # 统计有多少行数据【一batch】
            bs = img.size(0)
            # 动态调整张量形状【1 * bs】
            iden = iden.view(-1)

            # 输入目标模型
            feats, out_prob = model(img)
            # 交叉熵损失
            cross_loss = criterion(out_prob, iden)
            loss = cross_loss

            # 将优化器中的梯度归零【PyTorch默认会在反向传播时累加梯度，而不是覆盖之前的梯度】
            optimizer.zero_grad()
            # 反向传播计算模型参数对损失的梯度
            loss.backward()
            # 利用计算得到的梯度，更新模型参数
            optimizer.step()

            # 每个样本的最大概率身份对应的索引【1 * bs】
            out_iden = torch.argmax(out_prob, dim=1).view(-1)
            # 目标模型预测成功样本数
            ACC += torch.sum(iden == out_iden).item()
            loss_tot += loss.item() * bs
            cnt += bs

        # 本次训练的平均损失和平均目标模型预测成功率
        train_loss, train_acc = loss_tot * 1.0 / cnt, ACC * 100.0 / cnt
        # 每训练一次后进行测试集测试模型训练效果
        test_acc = test(model, criterion, testloader)

        interval = time.time() - tf
        # 判断测试准确率是否比最佳测试准确率高
        if test_acc[0] > best_ACC[0]:
            best_ACC = test_acc
            # 深拷贝目标模型
            best_model = deepcopy(model)

        print("Epoch:{}\tTime:{:.2f}\tTrain Loss:{:.2f}\tTrain Acc:{:.2f}\tTest Acc:{:.2f}".format(epoch, interval, train_loss, train_acc, test_acc[0]))

    print("Best Acc:{:.2f}".format(best_ACC[0]))
    return best_model, best_ACC

# 目标模型训练【无监督学习】
def train_vib(args, model, criterion, optimizer, trainloader, testloader, n_epochs, device='cuda'):
	best_ACC = (0.0, 0.0)
	
	for epoch in range(n_epochs):
		tf = time.time()
		ACC, cnt, loss_tot = 0, 0, 0.0
		
		for i, (img, iden) in enumerate(trainloader):
			img, one_hot, iden = img.to(device), one_hot.to(device), iden.to(device)
			bs = img.size(0)
			iden = iden.view(-1)
			
			___, out_prob, mu, std = model(img, "train")
			cross_loss = criterion(out_prob, one_hot)
			info_loss = - 0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(dim=1).mean()
			loss = cross_loss + beta * info_loss

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			out_iden = torch.argmax(out_prob, dim=1).view(-1)
			ACC += torch.sum(iden == out_iden).item()
			loss_tot += loss.item() * bs
			cnt += bs

		train_loss, train_acc = loss_tot * 1.0 / cnt, ACC * 100.0 / cnt
		test_loss, test_acc = test(model, criterion, testloader)

		interval = time.time() - tf
		if test_acc[0] > best_ACC[0]:
			best_ACC = test_acc
			best_model = deepcopy(model)
			
		print("Epoch:{}\tTime:{:.2f}\tTrain Loss:{:.2f}\tTrain Acc:{:.2f}\tTest Acc:{:.2f}".format(epoch, interval, train_loss, train_acc, test_acc[0]))

	print("Best Acc:{:.2f}".format(best_ACC[0]))
	return best_model, best_ACC

# 获取目标模型
def get_T(model_name_T, cfg):
    if model_name_T.startswith("VGG16"):
        T = VGG16(cfg['dataset']["n_classes"])
    elif model_name_T.startswith('IR152'):
        T = IR152(cfg['dataset']["n_classes"])
    elif model_name_T == "FaceNet64":
        T = FaceNet64(cfg['dataset']["n_classes"])
    T = torch.nn.DataParallel(T).cuda()
    # 加载模型参数
    ckp_T = torch.load(cfg[cfg['dataset']['model_name']]['cls_ckpts'])
    # 模型参数拷贝
    T.load_state_dict(ckp_T['state_dict'], strict=False)

    return T

# 半监督GAN训练
def train_specific_gan(cfg):
    # 参数读取
    # ./datasets/celeba/meta/ganset.txt
    file_path = cfg['dataset']['gan_file_path']
    # VGG16 / IR152 / FaceNet64【目标模型】
    model_name_T = cfg['dataset']['model_name']
    # celeba / ffhq
    dataset_name = cfg['dataset']['name']
    # 批量大小【64】
    batch_size = cfg[model_name_T]['batch_size']
    # 潜在向量维度【100】
    z_dim = cfg[model_name_T]['z_dim']
    # 更新n次鉴别器后更新生成器【稳定训练过程，加速模型收敛】【5】
    n_critic = cfg[model_name_T]['n_critic']

    # 创建保存文件夹【训练后模型、生成图像】
    root_path = cfg["root_path"]
    # ./checkpoints/GAN/celeba/VGG16/
    save_model_dir = os.path.join(root_path, os.path.join(dataset_name, model_name_T))
    # ./checkpoints/GAN/celeba/VGG16/imgs/
    save_img_dir = os.path.join(save_model_dir, "imgs")
    os.makedirs(save_model_dir, exist_ok=True)
    os.makedirs(save_img_dir, exist_ok=True)

    # 日志文件
    # ./checkpoints/GAN/celeba/VGG16/attack_logs/
    log_path = os.path.join(save_model_dir, "attack_logs")
    os.makedirs(log_path, exist_ok=True)
    # improvedGAN_VGG16.txt
    log_file = "improvedGAN_{}.txt".format(model_name_T)
    utils.Tee(os.path.join(log_path, log_file), 'w')
    # SummaryWriter是PyTorch中TensorBoard的接口，用于将训练过程中的信息记录到TensorBoard日志中，以便于可视化和分析
    # 在训练过程中，可使用writer.add_scalar、writer.add_image、writer.add_histogram等方法将不同类型的信息写入到TensorBoard日志中
    # 可通过在终端中运行"tensorboard --logdir=<log_path>"来启动TensorBoard服务器，再通过浏览器访问TensorBoard的Web界面进行可视化
    writer = SummaryWriter(log_path)

    # 加载目标模型
    T = get_T(model_name_T=model_name_T, cfg=cfg)

    # 数据集对象、数据集迭代器对象
    dataset, dataloader = utils.init_dataloader(cfg, file_path, cfg[model_name_T]['batch_size'], mode="gan")

    # 开始训练
    print("Training GAN for %s" % model_name_T)
    # 打印数据集和模型训练参数信息
    utils.print_params(cfg["dataset"], cfg[model_name_T])

    # 生成器
    G = Generator(cfg[model_name_T]['z_dim'])
    # 鉴别器
    DG = MinibatchDiscriminator()
    
    G = torch.nn.DataParallel(G).cuda()
    DG = torch.nn.DataParallel(DG).cuda()

    # Adam优化器【模型参数、学习率、两个动量参数、eps为一个很小的值，用于维持数值稳定性，默认为1e-8】
    g_optimizer = torch.optim.Adam(G.parameters(), lr=cfg[model_name_T]['lr'], betas=(0.5, 0.999))
    dg_optimizer = torch.optim.Adam(DG.parameters(), lr=cfg[model_name_T]['lr'], betas=(0.5, 0.999))

    # 熵正则化项【L_entropy】
    # 目标网络是在私有数据上进行训练的，因此私有数据在输入到目标网络时应具有高置信度，进而应获得较低的预测熵
    # 为鼓励从公共数据中学习到的数据分布模仿私有数据，用熵正则化项显式约束损失函数中的熵，使生成数据在目标网络下具有较低的熵
    entropy = HLoss()

    # 步数【和n_critic结合，判断是否更新生成器参数】
    step = 0
    # 训练次数
    for epoch in range(cfg[model_name_T]['epochs']):
        start = time.time()
        # 数据集迭代器【无标签，用next()函数获取数据集的下一batch】
        _, unlabel_loader1 = init_dataloader(cfg, file_path, batch_size, mode="gan", iterator=True)
        _, unlabel_loader2 = init_dataloader(cfg, file_path, batch_size, mode="gan", iterator=True)

        # 一batch的数据
        for i, imgs in enumerate(dataloader):
            # 鉴别器训练
            # 迭代次数
            current_iter = epoch * len(dataloader) + i + 1
            step += 1
            imgs = imgs.cuda()
            bs = imgs.size(0)

            # 获取数据集的下一batch
            x_unlabel = unlabel_loader1.next()
            x_unlabel2 = unlabel_loader2.next()

            # 冻结和解冻模型【生成器/鉴别器是否参与训练】
            freeze(G)
            unfreeze(DG)

            # 采样得到潜在向量【标准正态分布】
            z = torch.randn(bs, z_dim).cuda()
            # 生成图像
            f_imgs = G(z)

            # 真实图像馈送到目标模型的分类情况【bs * 1000】
            y_prob = T(imgs)[-1]
            # 真实图像的伪标签【1 * bs】
            y = torch.argmax(y_prob, dim=1).view(-1)
            
            # 真实图像【有标签】在鉴别器中的分类情况【bs * 1000】
            _, output_label = DG(imgs)
            # 真实图像【无标签】在鉴别器中的分类情况【bs * 1000】
            _, output_unlabel = DG(x_unlabel)
            # 生成图像在鉴别器中的分类情况【bs * 1000】
            _, output_fake = DG(f_imgs)

            ## 鉴别器损失 = 监督学习损失 + 无监督学习损失
            # 监督学习损失【交叉熵损失】
            loss_lab = softXEnt(output_label, y_prob)
            # 无监督学习损失【无标签真实图像 - 无标签真实图像 + 生成图像】
            # softplus：平滑地近似ReLU激活函数
            loss_unlab = 0.5*(torch.mean(F.softplus(log_sum_exp(output_unlabel)))-torch.mean(log_sum_exp(output_unlabel))+torch.mean(F.softplus(log_sum_exp(output_fake))))
            dg_loss = loss_lab + loss_unlab

            # 鉴别器对真实图像【有标签】分类的准确率
            acc = torch.mean((output_label.max(1)[1] == y).float())

            # 将优化器中的梯度归零【PyTorch默认会在反向传播时累加梯度，而不是覆盖之前的梯度】
            dg_optimizer.zero_grad()
            # 反向传播计算模型参数对损失的梯度
            dg_loss.backward()
            # 利用计算得到的梯度，更新模型参数
            dg_optimizer.step()

            # 将训练过程中的信息记录到TensorBoard日志中，以便于可视化和分析
            writer.add_scalar('loss_label_batch', loss_lab, current_iter)
            writer.add_scalar('loss_unlabel_batch', loss_unlab, current_iter)
            writer.add_scalar('DG_loss_batch', dg_loss, current_iter)
            writer.add_scalar('Acc_batch', acc, current_iter)

            # 生成器训练
            if step % n_critic == 0:
                freeze(DG)
                unfreeze(G)
                # 采样得到潜在向量【标准正态分布】
                z = torch.randn(bs, z_dim).cuda()
                # 生成图像
                f_imgs = G(z)

                # 生成图像的特征图 ，64 * 1000
                mom_gen, output_fake = DG(f_imgs)
                # 真实图像【无标签】的特征图
                mom_unlabel, _ = DG(x_unlabel2)

                # 按列求均值
                mom_gen = torch.mean(mom_gen, dim = 0)
                mom_unlabel = torch.mean(mom_unlabel, dim = 0)

                # 熵正则化项【显式约束损失函数中的熵，使生成图像在目标网络下具有较低的熵】
                Hloss = entropy(output_fake)
                # 生成器损失【真实图像特征和生成图像特征的L2距离 + λh * 熵正则化项】
                g_loss = torch.mean((mom_gen - mom_unlabel).abs()) + 1e-4 * Hloss  

                # 优化器梯度归零、反向传播计算模型参数对损失的梯度、利用计算得到的梯度更新模型参数
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

                # 将训练过程中的信息记录到TensorBoard日志中，以便于可视化和分析
                writer.add_scalar('G_loss_batch', g_loss, current_iter)

        end = time.time()
        interval = end - start
        
        print("Epoch:%d \tTime:%.2f\tG_loss:%.2f\t train_acc:%.2f" % (epoch, interval, g_loss, acc))

        # 每epoch都保存一次模型
        torch.save({'state_dict':G.state_dict()}, os.path.join(save_model_dir, "improved_{}_G.tar".format(dataset_name)))
        torch.save({'state_dict':DG.state_dict()}, os.path.join(save_model_dir, "improved_{}_D.tar".format(dataset_name)))

        # 每10个epoch进行一次图像保存
        if (epoch+1) % 10 == 0:
            z = torch.randn(32, z_dim).cuda()
            fake_image = G(z)
            # ./checkpoints/GAN/celeba/VGG16/imgs/improved_celeba_img_epoch次数.png
            save_tensor_images(fake_image.detach(), os.path.join(save_img_dir, "improved_celeba_img_{}.png".format(epoch)), nrow = 8)

# 通用GAN训练
def train_general_gan(cfg):
    # 参数读取
    file_path = cfg['dataset']['gan_file_path']
    model_name = cfg['dataset']['model_name']
    dataset_name = cfg['dataset']['name']
    lr = cfg[model_name]['lr']
    batch_size = cfg[model_name]['batch_size']
    z_dim = cfg[model_name]['z_dim']
    epochs = cfg[model_name]['epochs']
    n_critic = cfg[model_name]['n_critic']

    # 文件和文件夹创建
    root_path = cfg["root_path"]
    # ./checkpoints/GAN/celeba/general_GAN/
    save_model_dir = os.path.join(root_path, os.path.join(dataset_name, 'general_GAN'))
    # ./checkpoints/GAN/celeba/general_GAN/imgs/
    save_img_dir = os.path.join(save_model_dir, "imgs")
    os.makedirs(save_model_dir, exist_ok=True)
    os.makedirs(save_img_dir, exist_ok=True)

    # 日志文件
    log_path = os.path.join(save_model_dir, "logs")
    os.makedirs(log_path, exist_ok=True)
    log_file = "GAN_{}.txt".format(dataset_name)
    utils.Tee(os.path.join(log_path, log_file), 'w')
    writer = SummaryWriter(log_path)

    # 数据集迭代器对象
    dataset, dataloader = init_dataloader(cfg, file_path, batch_size, mode="gan")

    # 开始训练
    print("Training general GAN for %s" % dataset_name)
    utils.print_params(cfg["dataset"], cfg[model_name])

    G = Generator(z_dim)
    DG = DGWGAN(3)
    
    G = torch.nn.DataParallel(G).cuda()
    DG = torch.nn.DataParallel(DG).cuda()

    g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    dg_optimizer = torch.optim.Adam(DG.parameters(), lr=lr, betas=(0.5, 0.999))

    step = 0
    for epoch in range(epochs):
        start = time.time()
        for i, imgs in enumerate(dataloader):
            # 鉴别器训练
            step += 1
            imgs = imgs.cuda()
            bs = imgs.size(0)
            
            freeze(G)
            unfreeze(DG)

            # 采样随机变量生成图像
            z = torch.randn(bs, z_dim).cuda()
            f_imgs = G(z)

            # 真实图像鉴别情况
            r_logit = DG(imgs)
            # 生成图像鉴别情况
            f_logit = DG(f_imgs)

            # 鉴别器损失
            # Wasserstein loss【真实图像鉴别情况均值 - 生成图像鉴别情况均值】
            wd = r_logit.mean() - f_logit.mean()  # Wasserstein-1 Distance
            # 梯度惩罚【确保鉴别器的Lipschitz连续性】
            gp = gradient_penalty(imgs.data, f_imgs.data, DG=DG)
            # WGAN-GP
            dg_loss = - wd + gp * 10.0
            
            dg_optimizer.zero_grad()
            dg_loss.backward()
            dg_optimizer.step()

            # 生成器训练
            if step % n_critic == 0:
                freeze(DG)
                unfreeze(G)
                z = torch.randn(bs, z_dim).cuda()
                f_imgs = G(z)
                logit_dg = DG(f_imgs)
                # calculate g_loss
                g_loss = - logit_dg.mean()
                
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

        end = time.time()
        interval = end - start
        
        print("Epoch:%d \t Time:%.2f\t Generator loss:%.2f" % (epoch, interval, g_loss))

        # 每epoch都保存一次模型
        torch.save({'state_dict':G.state_dict()}, os.path.join(save_model_dir, "ffhq_G.tar"))
        torch.save({'state_dict':DG.state_dict()}, os.path.join(save_model_dir, "ffhq_D.tar"))

        if (epoch+1) % 10 == 0:
            # 32 * 100
            z = torch.randn(32, z_dim).cuda()
            fake_image = G(z)
            # ./checkpoints/GAN/celeba/general_GAN/imgs/result_image_epoch次数.png
            save_tensor_images(fake_image.detach(), os.path.join(save_img_dir, "result_image_{}.png".format(epoch)), nrow = 8)

# 增强模型训练【知识蒸馏】
def train_augmodel(cfg):
    # 参数读取
    # IR152 / VGG16 / FaceNet64
    target_model_name = cfg['train']['target_model_name']
    # efficientnet_b0 / b1 / b2
    student_model_name = cfg['train']['student_model_name']
    device = cfg['train']['device']
    lr = cfg['train']['lr']
    temperature = cfg['train']['temperature']
    seed = cfg['train']['seed']
    epochs = cfg['train']['epochs']
    log_interval = cfg['train']['log_interval']
    dataset_name = cfg['dataset']['name']
    n_classes = cfg['dataset']['n_classes']
    batch_size = cfg['dataset']['batch_size']
    
    # 文件夹创建
    # ./checkpoints/aug_ckp/celeba/
    save_dir = os.path.join(cfg['root_path'], dataset_name)
    # ./checkpoints/aug_ckp/celeba/IR152_efficientnet_b0_0.01_1.0/
    save_dir = os.path.join(save_dir, '{}_{}_{}_{}'.format(target_model_name, student_model_name, lr, temperature))
    os.makedirs(save_dir, exist_ok=True)

    # 日志文件
    now = datetime.now() # current date and time
    # ./checkpoints/aug_ckp/celeba/IR152_efficientnet_b0_0.01_1.0/studentKD_logs_时间[秒].txt
    log_file = "studentKD_logs_{}.txt".format(now.strftime("%m_%d_%Y_%H_%M_%S"))
    utils.Tee(os.path.join(save_dir, log_file), 'w')
    # 设置CPU生成随机数的种子，方便下次复现实验结果
    torch.manual_seed(seed)

    kwargs = {'batch_size': batch_size}
    if device == 'cuda':
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},)
    
    # 获取模型【目标模型和增强模型】
    teacher_model = get_augmodel(target_model_name, n_classes, cfg['train']['target_model_ckpt'])
    model = get_augmodel(student_model_name, n_classes)
    model = model.to(device)
    print('Target model {}: {} params'.format(target_model_name, count_parameters(model)))
    print('Augmented model {}: {} params'.format(student_model_name, count_parameters(teacher_model)))

    # SGD优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    # 学习率调整【每20个step将学习率调整为原来的0.5】【逐渐减小学习率，避免结果在最优解处震荡】
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # 数据集迭代器对象（训练集和测试集）
    _, dataloader_train = init_dataloader(cfg, cfg['dataset']['gan_file_path'], batch_size, mode="gan")
    _, dataloader_test = init_dataloader(cfg, cfg['dataset']['test_file_path'], batch_size, mode="test")

    # 开始训练
    top1, top5 = test(teacher_model, dataloader=dataloader_test)
    print("Target model {}: top 1 = {}, top 5 = {}".format(target_model_name, top1, top5))

    # Kullback-Leibler divergence loss【相对熵】
    # KL散度【衡量两个概率分布之间的相似性，其值越小，概率分布越接近】
    kl_loss_function = nn.KLDivLoss(reduction='sum')
    ce_loss_function = nn.CrossEntropyLoss(reduction="mean")
    mse_loss_function = nn.MSELoss(reduction="mean")
    # 目标模型为评估模式
    teacher_model.eval()
    # 训练次数【1 ~ 20】
    for epoch in range(1, epochs + 1):
        # 增强模型为训练模式
        model.train()
        # 遍历训练集
        for batch_idx, data in enumerate(dataloader_train):
            data = data.to(device)
            curr_batch_size = len(data)

            # 教师模型输出
            _, output_t = teacher_model(data)
            # 学生模型输出
            _, output = model(data)

            # KL散度
            kl_loss = kl_loss_function(
                F.log_softmax(output / temperature, dim=-1),
                F.softmax(output_t / temperature, dim=-1)
            ) / (temperature * temperature) / curr_batch_size

            # 添加L2范数
            l2_loss = torch.mean((output_t - output).pow(2))
            total_loss = kl_loss + 1.0 * l2_loss

            # 使用学生模型的预测结果与真实标签进行损失计算（无真实标签）

            # 更换损失
            #with torch.no_grad():
            #    _, label = torch.max(output_t.data, dim=1)
            #label = label.detach().cpu().numpy()
            #label = torch.from_numpy(label).cuda().long()
            #ce_loss = ce_loss_function(output, label)
            #bd_loss = mse_loss_function(F.softmax(output, dim=1), F.softmax(output_t, dim=1))
            #total_loss = kl_loss + 1.0 * l2_loss + 1000.0 * bd_loss

            optimizer.zero_grad()
            total_loss.backward()
            # kl_loss.backward()
            optimizer.step()

            if (log_interval > 0) and (batch_idx % log_interval == 0):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tkl_loss: {:.6f}\tl2_loss: {:.6f}\ttotal_loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(dataloader_train.dataset), 100. * batch_idx / len(dataloader_train),
                    kl_loss.item(), l2_loss.item(), total_loss.item()))
                #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tkl_loss: {:.6f}'.format(
                #        epoch, batch_idx * len(data), len(dataloader_train.dataset),
                #               100. * batch_idx / len(dataloader_train), kl_loss.item()))
                #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tce_loss: {:.6f}\tbd_loss: {:.6f}\ttotal_loss: {:.6f}'.format(
                #    epoch, batch_idx * len(data), len(dataloader_train.dataset), 100. * batch_idx / len(dataloader_train),
                #    ce_loss.item(), bd_loss.item(), total_loss.item()))
                #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tkl_loss: {:.6f}\tl2_loss: {:.6f}\tbd_loss: {:.6f}\ttotal_loss: {:.6f}'.format(
                #    epoch, batch_idx * len(data), len(dataloader_train.dataset),
                #           100. * batch_idx / len(dataloader_train),
                #    kl_loss.item(), l2_loss.item(), bd_loss.item(), total_loss.item()))
                  
        scheduler.step()
        # 每次训练后使用测试集测试训练效果
        top1, top5 = test(model, dataloader=dataloader_test)
        print("epoch {}: top 1 = {}, top 5 = {}".format(epoch, top1, top5))
        
        if (epoch + 1) % 10 == 0:
            # ./checkpoints/aug_ckp/celeba/IR152_efficientnet_b0_0.01_1.0/IR152_efficientnet_b0_1_20.pt
            save_path = os.path.join(save_dir, "{}_{}_kd_{}_{}.pt".format(target_model_name, student_model_name, seed, epoch+1))
            torch.save({'state_dict':model.state_dict()}, save_path)