## Implementation of paper "Re-thinking Model Inversion Attacks Against Deep Neural Networks" Accepted by CVPR'2023
[Original Thesis Project Website](https://ngoc-nguyen-0.github.io/re-thinking_model_inversion_attacks/)【[Paper](https://arxiv.org/pdf/2304.01669.pdf) | [Code](https://openaccess.thecvf.com/content/CVPR2023/papers/Nguyen_Re-Thinking_Model_Inversion_Attacks_Against_Deep_Neural_Networks_CVPR_2023_paper.pdf) | [Models](https://drive.google.com/drive/folders/1kq4ArFiPmCWYKY7iiV0WxxUSXtP70bFQ) | [Demo](https://colab.research.google.com/drive/1k3ml6cRV0jBZyIKbu8CTBcbraSeNP3w1?usp=sharing)】
### 1. 环境设置
Python 3.7、PyTorch 1.11.0、Cuda 11.3
```
conda create -n Re-thinking-MI-py37 python=3.7

conda activate Re-thinking-MI-py37

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -r requirements.txt
```

### 2. 数据集 & 模型检查点 & 元数据

#### 下载CelebA数据集和FFHQ数据集
- [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)：`img_align_celeba` → `./datasets/celeba/img_align_celeba`

- [FFHQ](https://github.com/NVlabs/ffhq-dataset)：`thumbnails128x128` → `./datasets/ffhq/thumbnails128x128`

#### 可下载原论文保存在Google云端硬盘的模型检查点【目标模型、评估模型、GAN、增强模型】和元数据【数据集的数据信息】

- 原论文的Google云端硬盘地址：https://drive.google.com/drive/folders/1kq4ArFiPmCWYKY7iiV0WxxUSXtP70bFQ?usp=sharing

  - 模型检查点：`CVPR23-Rethinking MI/checkpoints/`【目标模型和评估模型：`target_model`、通用`GAN`和特定于反演的`GAN`：`GAN`、增强模型：`aug_ckp`、估计`p_reg`文件：`p_reg`】
  - 元数据：`CVPR23-Rethinking MI/datasets/`【`celeba`数据集的元数据：`celeba/meta/`、`ffhq`数据集的元数据：`ffhq/meta/`】

### 3. 模型训练

#### 3.1. 训练目标模型和评估模型

- 目标模型和评估模型中特征提取模块的预训练参数文件：【下载后放置在`checkpoints/backbone/`】
  - [backbone_ir50_ms1m_epoch120.pth](https://drive.google.com/drive/folders/1omzvXV_djVIW2A7I09DWMe9JR-9o_MYh)
  - [Backbone_IR_152_Epoch_112_Batch_2547328_Time_2019-07-13-02-59_checkpoint.pth](https://pan.baidu.com/s/1-9sFB3H1mL8bt2jH7EagtA#list/path=%2Fms1m-ir152)【密码：`b197`】

- 修改配置：
  - 在CelebA数据集【私有数据集】上训练`./config/celeba/training_classifiers/classify.json`中的`datasets.model_name`修改为`IR152`/`VGG16`/`FaceNet64`/`FaceNet【评估模型】`，然后运行以下命令行以获得目标模型/评估模型
  
      ```
      python train_classifier.py
      ```

#### 3.2. 训练GAN

最先进的白盒MIA攻击使用通用GAN【[GMI](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_The_Secret_Revealer_Generative_Model-Inversion_Attacks_Against_Deep_Neural_Networks_CVPR_2020_paper.pdf)】和有助于提高攻击精度的特定于反演的GAN【[KEDMI](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Knowledge-Enriched_Distributional_Model_Inversion_Attacks_ICCV_2021_paper.pdf)】

##### 3.2.1. 训练特定于反演的GAN
* 修改配置：
  * 在CelebA数据集【公共数据集】上训练：`./config/celeba/training_GAN/specific_gan/celeba.json`
    * `dataset.img_path`：CelebA数据集的路径
    * `dataset.model_name`：目标模型`IR152`/`VGG16`/`FaceNet64`
    * `IR152/VGG16/FaceNet64.cls_ckpts`：目标模型`IR152`/`VGG16`/`FaceNet64`的检查点
  * 在FFHQ数据集【公共数据集】上训练：`./config/celeba/training_GAN/specific_gan/ffhq.json`
    * `dataset.img_path`：FFHQ数据集的路径
    * `dataset.model_name`：目标模型`IR152`/`VGG16`/`FaceNet64`
    * `IR152/VGG16/FaceNet64.cls_ckpts`：目标模型`IR152`/`VGG16`/`FaceNet64`的检查点
  
* 运行以下命令行获取特定于反演的GAN：
    ```
    python train_gan.py --configs ./config/celeba/training_GAN/specific_gan/celeba.json --mode "specific"
    python train_gan.py --configs ./config/celeba/training_GAN/specific_gan/ffhq.json --mode "specific"
    ```
  
##### 3.2.2. 训练通用GAN
* 修改配置：
  * 在CelebA数据集【公共数据集】上训练：`./config/celeba/training_GAN/general_gan/celeba.json`
    * `dataset.img_gan_path`：CelebA数据集的路径
  * 在FFHQ数据集【公共数据集】上训练：`./config/celeba/training_GAN/general_gan/ffhq.json`
    * `dataset.img_gan_path`：FFHQ数据集的路径
  
* 运行以下命令行获取通用GAN：
  ```
  python train_gan.py --configs ./config/celeba/training_GAN/general_gan/celeba.json --mode "general"
  python train_gan.py --configs ./config/celeba/training_GAN/general_gan/ffhq.json --mode "general"
  ```

#### 3.3 训练增强模型
增强模型架构采用`efficientnet_b0`、`efficientnet_b1`、`efficientnet_b2`) 
* 修改配置：
  * 在CelebA数据集【公共数据集】上训练：`./config/celeba/training_augmodel/celeba.json`
    * `dataset.img_path`：CelebA数据集的路径【作为训练数据集】
    * `dataset.img_gan_path`：CelebA数据集的路径【作为测试数据集】
    * `dataset.model_name`：教师模型【目标模型`IR152`/`VGG16`/`FaceNet64`】
    * `train.target_model_name`：教师模型【目标模型`IR152`/`VGG16`/`FaceNet64`】
    * `train.target_model_ckpt`：教师模型【目标模型`IR152`/`VGG16`/`FaceNet64`】的检查点
    * `train.student_model_name`：学生模型【增强模型`efficientnet_b0`/`efficientnet_b1`/`efficientnet_b2`】
  * 在FFHQ数据集【公共数据集】上训练：`./config/celeba/training_augmodel/ffhq.json`
    * `dataset.img_path`：CelebA数据集的路径【作为测试数据集】
    * `dataset.img_gan_path`：FFHQ数据集的路径【作为训练数据集】
    * `dataset.model_name`：教师模型【目标模型`IR152`/`VGG16`/`FaceNet64`】
    * `train.target_model_name`：教师模型【目标模型`IR152`/`VGG16`/`FaceNet64`】
    * `train.target_model_ckpt`：教师模型【目标模型`IR152`/`VGG16`/`FaceNet64`】的检查点
    * `train.student_model_name`：学生模型【增强模型`efficientnet_b0`/`efficientnet_b1`/`efficientnet_b2`】
  
* 运行以下命令行获取增强模型：
  ```
  python train_augmented_model.py --configs ./config/celeba/training_augmodel/celeba.json
  python train_augmented_model.py --configs ./config/celeba/training_augmodel/ffhq.json
  ```

### 4. 执行模型反演攻击

* 修改配置：
  * 在CelebA数据集【公共数据集】上训练：`./config/celeba/attacking/celeba.json`
    * `dataset.model_name`：攻击的目标模型`IR152`/`VGG16`/`FaceNet64`
    * `dataset.p_reg_path`：p_reg的保存路径【攻击过程中根据是否使用增强模型判断是否进行计算，计算结果保存的路径】
    * `train.model_types`：目标模型`IR152`/`VGG16`/`FaceNet64`和增强模型`efficientnet_b0`、`efficientnet_b1`、`efficientnet_b2`
    * `train.cls_ckpts`：目标模型`IR152`/`VGG16`/`FaceNet64`和增强模型`efficientnet_b0`、`efficientnet_b1`、`efficientnet_b2`的检查点
    * `train.eval_dir`：评估模型`FaceNet`的检查点
    * `attack.method`：基线方法`GMI/gmi`/`KEDMI/kedmi`
    * `attack.variant`：`L_logit`【使用改进的身份损失】/`L_aug`【使用增强模型——仅使用KL散度损失训练】/`ours`【使用改进的身份损失和增强模型——仅使用KL散度损失训练/使用KL散度损失和L2范数结合训练】/`baseline`【`GMI`/`KEDMI`】
    * `attack.eval_metric`：评估指标`acc`/`knn`/`fid`
  * 在FFHQ数据集【公共数据集】上训练：`./config/celeba/attacking/ffhq.json`
    * `dataset.model_name`：攻击的目标模型`IR152`/`VGG16`/`FaceNet64`
    * `dataset.p_reg_path`：`p_reg`的保存路径【攻击过程中根据是否使用增强模型判断是否进行计算，计算结果保存的路径】
    * `train.model_types`：目标模型`IR152`/`VGG16`/`FaceNet64`和增强模型`efficientnet_b0`、`efficientnet_b1`、`efficientnet_b2`
    * `train.cls_ckpts`：目标模型`IR152`/`VGG16`/`FaceNet64`和增强模型`efficientnet_b0`、`efficientnet_b1`、`efficientnet_b2`的检查点
    * `train.eval_dir`：评估模型`FaceNet`的检查点
    * `attack.method`：基线方法`GMI/gmi`/`KEDMI/kedmi`
    * `attack.variant`：`L_logit`【使用改进的身份损失】/`L_aug`【使用增强模型——仅使用KL散度损失训练】/`ours`【使用改进的身份损失和增强模型——仅使用KL散度损失训练/使用KL散度损失和L2范数结合训练】/`baseline`【`GMI`/`KEDMI`】
    * `attack.eval_metric`：评估指标`acc`、`knn`、`fid`

* 运行以下命令行执行模型反演攻击：
  ```
  python recovery.py --configs ./config/celeba/attacking/celeba.json
  python recovery.py --configs ./config/celeba/attacking/ffhq.json
  ```

### 5. 评估模型反演攻击

* 执行模型反演攻击后，使用相同的配置文件运行以下命令行即可得到评估结果：
  ```
  python evaluation.py --configs ./config/celeba/attacking/celeba.json
  python evaluation.py --configs ./config/celeba/attacking/ffhq.json
  ```