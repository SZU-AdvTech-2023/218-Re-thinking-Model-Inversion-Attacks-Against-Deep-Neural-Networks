import engine
from utils import load_json
from argparse import  ArgumentParser

# 命令行解析器（描述：Train GAN）
parser = ArgumentParser(description='Train GAN')
# 定义命令行参数
# configs参数，字符串类型
# general_gan：通用GAN
# specific_gan：半监督GAN【特定于反演的GAN】
parser.add_argument('--configs', type=str, default='./config/celeba/training_GAN/specific_gan/ffhq.json')
# mode参数，字符串类型
# specific / general
parser.add_argument('--mode', type=str, choices=['specific', 'general'], default='specific')
# 解析命令行参数
args = parser.parse_args()

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'

    # 加载JSON配置文件
    cfg = load_json(json_file=args.configs)
    # 通过GAN / 半监督GAN
    if args.mode == 'specific':
        engine.train_specific_gan(cfg=cfg)
    elif args.mode == 'general':
        engine.train_general_gan(cfg=cfg)
