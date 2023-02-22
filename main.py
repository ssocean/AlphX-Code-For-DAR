import os

# from get_image import ImageFetcher #测试方编写的脚本，开发者无需编写
# from evaluate import Evaluator #测试方编写的脚本，用于计算指标，这里没有给出具体调用的代码，开发者无需编写
from TestModel import Infer # 开发者需要将推理模型封装为一个TestModel.py中的Infer类，并且具有eval()方法
import glob
from tqdm import tqdm
import argparse

# from torch.utils.tensorboard import SummaryWriter
def get_files_pth(dir_pth: str, suffix: str = '*'):
    '''
    返回dir_pth下以后缀名suffix结尾的文件绝对路径list
    :param dir_pth:文件夹路径
    :param suffix:限定的文件后缀
    :return: 文件绝对路径list
    '''
    rst = []
    glob_pth = os.path.join(dir_pth, f'*.{suffix}')
    for filename in glob.glob(glob_pth):
        rst.append(filename)
    return rst
'''  # 如需启用tensorboard，请取消相关注释行
def init_tensorboard(out_dir: str = 'logs'):
    if not os.path.exists(out_dir):  ##目录存在，返回为真
        os.makedirs(out_dir)

    writer = SummaryWriter(log_dir=out_dir)
    """
    https://pytorch.org/docs/stable/tensorboard.html
    writer.
    add_scalar(tag, scalar_value, global_step=None, walltime=None, new_style=False, double_precision=False)
    add_scalars(main_tag, tag_scalar_dict, global_step=None, walltime=None)
    add_image(tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')
    add_images(tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW')
    """
    return writer
'''

parser = argparse.ArgumentParser(description="Please type the path of the image folder")
parser.add_argument('-dir', '--input_dir', type=str)
args = parser.parse_args()

if __name__ == '__main__':
    # writer = init_tensorboard('outputs/tblogs')
    output_dir = 'outputs/'
    os.makedirs(output_dir, exist_ok=True)

    inferencer = Infer() # 初始化模型

    # fetcher = ImageFetcher() 
    # ImageFetcher 是一个迭代器，产生图片路径。开发者自己编写代码进行测试的时候可以直接用测试图片的路径list代替。
    # e.g. fetcher = ['images/image_1000.jpg', 'images/image_1001.jpg', 'images/image_1002.jpg', ...] 
    fetcher = get_files_pth(args.input_dir)
    #/mnt/lustre/pazhou015/data/dataset-2/testa /data/test

    for img_path in tqdm(fetcher): 
        # img_path = r'data/test-a/17-V005P0094.png'
        inferencer.eval(img_path) #,writer
    # writer.close()
        # 模型前向过程。eval()的流程为：调用图片路径，读取并转换数据并送入模型进行预测，得到所有结果（按阅读顺序排列好的文本框以及对应的文本内容），
        # 并且在outputs文件夹中必须要生成该图片对应的csv（格式与A榜相同），否则迭代器的输出路径不会更新