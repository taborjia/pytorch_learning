from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image


# dataset来构造数据集的类， 需要有数据集的索引用__getitem__来实现, 输出data+label的组合， 需要数据集的长度用__len__实现
# image_dir是image的名称的列表，image_item是第index的图片，image、label输出索引下的名称和标签
class Mydata(Dataset):

    def __init__(self, root_path, image_path, label_path):
        self.root_path = root_path
        self.image_path = image_path
        self.label_path = label_path
        self.image_dir = os.listdir(os.path.join(self.root_dir, self.label_dir))

    def __getitem__(self, item):
        image_item = self.image_dir[item]
        image = Image.open(os.path.join(self.root_dir, self.label_dir, image_item))
        label = self.label_dir
        return image, label

    def __len__(self):
        return len(self.image_dir)
