import os
import os.path

import torch.utils.data as data
from PIL import Image


def default_loader(path):
    size = 140, 140
    img = Image.open(path).convert('L')
    img = img.thumbnail(size, Image.ANTIALIAS)
    return img


def load_list(fileList):
    imgList = []
    max_label = 0
    with open(fileList, 'r') as file:
        for line in file.readlines():
            id, imgPath, yaw = line.strip().split('\t')
            imgList.append((imgPath, int(id), float(yaw)))
            max_label = max(max_label, int(id))
    return imgList, max_label


class ImageList(data.Dataset):
    def __init__(self, root, fileList, transform=None):
        self.root = root
        self.imgList, self.max_label = load_list(fileList)
        self.transform = transform
        self.loader = default_loader

    def __getitem__(self, index):
        imgPath, target, yaw = self.imgList[index]
        img = self.loader(os.path.join(self.root, imgPath))

        if self.transform is not None:
            img = self.transform(img)
        return img, target, yaw

    def __len__(self):
        return len(self.imgList)
