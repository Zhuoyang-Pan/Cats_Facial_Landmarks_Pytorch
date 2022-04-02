import os
import torch
import torchvision.transforms
from torch.utils.data import Dataset
import cv2


class landmarks(Dataset):

    def __init__(self, root_dir):
        super(landmarks,self).__init__()
        self.root_dir = root_dir
        self.img_path = os.path.join(root_dir,"cats")
        self.label_path = os.path.join(root_dir,"dots")
        files = os.listdir(self.img_path)
        files.sort()
        for i,file in enumerate(files):
            if file[0] != '.':
                self.files = files[i:]
                break

    def __getitem__(self, idx):
        img_name_jpg = self.files[idx]
        img_item_path = os.path.join(self.img_path,img_name_jpg)
        img = cv2.imread(img_item_path)
        ratio = (img.shape[0] / 224,img.shape[1]/224)
        img = cv2.resize(img,(224,224))
        label_item_path = os.path.join(self.label_path,img_name_jpg+".cat")
        with open(label_item_path) as f:
            label = f.read().split()
        label = [int(i) for i in label]
        target = []
        for i in range(1,10):
            target.append(label[i * 2] // ratio[0])
            target.append(label[i * 2 - 1] // ratio[1])
        trans_tensor = torchvision.transforms.ToTensor()
        img = trans_tensor(img)
        target = torch.Tensor(target)
        return img,target

    def __len__(self):
        return len(self.files)



if __name__ == '__main__':
    a = landmarks("archive/CAT_00")
    print(len(a))
    for i in range(1, 6):
        a = a + landmarks("archive/CAT_0" + str(i))
    print(len(a))
