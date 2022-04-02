import cv2
import torch
from torch.utils.data import DataLoader

from load import landmarks

model = torch.load("models/Cats_landmarks.pth")

test_data = landmarks("archive/CAT_07")
test_loader = DataLoader(test_data)

model.eval()
with torch.no_grad():
    for i,data in enumerate(test_loader):
        imgs, labels = data
        outputs = model(imgs)
        py_img = cv2.imread(f"archive/CAT_07/cats/{i}.jpg")
        py_img = cv2.resize(py_img,(224,224))
        outputs = outputs.reshape(-1)
        outputs = outputs.numpy()
        for j in range(9): # 9 represents the number of landmarks
            x = int(outputs[j*2])
            y = int(outputs[j*2+1])
            py_img[x,y] = [0,0,255]
        cv2.imwrite(f"pic{i}.png",py_img)