import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import torch
from torch.utils.data import Dataset
from  torchvision import transforms
from src.facial_key_points.config.config import configuration


class FacialKeyPointsDataset:
    def __int__(self):
        self.model = torch.load(configuration.saved_model)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # [r,g,b]
            std=[0.229, 0.224, 0.225]
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self,img):
        img,img_disp = self.preprocess(img)
        kps = self.model(img[None]).flatten().detach().cpu()
        kp = self.postprocess(kps)
        return img_disp, kp



    def preprocess(self, img):
        img = img.resize((224, 224))
        img =img_disp= np.asarray(img) / 255.0
        img = torch.tensor(img).permute(2,0,1)
        img = self.normalize(img).float()
        return img.to(self.device), img_disp


    def postprocess(self, img,kps):
        width,height = img.size
        kp_x, kp_y =kps[:68]*width,kps[68:]*height
        return kp_x,kp_y



if __name__ == "__main__":
    image = Image.open('').convert('RGB')
    facial_key_points = FacialKeyPointsDataset()
    image,kp = facial_key_points.predict(image)

    plt.figure()
    plt.imshow(image)
    plt.scatter(kp[0],kp[1],s=4,c='r')
    plt.savefig('')