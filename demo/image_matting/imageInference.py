import cv2

from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import albumentations as albu

from src.models.modnet import MODNet

torch_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

def makePreprocessing(width, heigth, size):
    imageSize = max(width, heigth)
    preprocess = albu.Compose([
        albu.PadIfNeeded(imageSize, imageSize, border_mode=cv2.BORDER_CONSTANT, value = 0, mask_value=0),
        albu.Resize(size, size)
    ])
    return preprocess

pretrained_ckpt = './pretrained/modnet_webcam_portrait_matting.ckpt'
# pretrained_ckpt = "models/model_epoch10.ckpt"
modnet = MODNet(backbone_pretrained=False)
modnet = nn.DataParallel(modnet)

modnet.load_state_dict(torch.load(pretrained_ckpt, map_location=torch.device('cpu')))
modnet.eval()

imagePath = "data/image101_MODNet.jpg"
# imagePath = "data/image96_MODNet.jpg"
# imagePath = "data/image100_MODNet.jpg"

image = cv2.imread(imagePath)
print(f"image {image.shape}")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height = 512
scale = height / image.shape[0]
width = int(image.shape[1] * scale)
print(f"resize to {width}x{height}")
cropWidth = 512 # 672

preprocess = makePreprocessing(image.shape[1], image.shape[0], 512)
image = preprocess(image=image)["image"]
# image = cv2.resize(image, (width, height), cv2.INTER_AREA)
# image = image[:, (width - cropWidth) // 2:(width + cropWidth) // 2, :]

imagePIL = Image.fromarray(image)
imageTensor = torch_transforms(imagePIL)
imageTensor = imageTensor[None, :, :, :]

with torch.no_grad():
    _, _, matte_tensor = modnet(imageTensor, True)

matte_tensor = matte_tensor.repeat(1, 3, 1, 1)
matte_np = matte_tensor[0].data.cpu().numpy().transpose(1, 2, 0)

cv2.imshow("image", image)
cv2.imshow("matte", matte_np)
cv2.waitKey(0)