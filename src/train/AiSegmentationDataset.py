import os
import glob
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Union

import torch
import torchvision
from torch.utils.data import Dataset

from src.train.trimap import makeTrimap

class AiSegmentationDataset(Dataset):
    """A custom Dataset(torch.utils.data) implement three functions: __init__, __len__, and __getitem__.
    Datasets are created from PTFDataModule.
    """

    def __init__(
        self,
        datasetDir: Union[str, Path],
        imageSize = 512,
        trimapSize = 5
    ) -> None:

        self.datasetDir = Path(datasetDir)
        self.imageSize = imageSize
        self.trimapSize = trimapSize

        self.image_names = glob.glob(f"{self.datasetDir}/clip_img/*/clip_*/*.jpg")
        self.mask_names = []
        clipPrefixLength = len("clip_")
        for imagePath in self.image_names:
            imageDir, imageName = os.path.split(imagePath)
            imagePartDir, clipDirName = os.path.split(imageDir)
            _, imagePartName = os.path.split(imagePartDir)
            clipNumber = clipDirName[clipPrefixLength:]

            imageNumber, _ = os.path.splitext(imageName)

            mathPath = os.path.join(self.datasetDir, "matting", imagePartName, f"matting_{clipNumber}", f"{imageNumber}.png")
            self.mask_names.append(mathPath)

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((self.imageSize, self.imageSize)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])        
        ])
        self.transform2 = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((self.imageSize, self.imageSize)),
            torchvision.transforms.ToTensor()
        ])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_pth = self.image_names[index]
        mask_pth = self.mask_names[index]

        frame = cv2.imread(frame_pth)
        frame = self.transform(frame)

        masked = cv2.imread(mask_pth)
        mask = (np.sum(masked, axis=2) > 0).astype(np.uint8).astype(float)
        trimap = makeTrimap(mask, self.trimapSize)
        trimap = torch.from_numpy(trimap).float()
        trimap = torch.unsqueeze(trimap, 0)
        mask = torch.from_numpy(mask)
        mask = torch.unsqueeze(mask, 0).float()

        mask = self.transform2(mask)
        trimap = self.transform2(trimap)
        
        return frame, trimap, mask

    def __len__(self):
        return len(self.image_names)