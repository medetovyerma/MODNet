import os
import argparse
import logging
import logging.handlers
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

import neptune.new as neptune

from src.models.modnet import MODNet
from src.trainer import supervised_training_iter
from src.train.AiSegmentationDataset import AiSegmentationDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetPath', type=str, required=True, help='path to dataset')
    parser.add_argument('--modelsPath', type=str, required=True, help='path to save trained MODNet models')
    parser.add_argument('--pretrainedPath', type=str, help='path of pre-trained MODNet')
    parser.add_argument('--startEpoch', type=int, default=-1, help='epoch to start with')
    parser.add_argument('--batchCount', type=int, default=16, help='batches count')
    args = parser.parse_args()
    return args

def logNeptune(neptuneRun, type: str, semantic_loss, detail_loss, matte_loss, semantic_iou):
    neptuneRun[f"training/{type}/semantic_loss"].log(semantic_loss)
    neptuneRun[f"training/{type}/detail_loss"].log(detail_loss)
    neptuneRun[f"training/{type}/matte_loss"].log(matte_loss)
    neptuneRun[f"training/{type}/semantic_iou"].log(semantic_iou)

def train(modnet, datasetPath: str, batch_size: int, startEpoch: int, modelsPath: str, device: str):
    lr = 0.01       # learn rate
    epochs = 40     # total epochs

    optimizer = torch.optim.SGD(modnet.parameters(), lr=lr, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.25 * epochs), gamma=0.1, last_epoch=startEpoch)


    dataset = AiSegmentationDataset(datasetPath, 512, 5)

    TEST_PART = 0.1
    indices = list(range(len(dataset)))[:2000]
    split = int(np.floor(TEST_PART * len(indices)))
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    trainDataloader = DataLoader(dataset, batch_size, sampler=train_sampler, drop_last=True)
    testDataloader = DataLoader(dataset, batch_size, sampler=test_sampler, drop_last=True)

    # project = 'motionlearning/modnet'
    # api_token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2NmJhMzJiMC1iY2M0LTQ5NGUtOGI0OS0yYzA4ZmNiNTRiMmEifQ=='
    # neptuneRun = neptune.init(project = project,
    #                         api_token = api_token,
    #                         source_files=[])

    for epoch in range(0, epochs):
        for idx, (image, trimap, gt_matte) in enumerate(trainDataloader):
            image = image.to(device)
            trimap = trimap.to(device)
            gt_matte = gt_matte.to(device)
            semantic_loss, detail_loss, matte_loss, semantic_iou = supervised_training_iter(modnet, optimizer, image, trimap, gt_matte, semantic_scale=1, detail_scale=10, matte_scale=1)
            if idx % 100 == 0:
                logger.info(f'idx: {idx}, semantic_loss: {semantic_loss:.5f}, detail_loss: {detail_loss:.5f}, matte_loss: {matte_loss:.5f}, semantic_iou: {semantic_iou:.5f}')
                #logNeptune(neptuneRun, "batch", semantic_loss, detail_loss, matte_loss, semantic_iou)
        logger.info(f'Epoch: {epoch}, semantic_loss: {semantic_loss:.5f}, detail_loss: {detail_loss:.5f}, matte_loss: {matte_loss:.5f}, semantic_iou: {semantic_iou:.5f}')
        
        #logNeptune(neptuneRun, "epoch", semantic_loss, detail_loss, matte_loss, semantic_iou)

        modelPath = os.path.join(modelsPath, f"model_epoch{epoch}.ckpt")
        torch.save(modnet.state_dict(), modelPath)
        logger.info(f"model saved to {modelPath}")
        lr_scheduler.step()

    torch.save(modnet.state_dict(), os.path.join(modelsPath, "model.ckpt"))

    #neptuneRun.stop()

def tune(modnet, datasetPath: str, batch_size: int, modelsPath: str, device: str):
    pass

def twoStepTrain(datasetPath: str, batch_size: int, startEpoch, pretrainedPath, modelsPath):
    device = "cuda"

    modnet = MODNet(backbone_pretrained=True)
    modnet = nn.DataParallel(modnet).to(device)

    if pretrainedPath is not None:
        modnet.load_state_dict(
            torch.load(pretrainedPath)
        )

    train(modnet, datasetPath, batch_size, startEpoch, modelsPath, device)
    tune(modnet, datasetPath, batch_size, modelsPath, device)

args = parseArgs()

twoStepTrain(args.datasetPath, args.batchCount, args.startEpoch, args.pretrainedPath, args.modelsPath)