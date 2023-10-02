import torch
import torch.nn as nn
import sys
import os
import time
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import common as utils		# 下面给出代码
from tqdm import tqdm
import onnxruntime as ort
import onnx
import glob
import argparse
import multiprocessing

class Data:
    def __init__(self, data_path):
        scale_size = 224

        valdir = os.path.join(data_path, 'val')
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        testset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Resize(scale_size),
                transforms.ToTensor(),
                normalize,
            ]))

        self.loader_test = DataLoader(
            testset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True)

def test_onnxruntime(ort_session, testLoader, logger, topk=(1,)):
    accuracy = utils.AverageMeter('Acc@1', ':6.2f')
    top5_accuracy = utils.AverageMeter('Acc@5', ':6.2f')

    start_time = time.time()
    testLoader = tqdm(testLoader, file=sys.stdout)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs_origin = inputs
            inputs, targets = inputs.numpy(), targets
            ort_inputs = {"input.1": inputs}
            outputs = ort_session.run(None, ort_inputs)
            outputs = torch.from_numpy(outputs[0])

            predicted = utils.accuracy(outputs, targets, topk=topk)
            accuracy.update(predicted[0], inputs_origin.size(0))
            top5_accuracy.update(predicted[1], inputs_origin.size(0))

        current_time = time.time()
        logger.info(
            'Test Top1 {:.2f}%\tTop5 {:.2f}%\tTime {:.2f}s\n'
                .format(float(accuracy.avg), float(top5_accuracy.avg), (current_time - start_time))
        )

    return top5_accuracy.avg, accuracy.avg

def onnx_inference_imagenet(args):
    job_dir = './experiment'
    logger = utils.get_logger(os.path.join(job_dir + 'logger.log'))

    # Data
    print('==> Preparing data..')
    data_path = args.data
    loader = Data(data_path)
    testLoader = loader.loader_test

    onnx_path = args.onnx_path

    print(f"{onnx_path}")
    model = onnx.load_model(onnx_path)
    # 创建一个SessionOptions对象
    rtconfig = ort.SessionOptions()
    number_of_threads = 12
    rtconfig.intra_op_num_threads = number_of_threads
    # rtconfig.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    rtconfig.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    # 设置使用的ExecutionProvider为CPUExecutionProvider
    providers = ['CPUExecutionProvider']
    # 创建一个InferenceSession对象
    ort_session = ort.InferenceSession(model.SerializeToString(), providers=providers, sess_options=rtconfig)
    #---------------------------------------------------------#
    #   进test_onnxruntime函数
    #---------------------------------------------------------#
    test_onnxruntime(ort_session, testLoader, logger, topk=(1, 5))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-path", type=str, help="Path to the ONNX file")
    parser.add_argument("--data", type=str, help="Imagenet data path")
    args = parser.parse_args()
    onnx_inference_imagenet(args)
