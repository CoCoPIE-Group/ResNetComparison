# Introduction

This repository includes instructions on how to download and test on CPU the speed and accuracy of the ResNet models optimized by CoCoPIE. The base framework is OpenVINO. 

*All copyright of the optimized ResNet models (resnet50_78top1, resnet50_76top1, resnet50_75top1) in referenced in this repository belongs to CoCoPIE Inc. No redistribution or commercial use without the written consent by CoCoPIE Inc.*

# Results


**Testing Platform:**
```
Distributor ID: Ubuntu
Description: Ubuntu 20.04.6 LTS
Release: 20.04
Codename: focal
```

**CPU:**
```
12th Gen Intel(R) Core(TM) i7-12700K
```

**Models**

  Baselines: resnet50-origin, resnet101_origin
  
  Optimized by CoCoPIE: resnet50_78top1, resnet50_76top1, resnet50_75top1

**Dataset**

[ImageNet](https://www.image-net.org/download.php)

**Speed and Accuracy**

(Batch size: 1)

| Model             | Accuracy   | Median Latency (ms) | Average Latency (ms) | Min Latency (ms) | Max Latency (ms) | FPS    | FLOPs (G)   | Params (M)    |
|-------------------|------------|---------------------|----------------------|------------------|------------------|--------|-------------|---------------|
| resnet50-origin   | 76.15%     | 9.57                | 10.29                | 9.22             | 28.45            | 96.26  | 4.11 | 25.55    |
| resnet101_origin  | 77.37%     | 17.92               | 18.65                | 17.37            | 30.93            | 53.35  | 7.83  | 44.55      |
| **resnet50_78top1**   | 78.09%     | 8.51                | 8.91                 | 8.18             | 18.25            | 111.17 | 3.38   | 21.81     |
| **resnet50_76top1**   | 76.24%     | 4.07                | 4.28                 | 3.82             | 12.17            | 228.85 | 1.41  | 11.14     |
| **resnet50_75top1**   | 75.17%     | 3.53                | 3.71                 | 3.26             | 10.87            | 263.25 | 1.21 | 9.10      |


## Instructions for Reproducing the Results

0. Download the [ResNet models](https://cocopieai-my.sharepoint.com/:f:/g/personal/xipeng_shen_cocopie_ai/Ep7pcuZ4PGVPnJVWzq_M95IB84CIHP7QdThLKIpYZK8x4g?email=jay.liu%40cocopie.ai&e=90nOPj)

1. Install OpenVINO by referring to the official installation guide: [Installing OpenVINO with pip](https://docs.openvino.ai/2023.1/openvino_docs_install_guides_installing_openvino_pip.html).

2. Convert the ONNX models to the OpenVINO format. Run the following command:
   ```
   python convert_onnx2openvino.py --onnx-path <your onnx path> --output-path <your output openvino file path>
   ```
   Replace `<your onnx path>` with the path to your ONNX model, and `<your output openvino file path>` with the desired output path for the converted model.

3. Measure the model's performance using OpenVINO by running command `benchmark_app` offered by OpenVINO. 
   ```
   benchmark_app -m <your openvino>.xml -d CPU -hint latency
   ```
   Replace `<your openvino>` with the path to your OpenVINO model XML file.

   Please refer to the official OpenVINO documentation for more detailed instructions and troubleshooting.

4. Test the accuracy of the ONNX model. Download [ImageNet](https://www.image-net.org/download.php) if you haven't. Run the following command:

   ```
   python inference_onnx.py --onnx-path <path_to_onnx_model> --data <path_to_data>
   ```

   Note: The `inference_onnx.py` script uses the ONNX Runtime (rather than OpenVINO) with CPU support, and the number of threads is set to 12.


CoCoPIE
10/2023
