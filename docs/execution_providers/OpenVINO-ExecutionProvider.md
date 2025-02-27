# OpenVINO Execution Provider

OpenVINO Execution Provider enables deep learning inference on Intel CPUs, Intel integrated GPUs and Intel<sup>®</sup> Movidius<sup>TM</sup> Vision Processing Units (VPUs). Please refer to [this](https://software.intel.com/en-us/openvino-toolkit/hardware) page for details on the Intel hardware supported.

## Build
For build instructions, please see the [BUILD page](../../BUILD.md#openvino).

## ONNX Layers supported using OpenVINO

The table below shows the ONNX layers supported using OpenVINO Execution Provider and the mapping between ONNX layers and OpenVINO layers. The below table also lists the Intel hardware support for each of the layers. CPU refers to Intel<sup>®</sup>
Atom, Core, and Xeon processors. GPU refers to the Intel Integrated Graphics. VPU refers to USB based Intel<sup>®</sup> Movidius<sup>TM</sup>
VPUs as well as Intel<sup>®</sup> Vision accelerator Design with Intel Movidius <sup>TM</sup> MyriadX VPU.

| **ONNX Layers** | **OpenVINO Layers** | **CPU** | **GPU** | **VPU** |
| --- | --- | --- | --- | --- |
| Add | Eltwise (operation=sum) | Yes | Yes | Yes
| AveragePool | Pooling(pool\_method=avg) | Yes | Yes | Yes
| BatchNormalization | Scaleshift (can be fused into Convlution or Fully Connected) | Yes | Yes | Yes
| Concat  | Concat | Yes | Yes | Yes
| Conv | Convolution | Yes | Yes | Yes
| Dropout | Ignored | Yes | Yes | Yes
| Flatten  | Reshape | Yes | Yes | Yes
| Gemm | FullyConnected | Yes | Yes | Yes
| GlobalAveragePool | Pooling | Yes | Yes | Yes
| Identity | Ignored | Yes | Yes | Yes
| ImageScaler | ScaleShift  | Yes  | Yes  | Yes
| LRN  | Norm | Yes | Yes | Yes
| MatMul | FullyConnected | Yes | Yes* | No
| MaxPool | Pooling(pool\_method=max) | Yes | Yes | Yes
| Mul | Eltwise(operation=mul) | Yes | Yes | Yes
| Relu |  ReLU  | Yes | Yes | Yes
| Reshape | Reshape | Yes | Yes | Yes
|  Softmax  | SoftMax | Yes | Yes | Yes
| Sum | Eltwise(operation=sum) | Yes | Yes | Yes
| Transpose | Permute | Yes | Yes | Yes
| UnSqueeze | Reshape  | Yes  | Yes  | Yes
| LeakyRelu | ReLU | Yes  | Yes  | Yes

*MatMul is supported in GPU only when the following layer is an Add layer in the topology.

## Topology Support

Below topologies are supported from ONNX open model zoo using OpenVINO Execution Provider

### Image Classification Networks

| **Topology** | **CPU** | **GPU** | **VPU** |
| --- | --- | --- | --- |
| bvlc\_alexnet | Yes | Yes | Yes
| bvlc\_googlenet | Yes | Yes | Yes
| bvlc\_reference\_caffenet | Yes | Yes | Yes
| bvlc\_reference\_rcnn\_ilsvrc13 | Yes | Yes | Yes
| densenet121 | Yes | Yes | Yes
| Inception\_v1 | Yes | Yes | Yes**
| Inception\_v2 | Yes | Yes | Yes
| Shufflenet | Yes | Yes | Yes
| Zfnet512 | Yes | Yes | Yes
| Squeeznet 1.1 | Yes | Yes | Yes
| Resnet18v1 | Yes | Yes | Yes
| Resnet34v1 | Yes | Yes | Yes
| Resnet50v1 | Yes | Yes | Yes
| Resnet101v1 | Yes | Yes | Yes
| Resnet152v1 | Yes | Yes | Yes
| Resnet18v2  | Yes | Yes | Yes
| Resnet34v2  | Yes | Yes | Yes
| Resnet50v2  | Yes | Yes | Yes
| Resnet101v2 | Yes | Yes | Yes
| Resnet152v2 | Yes | Yes | Yes
| Mobilenetv2 | Yes | Yes | Yes
| vgg16       | Yes | Yes | Yes
| vgg19       | Yes | Yes | Yes


### Image Recognition Networks

| **Topology** | **CPU** | **GPU** | **VPU** |
| --- | --- | --- | --- |
| MNIST | Yes | Yes | Yes**

**Inception_v1 and MNIST are supported in OpenVINO R1.1 and are not supported in OpenVINO R5.0.1.

### Object Detection Networks

| **Topology** | **CPU** | **GPU** | **VPU** |
| --- | --- | --- | --- |
|TinyYOLOv2 | Yes | Yes | Yes
| ResNet101\_DUC\_HDC | Yes | No | No

## Application code changes for VAD-M performance scaling

VAD-M has 8 VPUs and is suitable for applications that require multiple inferences to run in parallel. We use batching approach for performance scaling on VAD-M.

Below python code snippets provide sample classification code to batch input images, load a model and process the output results.

~~~
import onnxruntime as rt
from onnxruntime import get_device
import os
import os.path
import sys
import cv2
import numpy
import time
import glob
~~~
#### Load the input onnx model

~~~
sess = rt.InferenceSession(str(sys.argv[1]))
print("\n")
~~~

#### Preprocessing input images
~~~
for i in range(iters):
   y = None
   images = [cv2.imread(file) for file in glob.glob(str(sys.argv[2])+'/*.jpg')]
   for img in images:
     # resizing the image
     img = cv2.resize(img, (224,224))
     # convert image to numpy
     x = numpy.asarray(img).astype(numpy.float32)
     x = numpy.transpose(x, (2,0,1))
     # expand the dimension and batch the images
     x = numpy.expand_dims(x,axis=0)
     if y is None:
        y = x
     else:
        y = numpy.concatenate((y,x), axis=0)
~~~

#### Start Inference
~~~
   res = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name: y})
~~~
#### Post-processing output results
~~~
   print("Output probabilities:")
   i = 0
   for k in range(batch_size):
       for prob in res[0][k][0]:
          print("%d : %7.4f" % (i, prob))
~~~

