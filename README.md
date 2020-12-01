/ # Overview of OpenVINO&trade; Toolkit Intel's Pre-Trained Models

OpenVINO&trade; toolkit provides a set of pre-trained models
that you can use for learning and demo purposes or for developing deep learning
software. Most recent version is available in the [repo on Github](https://github.com/opencv/open_model_zoo).

The models can be downloaded via Model Downloader
(`<OPENVINO_INSTALL_DIR>/deployment_tools/open_model_zoo/tools/downloader`).
They can also be downloaded manually from [01.org](https://download.01.org/opencv).

## Object Detection Models

Several detection models can be used to detect a set of the most popular
objects - for example, faces, people, vehicles. Most of the networks are
SSD-based and provide reasonable accuracy/performance trade-offs. Networks that
detect the same types of objects (for example, `face-detection-adas-0001` and
`face-detection-retail-0004`) provide a choice for higher accuracy/wider
applicability at the cost of slower performance, so you can expect a "bigger"
network to detect objects of the same type better.

| Model Name | Major detectable features | Source Framework  | Model size(MB)  | Complexity (GFLOPs) | CPU support | GPU support  | VPU (Myriad X) support |        
|------ |------ |------ |------ |------ |------ |------ |------ |
| [face-detection-adas-0001](./Object-detection/face-detection-adas-0001.zip) | Face | Caffe | 1.053	 | 2.835 |V|V|V|
| [face-detection-retail-0004](./Object-Detection/face-detection-adas-0004.zip) | Face | Caffe | 0.588	 | 1.067 |V|V|V|
| [face-detection-retail-0005](./Object-Detection/face-detection-adas-0005.zip) | Face | PyTorch | 1.021 | 0.982 |V|V|V|
| [face-detection-retail-0044](./Object-Detection/face-detection-adas-0044.zip) | Face |   |   |   |V|V|V|
| [faster_rcnn_inception_resnet_v2_atrous_coco](./Object-Detection/faster_rcnn_inception_resnet_v2_atrous_coco.zip) | Object |TensorFlow|13.307|30.687|V|V| |
| [faster_rcnn_inception_v2_coco](./Object-Detection/faster_rcnn_inception_v2_coco.zip) |Object| TensorFlow |13.307|30.687|V|V|V|
| [faster_rcnn_resnet50_coco](./Object-Detection/faster_rcnn_resnet50_coco.zip)|Object|TensorFlow |29.162|57.203|V|V|V|
| [faster_rcnn_resnet101_coco](./Object-Detection/faster_rcnn_resnet101_coco.zip)|Object|TensorFlow |48.128|112.052|V|V|V|
| [Gaze-Estimation](./Object-Detection/Gaze-Estimation.zip)|| |||V|V|V|
| [mobilenet-ssd](./Object-Detection/mobilenet-ssd.zip)|Object|Caffe |5.783|2.316|V|V|V|
| [pedestrian-and-vehicle-detector-adas-0001](./Object-Detection/pedestrian-and-vehicle-detector-adas-0001.zip)|Person  Vehicle|Caffe|1.650|3.974|V|V|V|
| [pedestrian-detection-adas-0002](./Object-Detection/pedestrian-detection-adas-0002.zip)|Person |Caffe|6.807|2.494|V|V|V|
| [person-detection-0200](./Object-Detection/person-detection-0200.zip)|Person|PyTorch|1.817|0.786|V|V|V|
| [person-detection-0201](./Object-Detection/person-detection-0201.zip)|Person|PyTorch|1.817|1.768|V|V|V|
| [person-detection-0202](./Object-Detection/person-detection-0202.zip)|Person|PyTorch|1.817|3.143|V|V|V|
| [person-detection-retail-0002](./Object-Detection/person-detection-retail-0002.zip)|Person|Caffe|3.244|12.427|V|V|V|
| [person-detection-retail-0002_person-reidentification-retail-0031](./Object-Detection/person-detection-retail-0002_person-reidentification-retail-0031.zip)|Person| | | |V|V|V|
| [person-detection-retail-0013](./Object-Detection/person-detection-retail-0013.zip)|Person|Caffe|0.723|2.300|V|V|V|
| [person-detection-retail-0013_person-reidentification-retail-0031](./Object-Detection/person-detection-retail-0013_person-reidentification-retail-0031.zip)|Person| | | |V|V|V|
| [rfcn-resnet101-coco-tf](./Object-Detection/rfcn-resnet101-coco-tf.zip)|Object |TensorFlow|171.85|53.462|V|V|V|
| [ssd_mobilenet_v1_coco](./Object-Detection/ssd_mobilenet_v1_coco.zip)|Object|TensorFlow|6.807|2.494|V|V|V|
| [ssd_mobilenet_v1_fpn_coco](./Object-Detection/ssd_mobilenet_v1_fpn_coco.zip)|Object|TensorFlow|36.188|123.309|V|V|V|
| [ssd_mobilenet_v2_coco](./Object-Detection/ssd_mobilenet_v2_coco.zip)|Object|TensorFlow|16.818|3.775|V|V|V|
| [ssd_resnet50_v1_fpn_coco](./Object-Detection/ssd_resnet50_v1_fpn_coco.zip)|Object|TensorFlow|56.9326|178.6807|V|V|V|
| [ssd300](./Object-Detection/ssd300.zip)|Object|Caffe|26.285|62.815|V|V|V|
| [ssd512](./Object-Detection/ssd512.zip)|Object|Caffe|27.189|180.611|V|V|V|
| [ssdlite_mobilenet_v2](./Object-Detection/ssdlite_mobilenet_v2.zip)|Object|TensorFlow|4.475|1.525|V|V|V|
| [vehicle-detection-0200](./Object-Detection/vehicle-detection-0200.zip)|Vehicle|PyTorch|1.817|0.786|V|V|V|
| [vehicle-detection-0201](./Object-Detection/vehicle-detection-0201.zip)|Vehicle|PyTorch|1.817|1.768|V|V|V|
| [vehicle-detection-0202](./Object-Detection/vehicle-detection-0202.zip)|Vehicle|PyTorch|1.817|3.143|V|V|V|
| [vehicle-license 1](./Object-Detection/vehicle-license1.zip)|Vehicle| | | |V|V|V|
| [vehicle-license 2](./Object-Detection/vehicle-license2.zip)|Vehicle| | | |V|V|V|
| [vehicle-license-plate-detection-barrier-0106](./Object-Detection/vehicle-license-plate-detection-barrier-0106.zip)|Vehicle License Plate|TensorFlow|0.634|0.349|V|V|V|
| [yolo-v3-tf](./Object-Detection/yolo-v3-tf.zip)|Object|Keras|61.922|65.984|V|V|V|
| [yolo-v3-tiny-tf](./Object-Detection/yolo-v3-tiny-tf.zip)|Object|TensorFlow|15.858|6.988|V|V|V|
 
## Classification Models
| Model Name | Major detectable features | Source Framework  | Model size(MB)  | Complexity (GFLOPs) | CPU support | GPU support  | VPU (Myriad X) support |        
|------ |------ |------ |------ |------ |------ |------ |------ |
|[alexnet](./Classification/alexnet.zip)|Classification|Caffe|60.965|1.5|V|V|V|
|[caffenet](./Classification/caffenet.zip)|Classification|Caffe|60.965|1.463|V|V|V|
|[densenet-121](./Classification/densenet-121.zip)|Classification|Caffe|7.971|5.724|V|V|V|
|[densenet-121-tf](./Classification/densenet-121-tf.zip)|Classification|TensorFlow|7.971|5.289|V|V|V|
|[densenet-161](./Classification/densenet-161.zip)|Classification|Caffe|28.666|15.561|V|V|V|
|[densenet-161-tf](./Classification/densenet-161-tf.zip)|Classification|TensorFlow|28.666|14.128|V|V|V|
|[densenet-169](./Classification/densenet-169.zip)|Classification|Caffe|14.139|6.788|V|V|V|
|[densenet-169-tf](./Classification/densenet-169-tf.zip)|Classification|TensorFlow|14.139|6.16|V|V|V|
|[efficientnet-b0](./Classification/efficientnet-b0.zip)|Classification|TensorFlow|5.268|0.189|V|V|V|
|[efficientnet-b0_auto_aug](./Classification/efficientnet-b0_auto_aug.zip)|Classification|TensorFlow|5.268|0.819|V|V|V|
|[efficientnet-b0-pytorch](./Classification/efficientnet-b0-pytorch.zip)|Classification|PyTorch|5.268|0.819|V|V|V|
|[efficientnet-b5](./Classification/efficientnet-b5.zip)|Classification|TensorFlow|30.303|21.252|V|V|V|
|[efficientnet-b5-pytorch](./Classification/efficientnet-b5-pytorch.zip)|Classification|PyTorch|30.303|21.252|V|V|V|
|[efficientnet-b7_auto_aug](./Classification/efficientnet-b7_auto_aug.zip)|Classification|TensorFlow|66.193|77.168|V|V|V|
|[efficientnet-b7-pytorch](./Classification/efficientnet-b7-pytorch.zip)|Classification|PyTorch|66.193|77.168|V|V|V|
|[googlenet-v1](./Classification/googlenet-v1.zip)|Classification|Caffe|6.999|3.266|V|V|V|
|[googlenet-v1-tf](./Classification/googlenet-v1.zip)|Classification|TensorFlow|6.619|3.016|V|V|V|
|[googlenet-v2](./Classification/googlenet-v2.zip)|Classification|Caffe|11.185|4.058|V|V|V|
|[googlenet-v2-tf](./Classification/googlenet-v2-tf.zip)|Classification|TensorFlow|11.185|4.058|V|V|V|
|[googlenet-v3](./Classification/googlenet-v3.zip)|Classification|TensorFlow|23.819|11.469|V|V|V|
|[googlenet-v3-pytorch](./Classification/googlenet-v3-pytorch.zip)|Classification|PyTorch|23.817|11.469|V|V|V|
|[googlenet-v4-tf](./Classification/googlenet-v4-tf.zip)|Classification|TensorFlow|42.648|24.584|V|V|V|
|[hbonet-0.5](./Classification/hbonet-0.5.zip)|Classification|PyTorch|2.5289|0.096|V|V|V|
|[hbonet-0.25](./Classification/hbonet-0.25.zip)|Classification|PyTorch|1.9300|0.037|V|V|V|
|[hbonet-1.0](./Classification/hbonet-1.0.zip)|Classification|PyTorch|4.5447|0.305|V|V|V|
|[inception-resnet-v2-tf](./Classification/inception-resnet-v2-tf.zip)|Classification|TensorFlow|30.223|22.227|V|V|V|
|[mobilenet-v2-pytorch](./Classification/mobilenet-v2-pytorch.zip)|Classification|PyTorch|3.489|0.615|V|V|V|
|[octave-densenet-121-0.125](./Classification/octave-densenet-121-0.125.zip)|Classification|MXNet|7.977|4.883|V|V|V|
|[octave-resnet-26-0.25](./Classification/octave-resnet-26-0.25.zip)|Classification|MXNet|15.99|3.768|V|V|V|
|[octave-resnet-50-0.125](./Classification/octave-resnet-50-0.125.zip)|Classification|MXNet|25.551|7.221|V|V|V|
|[octave-resnet-101-0.125](./Classification/octave-resnet-101-0.125.zip)|Classification|MXNet|44.543|13.387|V|V|V|
|[octave-resnet-200-0.125](./Classification/octave-resnet-200-0.125.zip)|Classification|MXNet|64.667|25.407|V|V|V|
|[octave-resnext-50-0.25](./Classification/octave-resnext-50-0.25.zip)|Classification|MXNet|25.02|6.444|V|V|V|
|[octave-resnext-101-0.25](./Classification/octave-resnext-101-0.25.zip)|Classification|MXNet|44.169|11.521|V|V|V|
|[octave-se-resnet-50-0.125](./Classification/octave-se-resnet-50-0.125.zip)|Classification|MXNet|28.082|7.246|V|V|V|
|[resnest-50-pytorch](./Classification/resnest-50-pytorch.zip)|Classification|PyTorch|27.4493|10.8148|V|V|V|
|[resnet-18-pytorch](./Classification/resnet-18-pytorch.zip)|Classification|PyTorch|11.68|3.637|V|V|V|
|[resnet-34-pytorch](./Classification/resnet-34-pytorch.zip)|Classification|PyTorch|21.7892|7.3409|V|V|V|
|[resnet-50-pytorch](./Classification/resnet-50-pytorch.zip)|Classification|PyTorch|25.53|8.216|V|V|V|
|[resnet-50-tf](./Classification/resnet-50-tf.zip)|Classification|TensorFlow|25.53|8.2164|V|V|V|
|[se-inception](./Classification/se-inception.zip)|Classification|TensorFlow|11.922|4.091|V|V|V|
|[se-resnet-50](./Classification/se-resnet-50.zip)|Classification|Caffe|28.061|7.775|V|V|V|
|[se-resnet-101](./Classification/se-resnet-101.zip)|Classification|Caffe|49.274|15.239|V|V|V|
|[se-resnet-152](./Classification/se-resnet-152.zip)|Classification|Caffe|66.746|22.709|V|V|V|
|[se-resnext-50](./Classification/se-resnext-50.zip)|Classification|Caffe|27.526|8.533|V|V|V|
|[se-resnext-101](./Classification/se-resnext-101.zip)|Classification|Caffe|48.886|16.054|V|V|V|
|[squeezenet1.0](./Classification/squeezenet1.0.zip)|Classification|Caffe|1.248|1.737|V|V|V|
|[squeezenet1.1](./Classification/squeezenet1.1.zip)|Classification|Caffe|1.236|0.785|V|V|V|
|[vgg16](./Classification/vgg16.zip)|Classification|Caffe|138.358|30.974|V|V|V|
|[vgg19](./Classification/vgg19.zip)|Classification|Caffe|39.3|143.667|V|V|V|

 


## Segmentation Models

Segmentation is an extension of object detection problem. Instead of
returning bounding boxes, semantic segmentation models return a "painted"
version of the input image, where the "color" of each pixel represents a certain
class. These networks are much bigger than respective object detection networks,
but they provide a better (pixel-level) localization of objects and they can
detect areas with complex shape (for example, free space on the road).

| Model Name | Major detectable features | Source Framework  | Model size(MB)  | Complexity (GFLOPs) | CPU support | GPU support  | VPU (Myriad X) support | 
|------ |------ |------ |------ |------ |------ |------ |------ |
| [road-segmentation-adas-0001](./road-segmentation-adas-0001/description/road-segmentation-adas-0001.md)                                        | 4.770                | 0.184      |
| [semantic-segmentation-adas-0001](./semantic-segmentation-adas-0001/description/semantic-segmentation-adas-0001.md)                            | 58.572               | 6.686      |
| [unet-camvid-onnx-0001](./unet-camvid-onnx-0001/description/unet-camvid-onnx-0001.md)                                                          | 260.1                | 31.03      |
| [icnet-camvid-ava-0001](./icnet-camvid-ava-0001/description/icnet-camvid-ava-0001.md)                                                   | 151.82                | 25.45       |
| [icnet-camvid-ava-sparse-30-0001](./icnet-camvid-ava-sparse-30-0001/description/icnet-camvid-ava-sparse-30-0001.md)                     | 151.82                | 25.45       |
| [icnet-camvid-ava-sparse-60-0001](./icnet-camvid-ava-sparse-60-0001/description/icnet-camvid-ava-sparse-60-0001.md)                     | 151.82                | 25.45       |

## Instance Segmentation Models

Instance segmentation is an extension of object detection and semantic
segmentation problems. Instead of predicting a bounding box around each object
instance instance segmentation model outputs pixel-wise masks for all instances.

| Model Name | Major detectable features | Source Framework  | Model size(MB)  | Complexity (GFLOPs) | CPU support | GPU support  | VPU (Myriad X) support | 
|------ |------ |------ |------ |------ |------ |------ |------ |
| [instance-segmentation-security-1025](./instance-segmentation-security-1025/description/instance-segmentation-security-1025.md)                | 30.146               | 26.69      |
| [instance-segmentation-security-0050](./instance-segmentation-security-0050/description/instance-segmentation-security-0050.md)                | 46.602               | 30.448     |
| [instance-segmentation-security-0083](./instance-segmentation-security-0083/description/instance-segmentation-security-0083.md)                | 365.626              | 143.444    |
| [instance-segmentation-security-0010](./instance-segmentation-security-0010/description/instance-segmentation-security-0010.md)                | 899.568              | 174.568    |

 
## Text Detection

Deep Learning models for text detection in various applications.

| Model Name | Major detectable features | Source Framework  | Model size(MB)  | Complexity (GFLOPs) | CPU support | GPU support  | VPU (Myriad X) support | 
|------ |------ |------ |------ |------ |------ |------ |------ |
| [text-detection-0003](./text-detection-0003/description/text-detection-0003.md)                                                                | 51.256               | 6.747      |
| [text-detection-0004](./text-detection-0004/description/text-detection-0004.md)                                                                | 23.305               | 4.328      |
| [horizontal-text-detection-0001](./horizontal-text-detection-0001/description/horizontal-text-detection-0001.md)                               | 7.718              | 2.259     |

 

 


## Legal Information
[*] Other names and brands may be claimed as the property of others.
