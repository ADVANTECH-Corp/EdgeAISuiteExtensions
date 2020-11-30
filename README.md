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
 

## Object Recognition Models

Object recognition models are used for classification, regression, and character
recognition. Use these networks after a respective detector (for example,
Age/Gender recognition after Face Detection).

| Model Name | Major detectable features | Source Framework  | Model size(MB)  | Complexity (GFLOPs) | CPU support | GPU support  | VPU (Myriad X) support | 
|------ |------ |------ |------ |------ |------ |------ |------ |
| [age-gender-recognition-retail-0013](./age-gender-recognition-retail-0013/description/age-gender-recognition-retail-0013.md)                                | 0.094                | 2.138      |
| [head-pose-estimation-adas-0001](./head-pose-estimation-adas-0001/description/head-pose-estimation-adas-0001.md)                                            | 0.105                | 1.911      |
| [license-plate-recognition-barrier-0001](./license-plate-recognition-barrier-0001/description/license-plate-recognition-barrier-0001.md)                    | 0.328                | 1.218      |
| [vehicle-attributes-recognition-barrier-0039](./vehicle-attributes-recognition-barrier-0039/description/vehicle-attributes-recognition-barrier-0039.md)     | 0.126                | 0.626      |
| [vehicle-attributes-recognition-barrier-0042](./vehicle-attributes-recognition-barrier-0042/description/vehicle-attributes-recognition-barrier-0042.md)     | 0.462                | 11.177     |
| [emotions-recognition-retail-0003](./emotions-recognition-retail-0003/description/emotions-recognition-retail-0003.md)                                      | 0.126                | 2.483      |
| [landmarks-regression-retail-0009](./landmarks-regression-retail-0009/description/landmarks-regression-retail-0009.md)                                      | 0.021                | 0.191      |
| [facial-landmarks-35-adas-0002](./facial-landmarks-35-adas-0002/description/facial-landmarks-35-adas-0002.md)                                               | 0.042                | 4.595      |
| [person-attributes-recognition-crossroad-0230](./person-attributes-recognition-crossroad-0230/description/person-attributes-recognition-crossroad-0230.md)  | 0.174                | 0.735      |
| [person-attributes-recognition-crossroad-0234](./person-attributes-recognition-crossroad-0234/description/person-attributes-recognition-crossroad-0234.md)  | 2.167                | 23.510     |
| [person-attributes-recognition-crossroad-0238](./person-attributes-recognition-crossroad-0238/description/person-attributes-recognition-crossroad-0238.md)  | 1.034                | 21.797     |
| [gaze-estimation-adas-0002](./gaze-estimation-adas-0002/description/gaze-estimation-adas-0002.md)                                                           | 0.139                | 1.882      |

## Reidentification Models

Precise tracking of objects in a video is a common application of Computer
Vision (for example, for people counting). It is often complicated by a set of
events that can be described as a "relatively long absence of an object". For
example, it can be caused by occlusion or out-of-frame movement. In such cases,
it is better to recognize the object as "seen before" regardless of its current
position in an image or the amount of time passed since last known position.

The following networks can be used in such scenarios. They take an image of a
person and evaluate an embedding - a vector in high-dimensional space that
represents an appearance of this person. This vector can be used for further
evaluation: images that correspond to the same person will have embedding
vectors that are "close" by L2 metric (Euclidean distance).

There are multiple models that provide various trade-offs between performance
and accuracy (expect a bigger model to perform better).

| Model Name | Major detectable features | Source Framework  | Model size(MB)  | Complexity (GFLOPs) | CPU support | GPU support  | VPU (Myriad X) support | 
|------ |------ |------ |------ |------ |------ |------ |------ |
| [person-reidentification-retail-0288](./person-reidentification-retail-0288/description/person-reidentification-retail-0288.md)   | 0.174                | 0.183      | 86.1%             |
| [person-reidentification-retail-0287](./person-reidentification-retail-0287/description/person-reidentification-retail-0287.md)   | 0.564                | 0.595      | 92.9%             |
| [person-reidentification-retail-0286](./person-reidentification-retail-0286/description/person-reidentification-retail-0286.md)   | 1.170                | 1.234      | 94.8%             |
| [person-reidentification-retail-0277](./person-reidentification-retail-0277/description/person-reidentification-retail-0277.md)   | 1.993                | 2.103      | 96.2%             |

## Semantic Segmentation Models

Semantic segmentation is an extension of object detection problem. Instead of
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


## Human Pose Estimation Models

Human pose estimation task is to predict a pose: body skeleton, which consists
of keypoints and connections between them, for every person in an input image or
video.  Keypoints are body joints, i.e. ears, eyes, nose, shoulders, knees, etc.
There are two major groups of such metods: top-down and bottom-up.  The first
detects persons in a given frame, crops or rescales detections, then runs pose
estimation network for every detection. These methods are very accurate. The
second finds all keypoints in a given frame, then groups them by person
instances, thus faster than previous, because network runs once.

| Model Name | Major detectable features | Source Framework  | Model size(MB)  | Complexity (GFLOPs) | CPU support | GPU support  | VPU (Myriad X) support | 
|------ |------ |------ |------ |------ |------ |------ |------ |
| [human-pose-estimation-0001](./human-pose-estimation-0001/description/human-pose-estimation-0001.md)                       | 15.435               | 4.099      |

## Image Processing

Deep Learning models find their application in various image processing tasks to
increase the quality of the output.

| Model Name | Major detectable features | Source Framework  | Model size(MB)  | Complexity (GFLOPs) | CPU support | GPU support  | VPU (Myriad X) support | 
|------ |------ |------ |------ |------ |------ |------ |------ |
| [single-image-super-resolution-1032](./single-image-super-resolution-1032/description/single-image-super-resolution-1032.md)                | 11.654               | 0.030      |
| [single-image-super-resolution-1033](./single-image-super-resolution-1033/description/single-image-super-resolution-1033.md)                | 16.062               | 0.030      |
| [text-image-super-resolution-0001](./text-image-super-resolution-0001/description/text-image-super-resolution-0001.md)                      | 1.379                | 0.003      |

## Text Detection

Deep Learning models for text detection in various applications.

| Model Name | Major detectable features | Source Framework  | Model size(MB)  | Complexity (GFLOPs) | CPU support | GPU support  | VPU (Myriad X) support | 
|------ |------ |------ |------ |------ |------ |------ |------ |
| [text-detection-0003](./text-detection-0003/description/text-detection-0003.md)                                                                | 51.256               | 6.747      |
| [text-detection-0004](./text-detection-0004/description/text-detection-0004.md)                                                                | 23.305               | 4.328      |
| [horizontal-text-detection-0001](./horizontal-text-detection-0001/description/horizontal-text-detection-0001.md)                               | 7.718              | 2.259     |

## Text Recognition

Deep Learning models for text recognition in various applications.

| Model Name | Major detectable features | Source Framework  | Model size(MB)  | Complexity (GFLOPs) | CPU support | GPU support  | VPU (Myriad X) support | 
|------ |------ |------ |------ |------ |------ |------ |------ |
| [text-recognition-0012](./text-recognition-0012/description/text-recognition-0012.md)                                                          | 1.485                | 5.568      |
| [handwritten-score-recognition-0003](./handwritten-score-recognition-0003/description/handwritten-score-recognition-0003.md)                   | 0.792                | 5.555      |
| [handwritten-japanese-recognition-0001](./handwritten-japanese-recognition-0001/description/handwritten-japanese-recognition-0001.md)          | 117.136              | 15.31      |
| [formula-recognition-medium-scan-0001](./formula-recognition-medium-scan-0001/description/formula-recognition-medium-scan-0001.md): <br> encoder <br> decoder | <br>16.56<br>1.86 | <br>1.69<br>2.56 |

## Text Spotting

Deep Learning models for text spotting (simultaneous detection and recognition).

| Model Name | Major detectable features | Source Framework  | Model size(MB)  | Complexity (GFLOPs) | CPU support | GPU support  | VPU (Myriad X) support | 
|------ |------ |------ |------ |------ |------ |------ |------ |
| [text-spotting-0002](./text-spotting-0002/description/text-spotting-0002.md)                                                                   |                      |            |
| text-spotting-0002-detector                                                                                                                    | 185.169              | 26.497     |
| text-spotting-0002-recognizer-encoder                                                                                                          | 2.082                | 1.328      |
| text-spotting-0002-recognizer-decoder                                                                                                          | 0.002                | 0.273      |

## Action Recognition Models

Action Recognition models predict action that is being performed on a short video clip
(tensor formed by stacking sampled frames from input video). Some models (for example `driver-action-recognition-adas-0002` may use precomputed high-level spatial
or spatio-temporal) features (embeddings) from individual clip fragments and then aggregate them in a temporal model
to predict a vector with classification scores. Models that compute embeddings are called *encoder*, while models
that predict an actual labels are called *decoder*.

| Model Name | Major detectable features | Source Framework  | Model size(MB)  | Complexity (GFLOPs) | CPU support | GPU support  | VPU (Myriad X) support | 
|------ |------ |------ |------ |------ |------ |------ |------ |
| [driver-action-recognition-adas-0002](./driver-action-recognition-adas-0002/description/driver-action-recognition-adas-0002.md)                         |                      |            |
| driver-action-recognition-adas-0002-encoder                                                                                                             | 0.676                | 2.863      |
| driver-action-recognition-adas-0002-decoder                                                                                                             | 0.147                | 4.205      |
| [action-recognition-0001](./action-recognition-0001/description/action-recognition-0001.md)                                                             |                      |            |
|   action-recognition-0001-encoder                                                                                                                       | 7.340                | 21.276     |
|   action-recognition-0001-decoder                                                                                                                       | 0.147                | 4.405      |
| [asl-recognition-0004](./asl-recognition-0004/description/asl-recognition-0004.md)                                                                      | 6.660                | 4.133      |
| [weld-porosity-detection-0001](./weld-porosity-detection-0001/description/weld-porosity-detection-0001.md)                                                                      | 3.636                | 11.173      |

## Image Retrieval

Deep Learning models for image retrieval (ranking 'gallery' images according to their similarity to some 'probe' image).

| Model Name | Major detectable features | Source Framework  | Model size(MB)  | Complexity (GFLOPs) | CPU support | GPU support  | VPU (Myriad X) support | 
|------ |------ |------ |------ |------ |------ |------ |------ |
| [image-retrieval-0001](./image-retrieval-0001/description/image-retrieval-0001.md)                                                          | 0.613                | 2.535      |

## Compressed models

Deep Learning compressed models

| Model Name | Major detectable features | Source Framework  | Model size(MB)  | Complexity (GFLOPs) | CPU support | GPU support  | VPU (Myriad X) support | 
|------ |------ |------ |------ |------ |------ |------ |------ |
| [resnet50-binary-0001](./resnet50-binary-0001/description/resnet50-binary-0001.md)                                                             | 1.002                | 7.446      |
| [resnet18-xnor-binary-onnx-0001](./resnet18-xnor-binary-onnx-0001/description/resnet18-xnor-binary-onnx-0001.md)                               | -                    | -          |

## Question Answering

| Model Name | Major detectable features | Source Framework  | Model size(MB)  | Complexity (GFLOPs) | CPU support | GPU support  | VPU (Myriad X) support | 
|------ |------ |------ |------ |------ |------ |------ |------ |
| [bert-large-uncased-whole-word-masking-squad-fp32-0001](./bert-large-uncased-whole-word-masking-squad-fp32-0001/description/bert-large-uncased-whole-word-masking-squad-fp32-0001.md) | 246.93 | 333.96 |
| [bert-large-uncased-whole-word-masking-squad-int8-0001](./bert-large-uncased-whole-word-masking-squad-int8-0001/description/bert-large-uncased-whole-word-masking-squad-int8-0001.md) | 246.93 | 333.96 |
| [bert-small-uncased-whole-word-masking-squad-0001](./bert-small-uncased-whole-word-masking-squad-0001/description/bert-small-uncased-whole-word-masking-squad-0001.md) | 23.9 | 57.94 |
| [bert-small-uncased-whole-word-masking-squad-0002](./bert-small-uncased-whole-word-masking-squad-0002/description/bert-small-uncased-whole-word-masking-squad-0002.md) | 23.9 | 41.1 |
| [bert-large-uncased-whole-word-masking-squad-emb-0001](./bert-large-uncased-whole-word-masking-squad-emb-0001/description/bert-large-uncased-whole-word-masking-squad-emb-0001.md) | 246.93 (for [1,384] input size) | 333.96 |

## Machine Translation

| Model Name | Major detectable features | Source Framework  | Model size(MB)  | Complexity (GFLOPs) | CPU support | GPU support  | VPU (Myriad X) support | 
|------ |------ |------ |------ |------ |------ |------ |------ |
| [machine-translation-nar-en-ru-0001](./machine-translation-nar-en-ru-0001/description/machine-translation-nar-en-ru-0001.md) | 23.17 | 69.29 |
| [machine-translation-nar-ru-en-0001](./machine-translation-nar-ru-en-0001/description/machine-translation-nar-ru-en-0001.md) | 23.17 | 69.29 |


## Legal Information
[*] Other names and brands may be claimed as the property of others.
