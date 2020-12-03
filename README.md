 
## How the Extensions Work


Here ADVANTECH provides various edge inferences in quick experiencing manners. Each sample/instance is encapsulated into a single *.zip file composed of IR model, inference engine app, a sample file as inference and a batch file. Each instance is executable through clicking the batch file without any coding effort. In addition, please remember to install Edge AI Suite in advance to launch each instance successfully.


## Object Detection Models

Several detection models can be used to detect a set of the most popular objects - for example, faces, people, vehicles. Most of the networks are SSD-based and provide reasonable accuracy/performance trade-offs. Networks that detect the same types of objects (for example, face-detection-adas-0001 and face-detection-retail-0004) provide a choice for higher accuracy/wider applicability at the cost of slower performance, so you can expect a "bigger" network to detect objects of the same type better. In addition, some specific recognition networks/models have also been integrated into inference pipeline after a respective detector (for example, Age/Gender recognition after Face Detection).


| Model Name | Major detectable features | Source Framework  | Model size(MB)  | Complexity (GFLOPs) | CPU support | GPU support  | VPU (Myriad X) support |        
|------ |------ |------ |------ |------ |------ |------ |------ |
| [face-detection-adas-0001](./Object-detection/face-detection-adas-0001.zip) | Face | Caffe | 1.053	 | 2.835 |V|V|V|
| [face-detection-retail-0004](./Object-detection/face-detection-retail-0004.zip) | Face | Caffe | 0.588	 | 1.067 |V|V|V|
| [face-detection-retail-0005](./Object-detection/face-detection-retail-0005.zip) | Face | PyTorch | 1.021 | 0.982 |V|V|V|
| [face-detection-retail-0044](./Object-detection/face-detection-retail-0044.zip) | Face |   |   |   |V|V|V|
| [faster_rcnn_inception_resnet_v2_atrous_coco](./Object-detection/faster_rcnn_inception_resnet_v2_atrous_coco.zip) | Object |TensorFlow|13.307|30.687|V|V| |
| [faster_rcnn_inception_v2_coco](./Object-detection/faster_rcnn_inception_v2_coco.zip) |Object| TensorFlow |13.307|30.687|V|V|V|
| [faster_rcnn_resnet50_coco](./Object-detection/faster_rcnn_resnet50_coco.zip)|Object|TensorFlow |29.162|57.203|V|V|V|
| [faster_rcnn_resnet101_coco](./Object-detection/faster_rcnn_resnet101_coco.zip)|Object|TensorFlow |48.128|112.052|V|V|V|
| [Gaze-Estimation](./Object-detection/Gaze-Estimation.zip)|| |||V|V|V|
| [mobilenet-ssd](./Object-detection/mobilenet-ssd.zip)|Object|Caffe |5.783|2.316|V|V|V|
| [pedestrian-and-vehicle-detector-adas-0001](./Object-detection/pedestrian-and-vehicle-detector-adas-0001.zip)|Person  Vehicle|Caffe|1.650|3.974|V|V|V|
| [pedestrian-detection-adas-0002](./Object-detection/pedestrian-detection-adas-0002.zip)|Person |Caffe|6.807|2.494|V|V|V|
| [person-detection-0200](./Object-detection/person-detection-0200.zip)|Person|PyTorch|1.817|0.786|V|V|V|
| [person-detection-0201](./Object-detection/person-detection-0201.zip)|Person|PyTorch|1.817|1.768|V|V|V|
| [person-detection-0202](./Object-detection/person-detection-0202.zip)|Person|PyTorch|1.817|3.143|V|V|V|
| [person-detection-retail-0002](./Object-detection/person-detection-retail-0002.zip)|Person|Caffe|3.244|12.427|V|V|V|
| [person-detection-retail-0002_person-reidentification-retail-0031](./Object-detection/person-detection-retail-0002_person-reidentification-retail-0031.zip)|Person| | | |V|V|V|
| [person-detection-retail-0013](./Object-detection/person-detection-retail-0013.zip)|Person|Caffe|0.723|2.300|V|V|V|
| [person-detection-retail-0013_person-reidentification-retail-0031](./Object-detection/person-detection-retail-0013_person-reidentification-retail-0031.zip)|Person| | | |V|V|V|
| [rfcn-resnet101-coco-tf](./Object-detection/rfcn-resnet101-coco-tf.zip)|Object |TensorFlow|171.85|53.462|V|V|V|
| [ssd_mobilenet_v1_coco](./Object-detection/ssd_mobilenet_v1_coco.zip)|Object|TensorFlow|6.807|2.494|V|V|V|
| [ssd_mobilenet_v1_fpn_coco](./Object-detection/ssd_mobilenet_v1_fpn_coco.zip)|Object|TensorFlow|36.188|123.309|V|V|V|
| [ssd_mobilenet_v2_coco](./Object-detection/ssd_mobilenet_v2_coco.zip)|Object|TensorFlow|16.818|3.775|V|V|V|
| [ssd_resnet50_v1_fpn_coco](./Object-detection/ssd_resnet50_v1_fpn_coco.zip)|Object|TensorFlow|56.9326|178.6807|V|V|V|
| [ssd300](./Object-detection/ssd300.zip)|Object|Caffe|26.285|62.815|V|V|V|
| [ssd512](./Object-detection/ssd512.zip)|Object|Caffe|27.189|180.611|V|V|V|
| [ssdlite_mobilenet_v2](./Object-detection/ssdlite_mobilenet_v2.zip)|Object|TensorFlow|4.475|1.525|V|V|V|
| [vehicle-detection-0200](./Object-detection/vehicle-detection-0200.zip)|Vehicle|PyTorch|1.817|0.786|V|V|V|
| [vehicle-detection-0201](./Object-detection/vehicle-detection-0201.zip)|Vehicle|PyTorch|1.817|1.768|V|V|V|
| [vehicle-detection-0202](./Object-detection/vehicle-detection-0202.zip)|Vehicle|PyTorch|1.817|3.143|V|V|V|
| [vehicle-license 1](./Object-detection/vehicle-license1.zip)|Vehicle| | | |V|V|V|
| [vehicle-license 2](./Object-detection/vehicle-license2.zip)|Vehicle| | | |V|V|V|
| [vehicle-license-plate-detection-barrier-0106](./Object-detection/vehicle-license-plate-detection-barrier-0106.zip)|Vehicle License Plate|TensorFlow|0.634|0.349|V|V|V|
| [yolo-v3-tf](./Object-detection/yolo-v3-tf.zip)|Object|Keras|61.922|65.984|V|V|V|
| [yolo-v3-tiny-tf](./Object-detection/yolo-v3-tiny-tf.zip)|Object|TensorFlow|15.858|6.988|V|V|V|
 
## Classification Models

It is a systematic grouping of observations into categories, such as when biologists categorize plants, animals, and other lifeforms into different taxonomies.

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
|[googlenet-v1-tf](./Classification/googlenet-v1-tf.zip)|Classification|TensorFlow|6.619|3.016|V|V|V|
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

Semantic segmentation is an extension of object detection problem. Instead of returning bounding boxes, semantic segmentation models return a "painted" version of the input image, where the "color" of each pixel represents a certain class. These networks are much bigger than respective object detection networks, but they provide a better (pixel-level) localization of objects and they can detect areas with complex shape.

Segmentation is an extension of object detection problem. Instead of
returning bounding boxes, semantic segmentation models return a "painted"
version of the input image, where the "color" of each pixel represents a certain
class. These networks are much bigger than respective object detection networks,
but they provide a better (pixel-level) localization of objects and they can
detect areas with complex shape (for example, free space on the road).

| Model Name | Major detectable features | Source Framework  | Model size(MB)  | Complexity (GFLOPs) | CPU support | GPU support  | VPU (Myriad X) support | 
|------ |------ |------ |------ |------ |------ |------ |------ |
|[deeplabv3](./Segmentation/deeplabv3.zip)|Segmentation|TensorFlow|23.819|11.469|V|V||
|[icnet-camvid-ava-0001](./Segmentation/icnet-camvid-ava-0001.zip)|Segmentation|TensorFlow|25.45|151.82|V|V||
|[icnet-camvid-ava-sparse-30-0001](./Segmentation/icnet-camvid-ava-sparse-30-0001.zip)|Segmentation|TensorFlow|25.45|151.82|V|V||
|[icnet-camvid-ava-sparse-60-0001](./Segmentation/icnet-camvid-ava-sparse-60-0001.zip)|Segmentation|TensorFlow|25.45|151.82|V|V||
|[midasnet](./Segmentation/midasnet.zip)|Segmentation|PyTorch|104.0814|207.4915|V|V|V|
|[road-segmentation-adas-0001](./Segmentation/road-segmentation-adas-0001.zip)|Segmentation|PyTorch|0.184|4.770|V|V|V|
|[semantic-segmentation-adas-0001](./Segmentation/semantic-segmentation-adas-0001.zip)|Segmentation|Caffe|6.686|58.572|V|V||
|[unet-camvid-onnx-0001](./Segmentation/unet-camvid-onnx-0001.zip)|Segmentation|PyTorch|31.03|260.1|V|V|V|

 
## Instance Segmentation Models

Instance segmentation is an extension of object detection and semantic segmentation problems. Instead of predicting a bounding box around each object instance, instance segmentation model outputs pixel-wise masks for all instances.

| Model Name | Major detectable features | Source Framework  | Model size(MB)  | Complexity (GFLOPs) | CPU support | GPU support  | VPU (Myriad X) support | 
|------ |------ |------ |------ |------ |------ |------ |------ |
|[mask_rcnn_inception_resnet_v2_atrous_coco](./Instance-segmentation/mask_rcnn_inception_resnet_v2_atrous_coco.zip)|Segmentation|TensorFlow|92.368|675.314|V|V||
|[mask_rcnn_resnet101_atrous_coco](./Instance-segmentation/mask_rcnn_resnet101_atrous_coco.zip)|Segmentation|TensorFlow|69.188|674.58|V|V||
|[mask_rcnn_inception_v2_coco](./Instance-segmentation/mask_rcnn_inception_v2_coco.zip)|Segmentation|TensorFlow|21.772|54.926|V|V|V|
|[mask_rcnn_resnet50_atrous_coco](./Instance-segmentation/mask_rcnn_resnet50_atrous_coco.zip)|Segmentation|TensorFlow|50.222|294.738|V|V|V|


 
## Text Detection

Various applications for text detection can run with additional deep Learning models.

| Model Name | Major detectable features | Source Framework  | Model size(MB)  | Complexity (GFLOPs) | CPU support | GPU support  | VPU (Myriad X) support | 
|------ |------ |------ |------ |------ |------ |------ |------ |
|[horizontal-text-detection-0001](./Text-detection/horizontal-text-detection-0001.zip)|Text|PyTorch|2.26|7.78|V|V||
|[text-detection1](./Text-detection/text-detection1.zip)|Text| | | |V|V|V|
|[text-detection-0003](./Text-detection/text-detection-0003.zip)|Text|TensorFlow|6.747|51.256|V|V|V|
|[text-detection-0004](./Text-detection/text-detection-0004.zip)|Text|TensorFlow|4.328|23.305|V|V|V|
 

## Reference Links

- [https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/index.md](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/index.md) 
- [https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/index.md](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/index.md) 
- [https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/README.md#demos-that-support-pre-trained-models](https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/README.md#demos-that-support-pre-trained-models)  


## Legal Information
 
These provisions are licensed under [Apache License Version 2.0](https://github.com/openvinotoolkit/openvino/blob/master/LICENSE). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.