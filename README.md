## How the Extensions Work

Here ADVANTECH provides various edge inferences in quick experiencing manners. Each sample/instance is encapsulated into a single *.zip file composed of IR model, inference engine app, a sample file as inference and a batch file. Each instance is executable through clicking the batch file without any coding effort. In addition, please remember to install Edge AI Suite in advance to launch each instance successfully. More precisely, all samples need to run the environment installed with Edge AI Suite V1.2.2 based on OpenVINO 2021.1.
 
## How to Leverage Downloaded Inferences
Each downloaded inference is in form of single *.zip file. After extracting it, you just need to click the *.bat file therein to launch an inference app. In addition, you can also refer to the following table according to its inference execution. With the following table, we can find more experiencing manners for various inferences. The way to realize these is to follow the mentioned commands in the table and edit the extracted *.bat file from the zip file.

 
|Inference Execution|-d cmd for inference device|-i cmd for inference source|
|--------- |---------------------|---------|  
|classification_demo.exe|Ex: <br> -d CPU  <br> -d GPU <br>-d HDDL|Path to a folder with images or path to an image file. <br> Ex : File: -i image.png|
|object_detection_demo_ssd_async.exe|Ex: <br> -d CPU  <br> -d GPU <br>-d HDDL|Path to a video file (specify "cam" to work with camera). <br> Ex: <br> Webcam : -i cam 0 <br>  File: -i video.mp4|   
|pedestrian_tracker_demo.exe|Ex: <br> -d_det CPU  <br> -d_det GPU <br>-d_det HDDL|An input to process. The input must be a single image, a folder of images or anything that cv::VideoCapture can process.  <br>Ex:  <br> Webcam  : -i 0  <br> File:  -i video.mp4|
|security_barrier_camera_demo.exe|Ex: <br> -d CPU  <br> -d GPU <br>-d HDDL| Required for video or image files input. Path to video or image files. [Note] -nc:  Required for web camera input. Maximum number of processed camera inputs (web cameras). <br> Ex: <br> Webcam  : -nc 1  <br> File:  -i video.mp4|
|text_detection_demo.exe|Ex: <br> <pre> -d_td CPU  <br> -d_td GPU <br> -d_td HDDL </pre>|An input to process. The input must be a single image, a folder of images or anything that cv::VideoCapture can process.  <br>Ex:  <br> Webcam  : -i 0  <br> File:  -i video.mp4|
|segmentation_demo.exe|Ex:<br> -d CPU  <br> -d GPU <br>-d HDDL|An input to process. The input must be a single image, a folder of images or anything that cv::VideoCapture can process.  <br>Ex:  <br> Webcam  : -i 0  <br> File:  -i video.mp4|
|gaze_estimation_demo.exe|Ex:<br> -d CPU <br> -d GPU <br> -d HDDL|An input to process. The input must be a single image, a folder of images or anything that cv::VideoCapture can process.  <br>Ex:  <br> Webcam  : -i 0  <br> File:  -i video.mp4|
|interactive_face_detection_demo.exe|Ex: <br> -d CPU  <br> -d GPU <br>-d HDDL|An input to process. The input must be a single image, a folder of images or anything that cv::VideoCapture can process.  <br>Ex:  <br> Webcam  : -i 0  <br> File:  -i video.mp4|
|mask_rcnn_demo.exe|Ex: <br> -d CPU  <br> -d GPU <br>-d HDDL|Path to an image(jpg/bmp/png).<br> Ex : <br> File: -i image.jpg|
|object_detection_demo_yolov3_async.exe|Ex: <br> -d CPU  <br> -d GPU <br>-d HDDL|An input to process. The input must be a single image, a folder of images or anything that cv::VideoCapture can process.  <br>Ex:  <br> Webcam  : -i 0  <br> File:  -i video.mp4|

## Post-trained Object Detection Inferences from Advantech Co., Ltd.

Here are samples provided by ADVANTECH to show the capabilities to retrain public pre-trained models for specific purposes. For example, the model of “factory-operation detect” is based on ssd_mobilenet_v2 and further becomes familiar with the target components and operations.


| Model Name | Major detectable features | Source Framework  | Model size(MB)  | Complexity (GFLOPs) | CPU support | GPU support  | VPU (Myriad X) support |        
|------ |------ |------ |------ |------ |------ |------ |------ |
| [factory-operation detect](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Advantech/factory-operation.zip) | Object |TensorFlow|9.08|3.775|V|V|V|
| [unexpected-window detect](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Advantech/unexpected-window.zip) | Object |TensorFlow|24.7|30.687 |V|V|V|


## Object Detection Inferences

Several detection models can be used to detect a set of the most popular objects - for example, faces, people, vehicles. Most of the networks are SSD-based and provide reasonable accuracy/performance trade-offs. Networks that detect the same types of objects (for example, face-detection-adas-0001 and face-detection-retail-0004) provide a choice for higher accuracy/wider applicability at the cost of slower performance, so you can expect a "bigger" network to detect objects of the same type better. In addition, some specific recognition networks/models have also been integrated into inference pipeline after a respective detector (for example, Age/Gender recognition after Face Detection).

| Model Name | Major detectable features | Source Framework  | Model size(MB)  | Complexity (GFLOPs) | CPU support | GPU support  | VPU (Myriad X) support |        
|------ |------ |------ |------ |------ |------ |------ |------ |
| [face-detection-adas-0001](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/face-detection-adas-0001.zip) | Face | Caffe |2.22|2.835|V|V|V|
| [face-detection-retail-0004](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/face-detection-retail-0004.zip) | Face | Caffe |1.21|1.067|V|V|V|
| [face-detection-retail-0005](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/face-detection-retail-0005.zip) | Face | PyTorch |2.09|0.982|V|V|V|
| [face-detection-retail-0044](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/face-detection-retail-0044.zip) | Face |Caffe|1.21|1.067|V|V|V|
| [faster_rcnn_inception_resnet_v2_atrous_coco](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/faster_rcnn_inception_resnet_v2_atrous_coco.zip) | Object |TensorFlow|115|30.687|V|V| |
| [faster_rcnn_inception_v2_coco](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/faster_rcnn_inception_v2_coco.zip) |Object| TensorFlow |25.6|30.687|V|V|V|
| [faster_rcnn_resnet50_coco](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/faster_rcnn_resnet50_coco.zip)|Object|TensorFlow |55.8|57.203|V|V|V|
| [faster_rcnn_resnet101_coco](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/faster_rcnn_resnet101_coco.zip)|Object|TensorFlow |92.1|112.052|V|V|V|
| [Gaze-Estimation](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/Gaze-Estimation.zip)|Gaze direction vector|Caffe2|17.5|0.139|V|V|V|
| [mobilenet-ssd](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/mobilenet-ssd.zip)|Object|Caffe |11.1|2.316|V|V|V|
| [pedestrian-and-vehicle-detector-adas-0001](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/pedestrian-and-vehicle-detector-adas-0001.zip)|Person  Vehicle|Caffe|3.36|3.974|V|V|V|
| [pedestrian-detection-adas-0002](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/pedestrian-detection-adas-0002.zip)|Person |Caffe|2.44|2.494|V|V|V|
| [person-detection-0200](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/person-detection-0200.zip)|Person|PyTorch|3.64|0.786|V|V|V|
| [person-detection-0201](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/person-detection-0201.zip)|Person|PyTorch|3.64|1.768|V|V|V|
| [person-detection-0202](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/person-detection-0202.zip)|Person|PyTorch|3.64|3.143|V|V|V|
| [person-detection-retail-0002](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/person-detection-retail-0002.zip)|Person|Caffe|6.44|12.427|V|V|V|
| [person-detection-retail-0002_person-reidentification-retail-0031](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/person-detection-retail-0002_person-reidentification-retail-0031.zip)|Descriptors of persons for tracking|Caffe|7.10|0.028|V|V|V|
| [person-detection-retail-0013](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/person-detection-retail-0013.zip)|Person|Caffe|1.71|2.300|V|V|V|
| [person-detection-retail-0013_person-reidentification-retail-0031](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/person-detection-retail-0013_person-reidentification-retail-0031.zip)|Descriptors of persons for tracking|Caffe|2.38|0.028|V|V|V|
| [rfcn-resnet101-coco-tf](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/rfcn-resnet101-coco-tf.zip)|Object |TensorFlow|102|53.462|V|V|V|
| [ssd_mobilenet_v1_coco](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/ssd_mobilenet_v1_coco.zip)|Object|TensorFlow|13.1|2.494|V|V|V|
| [ssd_mobilenet_v1_fpn_coco](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/ssd_mobilenet_v1_fpn_coco.zip)|Object|TensorFlow|59.2|123.309|V|V|V|
| [ssd_mobilenet_v2_coco](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/ssd_mobilenet_v2_coco.zip)|Object|TensorFlow|32.3|3.775|V|V|V|
| [ssd_resnet50_v1_fpn_coco](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/ssd_resnet50_v1_fpn_coco.zip)|Object|TensorFlow|98.9|178.6807|V|V|V|
| [ssd300](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/ssd300.zip)|Object|Caffe|50.2|62.815|V|V|V|
| [ssd512](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/ssd512.zip)|Object|Caffe|52|180.611|V|V|V|
| [ssdlite_mobilenet_v2](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/ssdlite_mobilenet_v2.zip)|Object|TensorFlow|8.81|1.525|V|V|V|
| [vehicle-detection-0200](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/vehicle-detection-0200.zip)|Vehicle|PyTorch|3.64|0.786|V|V|V|
| [vehicle-detection-0201](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/vehicle-detection-0201.zip)|Vehicle|PyTorch|3.64|1.768|V|V|V|
| [vehicle-detection-0202](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/vehicle-detection-0202.zip)|Vehicle|PyTorch|3.64|3.143|V|V|V|
| [vehicle-license 1](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/vehicle-license1.zip)|License plate|Caffe|5.03|0.126|V|V|V|
| [vehicle-license 2](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/vehicle-license2.zip)|License plate|PyTorch|25.1|0.462|V|V|V|
| [vehicle-license-plate-detection-barrier-0106](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/vehicle-license-plate-detection-barrier-0106.zip)|Vehicle License Plate|TensorFlow|1.42|0.349|V|V|V|
| [yolo-v3-tf](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/yolo-v3-tf.zip)|Object|Keras|118|65.984|V|V|V|
| [yolo-v3-tiny-tf](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/yolo-v3-tiny-tf.zip)|Object|TensorFlow|16.9|6.988|V|V|V|
|[yolov4](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Object-detection/yolov4.zip)|Object|TensorFlow|245|62.02|V|V|V|

## Classification Inferences

It is a systematic grouping of observations into categories, such as when biologists categorize plants, animals, and other lifeforms into different taxonomies.

| Model Name | Major detectable features | Source Framework  | Model size(MB)  | Complexity (GFLOPs) | CPU support | GPU support  | VPU (Myriad X) support |        
|------ |------ |------ |------ |------ |------ |------ |------ |
|[alexnet](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/alexnet.zip)|Classification|Caffe|116|1.5|V|V|V|
|[caffenet](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/caffenet.zip)|Classification|Caffe|116|1.463|V|V|V|
|[densenet-121](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/densenet-121.zip)|Classification|Caffe|15.5|5.724|V|V|V|
|[densenet-121-tf](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/densenet-121-tf.zip)|Classification|TensorFlow|15.6|5.289|V|V|V|
|[densenet-161](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/densenet-161.zip)|Classification|Caffe|55.1|15.561|V|V|V|
|[densenet-161-tf](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/densenet-161-tf.zip)|Classification|TensorFlow|55.2|14.128|V|V|V|
|[densenet-169](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/densenet-169.zip)|Classification|Caffe|27.4|6.788|V|V|V|
|[densenet-169-tf](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/densenet-169-tf.zip)|Classification|TensorFlow|27.5|6.16|V|V|V|
|[efficientnet-b0](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/efficientnet-b0.zip)|Classification|TensorFlow|10.2|0.189|V|V|V|
|[efficientnet-b0_auto_aug](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/efficientnet-b0_auto_aug.zip)|Classification|TensorFlow|10.2|0.819|V|V|V|
|[efficientnet-b0-pytorch](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/efficientnet-b0-pytorch.zip)|Classification|PyTorch|10.2|0.819|V|V|V|
|[efficientnet-b5](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/efficientnet-b5.zip)|Classification|TensorFlow|58.3|21.252|V|V|V|
|[efficientnet-b5-pytorch](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/efficientnet-b5-pytorch.zip)|Classification|PyTorch|58.3|21.252|V|V|V|
|[efficientnet-b7_auto_aug](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/efficientnet-b7_auto_aug.zip)|Classification|TensorFlow|127|77.168|V|V|V|
|[efficientnet-b7-pytorch](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/efficientnet-b7-pytorch.zip)|Classification|PyTorch|127|77.168|V|V|V|
|[googlenet-v1](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/googlenet-v1.zip)|Classification|Caffe|13.5|3.266|V|V|V|
|[googlenet-v1-tf](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/googlenet-v1-tf.zip)|Classification|TensorFlow|12.7|3.016|V|V|V|
|[googlenet-v2](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/googlenet-v2.zip)|Classification|Caffe|21.5|4.058|V|V|V|
|[googlenet-v2-tf](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/googlenet-v2-tf.zip)|Classification|TensorFlow|21.5|4.058|V|V|V|
|[googlenet-v3](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/googlenet-v3.zip)|Classification|TensorFlow|45.7|11.469|V|V|V|
|[googlenet-v3-pytorch](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/googlenet-v3-pytorch.zip)|Classification|PyTorch|45.6|11.469|V|V|V|
|[googlenet-v4-tf](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/googlenet-v4-tf.zip)|Classification|TensorFlow|81.7|24.584|V|V|V|
|[hbonet-0.5](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/hbonet-0.5.zip)|Classification|PyTorch|5.07|0.096|V|V|V|
|[hbonet-0.25](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/hbonet-0.25.zip)|Classification|PyTorch|3.93|0.037|V|V|V|
|[hbonet-1.0](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/hbonet-1.0.zip)|Classification|PyTorch|8.92|0.305|V|V|V|
|[inception-resnet-v2-tf](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/inception-resnet-v2-tf.zip)|Classification|TensorFlow|58.1|22.227|V|V|V|
|[mobilenet-v2-pytorch](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/mobilenet-v2-pytorch.zip)|Classification|PyTorch|6.77|0.615|V|V|V|
|[octave-densenet-121-0.125](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/octave-densenet-121-0.125.zip)|Classification|MXNet|16.5|4.883|V|V|V|
|[octave-resnet-26-0.25](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/octave-resnet-26-0.25.zip)|Classification|MXNet|30.8|3.768|V|V|V|
|[octave-resnet-50-0.125](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/octave-resnet-50-0.125.zip)|Classification|MXNet|49.3|7.221|V|V|V|
|[octave-resnet-101-0.125](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/octave-resnet-101-0.125.zip)|Classification|MXNet|86.2|13.387|V|V|V|
|[octave-resnet-200-0.125](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/octave-resnet-200-0.125.zip)|Classification|MXNet|125|25.407|V|V|V|
|[octave-resnext-50-0.25](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/octave-resnext-50-0.25.zip)|Classification|MXNet|48.3|6.444|V|V|V|
|[octave-resnext-101-0.25](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/octave-resnext-101-0.25.zip)|Classification|MXNet|85.5|11.521|V|V|V|
|[octave-se-resnet-50-0.125](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/octave-se-resnet-50-0.125.zip)|Classification|MXNet|54.5|7.246|V|V|V|
|[resnest-50-pytorch](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/resnest-50-pytorch.zip)|Classification|PyTorch|52.8|10.8148|V|V|V|
|[resnet-18-pytorch](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/resnet-18-pytorch.zip)|Classification|PyTorch|22.3|3.637|V|V|V|
|[resnet-34-pytorch](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/resnet-34-pytorch.zip)|Classification|PyTorch|41.6|7.3409|V|V|V|
|[resnet-50-pytorch](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/resnet-50-pytorch.zip)|Classification|PyTorch|48.8|8.216|V|V|V|
|[resnet-50-tf](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/resnet-50-tf.zip)|Classification|TensorFlow|48.8|8.2164|V|V|V|
|[se-inception](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/se-inception.zip)|Classification|TensorFlow|23|4.091|V|V|V|
|[se-resnet-50](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/se-resnet-50.zip)|Classification|Caffe|53.7|7.775|V|V|V|
|[se-resnet-101](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/se-resnet-101.zip)|Classification|Caffe|94.5|15.239|V|V|V|
|[se-resnet-152](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/se-resnet-152.zip)|Classification|Caffe|128|22.709|V|V|V|
|[se-resnext-50](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/se-resnext-50.zip)|Classification|Caffe|52.7|8.533|V|V|V|
|[se-resnext-101](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/se-resnext-101.zip)|Classification|Caffe|93.7|16.054|V|V|V|
|[squeezenet1.0](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/squeezenet1.0.zip)|Classification|Caffe|2.45|1.737|V|V|V|
|[squeezenet1.1](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/squeezenet1.1.zip)|Classification|Caffe|2.42|0.785|V|V|V|
|[vgg16](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/vgg16.zip)|Classification|Caffe|263|30.974|V|V|V|
|[vgg19](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Classification/vgg19.zip)|Classification|Caffe|274|143.667|V|V|V|
 

## Segmentation Inferences 

Segmentation is an extension of object detection problem. Instead of
returning bounding boxes, semantic segmentation models return a "painted"
version of the input image, where the "color" of each pixel represents a certain
class. These networks are much bigger than respective object detection networks,
but they provide a better (pixel-level) localization of objects and they can
detect areas with complex shape (for example, free space on the road).

| Model Name | Major detectable features | Source Framework  | Model size(MB)  | Complexity (GFLOPs) | CPU support | GPU support  | VPU (Myriad X) support | 
|------ |------ |------ |------ |------ |------ |------ |------ |
|[deeplabv3](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Segmentation/deeplabv3.zip)|Segmentation|TensorFlow|4.14|11.469|V|V||
|[icnet-camvid-ava-0001](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Segmentation/icnet-camvid-ava-0001.zip)|Segmentation|TensorFlow|51.1|151.82|V|V||
|[icnet-camvid-ava-sparse-30-0001](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Segmentation/icnet-camvid-ava-sparse-30-0001.zip)|Segmentation|TensorFlow|51.1|151.82|V|V||
|[icnet-camvid-ava-sparse-60-0001](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Segmentation/icnet-camvid-ava-sparse-60-0001.zip)|Segmentation|TensorFlow|51.1|151.82|V|V||
|[midasnet](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Segmentation/midasnet.zip)|Segmentation|PyTorch|198|207.4915|V|V|V|
|[road-segmentation-adas-0001](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Segmentation/road-segmentation-adas-0001.zip)|Segmentation|PyTorch|0.724|4.770|V|V|V|
|[semantic-segmentation-adas-0001](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Segmentation/semantic-segmentation-adas-0001.zip)|Segmentation|Caffe|25.6|58.572|V|V||
|[unet-camvid-onnx-0001](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Segmentation/unet-camvid-onnx-0001.zip)|Segmentation|PyTorch|59.2|260.1|V|V|V|

 
## Instance Segmentation Inferences

Instance segmentation is an extension of object detection and semantic segmentation problems. Instead of predicting a bounding box around each object instance, instance segmentation model outputs pixel-wise masks for all instances.

| Model Name | Major detectable features | Source Framework  | Model size(MB)  | Complexity (GFLOPs) | CPU support | GPU support  | VPU (Myriad X) support | 
|------ |------ |------ |------ |------ |------ |------ |------ |
|[mask_rcnn_inception_resnet_v2_atrous_coco](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Instance-segmentation/mask_rcnn_inception_resnet_v2_atrous_coco.zip)|Segmentation|TensorFlow|124|675.314|V|V||
|[mask_rcnn_resnet101_atrous_coco](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Instance-segmentation/mask_rcnn_resnet101_atrous_coco.zip)|Segmentation|TensorFlow|103|674.58|V|V||
|[mask_rcnn_inception_v2_coco](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Instance-segmentation/mask_rcnn_inception_v2_coco.zip)|Segmentation|TensorFlow|30.5|54.926|V|V|V|
|[mask_rcnn_resnet50_atrous_coco](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Instance-segmentation/mask_rcnn_resnet50_atrous_coco.zip)|Segmentation|TensorFlow|67.5|294.738|V|V|V|


 
## Text Detection Inferences

Various applications for text detection can run with additional deep Learning models.

| Model Name | Major detectable features | Source Framework  | Model size(MB)  | Complexity (GFLOPs) | CPU support | GPU support  | VPU (Myriad X) support | 
|------ |------ |------ |------ |------ |------ |------ |------ |
|[horizontal-text-detection-0001](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Text-detection/horizontal-text-detection-0001.zip)|Text|PyTorch|4.35|7.78|V|V||
|[text-detection1](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Text-detection/text-detection1.zip)|Text content|TensorFlow|35.7|1.485|V|V|V|
|[text-detection-0003](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Text-detection/text-detection-0003.zip)|Text|TensorFlow|13|51.256|V|V|V|
|[text-detection-0004](https://edgeaisuite.blob.core.windows.net/openvino2021-1/model/Text-detection/text-detection-0004.zip)|Text|TensorFlow|8.43|23.305|V|V|V|
 
## INTEL OpenVINO™ Inference Engine
OpenVINO™ toolkit is a comprehensive toolkit for quickly developing applications and solutions that solve a variety of tasks including emulation of human vision, automatic speech recognition, natural language processing, recommendation systems, and many others. Based on latest generations of artificial neural networks, including Convolutional Neural Networks (CNNs), recurrent and attention-based networks, the toolkit extends computer vision and non-vision workloads across Intel® hardware, maximizing performance. It accelerates applications with high-performance, AI and deep learning inference deployed from edge to cloud. (Reference from official Intel OpenVINO website)

| Item | Version | Reference |
|------ |------ |------ |
|[OV_2021.1](https://edgeaisuite.blob.core.windows.net/openvino2021-1/intel-install/w_openvino_toolkit_p_2021.1.110.exe)|OpenVINO 2021.1|[https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_inference_engine_intro.html](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_inference_engine_intro.html)|
 
[Note] The installation package is also referred to official Intel OpenVINO website and preliminarily verified by ADVANTECH.

## INTEL OpenVINO™ Model Optimizer for Conversion
Model Optimizer process assumes you have a network model trained using a supported deep learning framework. Model Optimizer produces an Intermediate Representation (IR) of the network, which can be read, loaded, and inferred with the Inference Engine. The Inference Engine API offers a unified API across a number of supported Intel® platforms. The Intermediate Representation is a pair of files describing the model:
<br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 
.xml - Describes the network topology.  
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 
.bin - Contains the weights and biases binary data.
<br>
(Above reference from official Intel OpenVINO website)
 
| Item | Version | Reference |
|------ |------ |------ |
|[Model_opt](https://edgeaisuite.blob.core.windows.net/openvino2021-1/mo-lib/python_386.zip)|OpenVINO 2021.1|[https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)|

Ex:
<br> &nbsp; &nbsp; &nbsp; &nbsp; Firstly, download "Model_opt" and extract it. 
<br> &nbsp; &nbsp; &nbsp; &nbsp; Secondly, go to [ssd_mobilenet_v2_coco]( http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz) and download a source model. 
<br> &nbsp; &nbsp; &nbsp; &nbsp; To put ssd_mobilenet_v2_coco_2018_03_29  folder into Model_opt folder.
<br> &nbsp; &nbsp; &nbsp; &nbsp; Finally, execute the following command in your cmd windows.

<pre>mo.py --reverse_input_channels --input_shape=[1,300,300,3] --input=image_tensor --data_type=FP16 --output_dir .//FP16/ --input_model .//ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb --transformations_config .//extensions/front/tf/ssd_v2_support.json --output=detection_classes,detection_scores,detection_boxes,num_detections --tensorflow_object_detection_api_pipeline_config .//ssd_mobilenet_v2_coco_2018_03_29/pipeline.config</pre>


## Reference Links
- [https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/index.md](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/index.md) 
- [https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/index.md](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/index.md) 
- [https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/README.md#demos-that-support-pre-trained-models](https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/README.md#demos-that-support-pre-trained-models)  


## Legal Information
 
These provisions are licensed under [Apache License Version 2.0](https://github.com/openvinotoolkit/openvino/blob/master/LICENSE). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.