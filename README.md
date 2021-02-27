# MaskRCNN OpenVINO benchmarks
Dockerized inference of OpenVINO usng MaskRCNN

## Prerequisites
* Ubuntu 18.04.4 LTS
* [Docker](https://docs.docker.com/engine/install/ubuntu/) version 19.03.12

## Freeze MaskRCNN file
#Download MaskRCNN coco weights before running the following script
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
cd docker/maskrcnn_inference/
sh build.sh
sh run.sh

## [ W I P ]Converting MaskRCNN tensorflow model to OpenVINO
Refer [this](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_onnx_specific_Convert_Mask_RCNN.html) link to download the onnx model

The following commands will output the xml and bin files for openvino
```sh
cd docker/openvino_inference/
sh build.sh
sh run.sh
#Run this command after entering the docker container
sh convert.sh 
```
