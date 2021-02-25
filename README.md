# maskrcnn_openvino
Dockerized inference of OpenVINO usng MaskRCNN


## Converting MaskRCNN ONNX model to OpenVINO
Refer [this] (https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_onnx_specific_Convert_Mask_RCNN.html) link to download the onnx model

The following commands will output the xml and bin files for openvino
```sh
cd docker/openvino_inference/
sh build.sh
sh run.sh
#Run this command after entering the docker container
sh convert.sh 
```
