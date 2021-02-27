docker run -v $PWD/../..:/workspace -it surya/openvino_inference python3 /opt/intel/openvino_2021.2.185/deployment_tools/model_optimizer/mo_tf.py --input_model mask_rcnn_R_50_FPN_1x.onnx \
--tensorflow_object_detection_api_pipeline_config mask_rcnn_inception_v2_coco.config 
