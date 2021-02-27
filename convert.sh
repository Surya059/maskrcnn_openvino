#python3 /opt/intel/openvino_2021.2.185/deployment_tools/model_optimizer/mo_onnx.py \
#--input_model mask_rcnn_R_50_FPN_1x.onnx \
#--input "0:2" \
#--input_shape [1,3,800,800] \
#--mean_values [102.9801,115.9465,122.7717] \
#--transformations_config /opt/intel/openvino_2021.2.185/deployment_tools/model_optimizer/extensions/front/onnx/mask_rcnn.json


python3 /opt/intel/openvino_2021.2.185/deployment_tools/model_optimizer/mo.py --input_model /workspace/model.pb --tensorflow_use_custom_operations_config=/opt/intel/openvino_2021.2.185/deployment_tools/model_optimizer/extensions/front/tf/mask_rcnn_support_api_v1.15.json --tensorflow_object_detection_api_pipeline_config mask_rcnn_inception_v2_coco.config
