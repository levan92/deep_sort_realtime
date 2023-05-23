# this script is used for .onnx ---> .om convert. Make sure installed CANN AND MindX SDK before use.
# ref:https://www.hiascend.com/zh/software/mindx-sdk/community  or Chinese version(usually newer)  https://www.hiascend.com/zh/software/mindx-sdk/community

# !!! soc_version need set to actual chip with Ascend310P3/Ascend310/Ascend910

# 'dynamic_batch_size' is used for input shape assignment, in this example limit to 4. 
# For performance consideration it should be like 1/2/4/8/16, and assemble batch in embedder_npu.py(TODO)
atc --model=mobilenet_v2_tf.onnx --framework=5 --output=mobilenet_v2_tf --output_type=FP32 --soc_version=Ascend310P3 --input_shape="input:-1,224,224,3"  --dynamic_batch_size="1,2,3,4" --log=info