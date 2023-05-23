import tensorflow as tf
import keras
from keras.applications.mobilenet_v2 import MobileNetV2
import tf2onnx
import onnx

# This file can convert h5 model to onnx
model = MobileNetV2(weights='./mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5')
temp_model_file = './mobilenet_v2_tf.onnx'
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
onnx_model = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=temp_model_file)

ori_model = onnx.load(temp_model_file)
del ori_model.opset_import[1]
onnx.save(ori_model,temp_model_file)