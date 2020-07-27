# 将训练到的TF模型转换为TRT模型
from tensorflow.python.compiler.tensorrt import trt_convert as trt
params=trt.DEFAULT_TRT_CONVERSION_PARAMS
params._replace(precision_mode=trt.TrtPrecisionMode.FP32)
converter = trt.TrtGraphConverterV2(input_saved_model_dir='tf_savedmodel',conversion_params=params)
converter.convert()#完成转换,但是此时没有进行优化,优化在执行推理时完成
print("Successful")
converter.save('trt_savedmodel')

