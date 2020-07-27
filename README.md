# Image-recognition-trash-can
In this project, it is mainly to design a device that uses MobileNetV2 to classify the garbage for image recognition, and then put the classified garbage into the designated garbage bin.

由于英语不好，所以就说中文了，这个项目说起来也还是个缝合怪，做的时候是参考了这位的https://github.com/jzx-gooner/DL-wastesort
但是后面做的时候由于个人水平问题，发现Inception V3没办法使用TRT加速，因此选了V2。

而步进电机也是参考野火的程序改的，后面自己进行一些调整

用的器件有：Jetson Nano（用来识别图像的) F429开发板(控制电机，其实不用这么贵，主要是手头目前就它）

文件名带Test的，都是用来测试单个模块的，重点文件就是，训练模型用的fine_tuning.py，将得到的TF模型转换为TRT模型 tf_to_trt.py ，运行模型trt_cemare_stepper.py     

视频https://www.bilibili.com/video/BV15K411n7bY
