import os
import pathlib

# 训练中如果提示图片错误
# 可以根据对应的错误信息
# 来删除这类型图片
data_root_orig = '/home/y/Documents/training_dataset'
data_root = pathlib.Path(data_root_orig)
all_image_paths = list(data_root.glob('*/*'))


for i, filename in enumerate(all_image_paths):
    with open(filename, 'rb') as imageFile:
        # GIF就是提示的错误信息
        if imageFile.read().startswith(b'GIF'):
            print(f"{i}: {filename} - found!")
            os.remove(filename)




