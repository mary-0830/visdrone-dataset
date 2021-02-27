
# coding:utf-8
#计算所有的图片（包括训练、验证和测试）的均值和标准差，直接将图片存放到同一个文件夹，把路径改下即可。

import cv2, os, argparse
import numpy as np
from tqdm import tqdm

def main():
    dirs = r'./images'  # 修改你自己的图片路径
    img_file_names = os.listdir(dirs)
    m_list, s_list = [], []
    for img_filename in tqdm(img_file_names):
        img = cv2.imread(dirs + '/' + img_filename)
        img = img / 255.0
        m, s = cv2.meanStdDev(img)
        m_list.append(m.reshape((3,)))
        s_list.append(s.reshape((3,)))
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)
    print("mean = ", m[0][::-1])
    print("std = ", s[0][::-1])


if __name__ == '__main__':
    main()

