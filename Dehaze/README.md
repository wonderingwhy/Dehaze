# 基于暗通道先验的图像去雾算法及有雾图像的深度估计

## 硬件环境

- 处理器 i5-4590 CPU @ 3.30GHz
- 内存 16.0 GB

## 软件环境

- Win 10 
- Visual Studio 2017 x64
- C/C++

## 依赖库

- OpenCV 3.4

## 使用方法

通过更改代码中的常量图片地址更换输入图片，其他参数也可根据需要修改。输出的6张图像分别为

- 原图及其大气光标记
- 去雾处理后的图像
- 原图像的暗通道
- 原图像的粗制透射率图
- 原图像的精化透射率图
- 通过精化透射率图估计的深度图

其中，去雾处理后的图像会被保存下来，保存路径和文件名可自行修改。

## 算法流程

参见项目报告

## 参考文献

[1].He K, Sun J, Tang X. Single image haze removal using dark channel prior[J]. IEEE transactions on pattern analysis and machine intelligence, 2011, 33(12): 2341-2353.

[2].He K, Sun J, Tang X. Guided image filtering[J]. IEEE transactions on pattern analysis and machine intelligence, 2013, 35(6): 1397-1409. 

[3].He K, Sun J. Fast Guided Filter[J]. Computer Science, 2015.

[4].http://blkstone.github.io/2015/08/20/single-image-haze-removal-using-dark-channel/

[5].https://blog.csdn.net/baimafujinji/article/details/74750283

[6].https://baike.baidu.com/item/%E5%8D%95%E8%B0%83%E9%98%9F%E5%88%97/4319570?fr=Aladdin

[7]. https://blog.csdn.net/a_bright_ch/article/details/77076228