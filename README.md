#### 数据下载
点击从下面这个这个链接下载 Coco train 2017图像数据集

http://images.cocodataset.org/zips/train2017.zip

下载完成后，由于MSCOCO数据集一张图像可能对应多个文本描述，因此需要对数据进行处理

运行data/data_process.py 

`python data/data_process.py`

后得到mscoco_exist.parquet文件，它的数据格式如下所示:



# 我的Markdown文档

## 简介

这是一个使用Markdown编写的示例文档。

### 列表

* 项目1
* 项目2
    * 子项目

1. 项目1
2. 项目2

### 引用

> 这是一个引用。

### 代码

这是一个`单行代码`示例。

