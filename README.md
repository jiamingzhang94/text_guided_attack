#### 数据下载与处理
点击从下面这个这个链接下载 Coco train 2017图像数据集

http://images.cocodataset.org/zips/train2017.zip

下载完成后，MSCOCO数据集一张图像可能对应多个文本描述，因此需要对数据进行处理

运行data/data_process.py 

`python data/data_process.py --image_path 'your image_path'`

后得到mscoco_exist.parquet文件，它的数据格式如下所示:




