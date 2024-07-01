#### 数据下载与处理
从链接中下载 [coco train2017](http://images.cocodataset.org/zips/train2017.zip)图像数据集

下载完成后，MSCOCO数据集一张图像可能对应多个文本描述，因此需要对数据进行处理

运行data/data_process.py 

`python data/data_process.py --image_path 'your image_path'`

后得到mscoco_exist.parquet文件，它的数据格式如下所示:

| URL               | TEXT                                                                                                                                                                                                                                                             |
|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 000000391895.jpg  | [A man with a red helmet on a small moped on a dirt road., Man riding a motor bike on a dirt road on the countryside., A man riding on the back of a motorcycle., A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains., A man in a red shirt and a red hat is on a motorcycle on a hill side.] |
| 000000522418.jpg  | [A woman wearing a net on her head cutting a cake., A woman cutting a large white sheet cake., A woman wearing a hair net cutting a large sheet cake., there is a woman that is cutting a white cake, A woman marking a cake with the back of a chef's knife.]                                           |
| 000000184613.jpg  | [A child holding a flowered umbrella and petting a yak., A young man holding an umbrella next to a herd of cattle., a young boy barefoot holding an umbrella touching the horn of a cow, A young boy with an umbrella who is touching the horn of a cow., A boy holding an umbrella while standing next to livestock.]   |
| 000000318219.jpg  | [A young boy standing in front of a computer keyboard., a little boy wearing headphones and looking at a computer monitor, He is listening intently to the computer at school., A young boy stares up at the computer monitor., a young kid with head phones on using a computer]                              |
| 000000554625.jpg  | [a boy wearing headphones using one computer in a long row of computers, A little boy with earphones on listening to something., A group of people sitting at desk using computers., Children sitting at computer stations on a long table., A small child wearing headphones plays on the computer.]          |

第一列为所下载图像的名字，第二列为对应图像的多个描述。

#### Data Download and Processing

Download the [coco train2017](http://images.cocodataset.org/zips/train2017.zip) image dataset from the link.

After downloading, the MSCOCO dataset may have multiple text descriptions for a single image, so the data needs to be processed.

Run `data/data_process.py`:

```bash
python data/data_process.py --image_path 'your image_path'
```

This will generate the `mscoco_exist.parquet` file, which has the following data format:

| URL               | TEXT                                                                                                                                                                                                                                                             |
|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 000000391895.jpg  | [A man with a red helmet on a small moped on a dirt road., Man riding a motor bike on a dirt road on the countryside., A man riding on the back of a motorcycle., A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains., A man in a red shirt and a red hat is on a motorcycle on a hill side.] |
| 000000522418.jpg  | [A woman wearing a net on her head cutting a cake., A woman cutting a large white sheet cake., A woman wearing a hair net cutting a large sheet cake., there is a woman that is cutting a white cake, A woman marking a cake with the back of a chef's knife.]                                           |
| 000000184613.jpg  | [A child holding a flowered umbrella and petting a yak., A young man holding an umbrella next to a herd of cattle., a young boy barefoot holding an umbrella touching the horn of a cow, A young boy with an umbrella who is touching the horn of a cow., A boy holding an umbrella while standing next to livestock.]   |
| 000000318219.jpg  | [A young boy standing in front of a computer keyboard., a little boy wearing headphones and looking at a computer monitor, He is listening intently to the computer at school., A young boy stares up at the computer monitor., a young kid with head phones on using a computer]                              |
| 000000554625.jpg  | [a boy wearing headphones using one computer in a long row of computers, A little boy with earphones on listening to something., A group of people sitting at desk using computers., Children sitting at computer stations on a long table., A small child wearing headphones plays on the computer.]          |

The first column is the name of the downloaded images, and the second column contains multiple descriptions corresponding to each image.
