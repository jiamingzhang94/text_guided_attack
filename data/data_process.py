import pandas as pd
import pyarrow.parquet as pq
import os
import argparse
from tqdm import tqdm
argparse = argparse.ArgumentParser()
argparse.add_argument("--train", type=str, default=True)
argparse.add_argument("--clip_model_path", type=str, default="/data2/ModelWarehouse/clip-vit-base-patch32")
argparse.add_argument("--image_path", type=str, default="/data2/zhiyu/data/coco/images/train2017")
argparse.add_argument("--data_path", type=str, default="/data2/junhong/proj/text_guide_attack/data/mscoco.parquet")
args = argparse.parse_args()

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
data_path = "mscoco.parquet"
# data=pd.read_table(data_path,encoding='latin1')
data = pq.read_table(data_path)
df = data.to_pandas()
# url_data=df["URL"]
# url_data=url_data.to_frame()
# # url_data.to_csv("mscoco_url.csv",index=False)
# # print(len(url_data))
# print(url_data.head())
# print(df.head())
exist_data = {
    "URL": [],
    "TEXT": []
}
listdir=os.listdir(args.image_path)
print(len(listdir))
# df["file_name"]=df["URL"].str.split('/').str[-1]
# for image,text in tqdm(zip(df["file_name"],df["TEXT"])):
#     if image not in exist_data["URL"]:
#         exist_data["URL"].append(image)
#         exist_data["TEXT"].append([text])
#     else:
#         index=exist_data["URL"].index(image)
#         exist_data["TEXT"][index].append(text)
# exist_df = pd.DataFrame(exist_data)
# exist_df.to_parquet("mscoco_exist.parquet", engine='pyarrow')
# print(len(exist_data["URL"]))
# print(df.head())
# for url in df["URL"]:
#     url= url.split("/")[-1]
# for url, text in tqdm(zip(df["URL"], df["TEXT"])):
#     image_name = url.split("/")[-1]
    # print(image_name)
    # print(image_name in listdir)
    # image_path = os.path.join(args.image_path, image_name)
    # if os.path.exists(image_path):
    # if image_name not in listdir:

        # print(image_name)
        # exist_data["URL"].append(url)
        # exist_data["TEXT"].append(text)


# exist_df = pd.DataFrame(exist_data)
# print(len(exist_df["URL"]))
# exist_df.to_parquet("mscoco_exist.parquet", engine='pyarrow')
# print(len(listdir))
data=pq.read_table("/data2/junhong/proj/text_guide_attack/data/mscoco_exist.parquet")
data=data.to_pandas()
print(data["TEXT"][0])
# print(data.head())
print("1")
