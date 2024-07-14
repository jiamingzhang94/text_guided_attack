import pandas as pd
import pyarrow.parquet as pq
import os
import argparse
from tqdm import tqdm
argparse = argparse.ArgumentParser()
argparse.add_argument("--image_path", type=str, default="/home/dycpu6_8tssd1/jmzhang/datasets/mscoco/train2017")
argparse.add_argument("--data_path", type=str, default="/home/dycpu6_8tssd1/jmzhang/datasets/mscoco/mscoco.parquet")
args = argparse.parse_args()

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
data_path = "mscoco.parquet"
# data=pd.read_table(data_path)
data = pq.read_table(data_path)
df = data.to_pandas()
# for i in df["URL"]:
#     if "2017" in i:
#         print(i)
# print(df.head())
exist_data = {
    "URL": [],
    "TEXT": []
}
listdir=os.listdir(args.image_path)
# print(len(listdir))
df["file_name"]=df["URL"].str.split('/').str[-1]
for image,text in tqdm(zip(df["file_name"],df["TEXT"])):
    if image not in exist_data["URL"]:
        exist_data["URL"].append(image)
        exist_data["TEXT"].append([text])
    else:
        index=exist_data["URL"].index(image)
        exist_data["TEXT"][index].append(text)
exist_df = pd.DataFrame(exist_data)
exist_df.to_parquet("mscoco_exist.parquet", engine='pyarrow')





# print(len(exist_data["URL"]))
# print(df.head())

# exist_df = pd.DataFrame(exist_data)
# print(len(exist_df["URL"]))
# exist_df.to_parquet("mscoco_exist.parquet", engine='pyarrow')

# data=pq.read_table("mscoco_exist.parquet")
# data=data.to_pandas()
# # print(data["TEXT"][0])
# print(data.head())
# print("1")
