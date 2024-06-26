import os
name=[]
for i in range(20):
    name.append(f"model_epoch_{i+1}.pth")
for i in name:
    comand=f"""python main.py --model_path 'save_model/{i}'"""
    os.system(comand)
    # break