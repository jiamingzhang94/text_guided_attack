### SU
Note that hyperparameters can be adjusted with your preference
```bash
CUDA_VISIBLE_DEVICES=0 python main.py  \
    --attack 'su' \
    --model 'ViT-B/32' #clip model name \
    --batch_size 40 \
    --eps 8/255 \
    --alpha 1.0/255 \
    --loss_type 'cos' # select from ['cos','mse'] \
    --clean_image_path 'imagenet-1K'  #path to the clean images \
    --target_data_path 'coco_karpathy_val.json'\
    --target_image_path 'coco' #path to target image \
    --output '/YOUR/OUTPUT/PATH' \

    
```

### SASW_WS
```bash
CUDA_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 python main.py  \
    --attack 'sasd_ws' \
    --model 'resnet50' #resnet model name \
    --batch_size 40 \
    --eps 8/255 \
    --alpha 1.0/255 \
    --loss_type 'cos' # select from ['cos','mse'] \
    --clean_image_path 'imagenet-1K'  #path to the clean images \
    --target_data_path 'coco_karpathy_val.json'\
    --target_image_path 'coco' #path to target image \
    --output '/YOUR/OUTPUT/PATH' \
    --checkpoint_path '/YOUR/CHECKPONIT/PATH' #Please download the checkpoint of the 'resnet50_SASD_Model' from 'https://drive.google.com/drive/folders/1CsNN53GYy9nFcJdSkS5Pcy_faisMDRRh', and put it into the path checkpoint_path.\
    
```
