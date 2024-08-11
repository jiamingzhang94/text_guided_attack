运行'generate_adv_img.py'
```bash
python generate_adv_img.py --model_name 'ViT-B/32' \
                           --decoder_path 'YOUR/DECODER/PATH' \
                           --image_only True #True 用图像的嵌入生成adv_image \
                           --projection_path 'YOUR/PROJECTION/PATH' #如果image_only为True 可以不设置这一项\
                           --clean_image_path 'imagenet-1K' #imagenet-1k 验证集 \
                           --target_caption 'coco_karpathy_val.json' #mscoco验证集 \
                           ---target_image_path 'YOUR/MSCOCO/PATH' #目标图像路径 \
                           --batch_size 40 #选择5000的约数 因为cleanimage数量大于5000 \
                           --output_path 'saved_result/our_image/vit_b_32'
```
