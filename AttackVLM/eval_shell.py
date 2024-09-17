import os
command = """CUDA_VISIBLE_DEVICES=0 python eval_clip_text_score.py --num_samples 1000 \
            --batch_size 40 \
            --tgt_text_path '/new_data/yifei2/junhong/AttackVLM-main/data/captions/coco_captions_10000.txt' \
            --pred_text_path '/new_data/yifei2/junhong/AttackVLM-main/output/lavis/dalle_clean_caption_pred.txt'
            """
os.system(command)