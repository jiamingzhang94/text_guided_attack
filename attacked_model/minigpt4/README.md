### minigpt4_v1_vicuna 生成回复并测试
```bash
CUDA_VISIBLE_DEVICES=0 python generate_response.py  \
    --cfg_path 'minigpt4_eval.yaml' \
    --prompt "Describe this image in one sentence" \
    --data_path 'coco_karpathy_test.json' \
    --image_path 'coco/images' \
    --gt_path 'ground_truth.json' \
    --output_path 'PATH/TO/OUTPUT.json' \ 
    --llama_path 'Vision-CAIR/vicuna-7b' \
    --ckpt_path '' 从这个链接下载adapter https://drive.usercontent.google.com/download?id=1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R&export=download&authuser=0
```
### minigpt4_v1_llama2
```bash
CUDA_VISIBLE_DEVICES=0 python generate_response.py  \
    --cfg_path 'minigpt4_llama2_eval.yaml' \
     --prompt "Describe this image in one sentence" \
    --data_path 'coco_karpathy_test.json' \
    --image_path 'coco/images' \
    --gt_path 'ground_truth.json' \
    --output_path 'PATH/TO/OUTPUT.json' \ 
    --llama_path 'meta-llama/Llama-2-7b-chat-hf' \
    --ckpt_path '' 从这个链接下载adapter https://drive.usercontent.google.com/download?id=11nAPjEok8eAGGEG1N2vXo3kBLCg0WgUk&export=download&authuser=0
   
```

### minigpt4_v2_llama2
```bash
CUDA_VISIBLE_DEVICES=0 python generate_response.py  \
    --cfg_path 'minigpt4_llama2_eval.yaml' \
    --prompt "Describe this image in one sentence" \
    --data_path 'coco_karpathy_test.json' \
    --image_path 'coco/images' \
    --gt_path 'ground_truth.json' \
    --output_path 'PATH/TO/OUTPUT.json' \ 
    --llama_path 'meta-llama/Llama-2-7b-chat-hf' \
    --ckpt_path '' 从这个链接下载adapter https://drive.usercontent.google.com/download?id=1aVbfW7nkCSYx99_vCRyP1sOlQiWVSnAl&export=download&authuser=0
    
```