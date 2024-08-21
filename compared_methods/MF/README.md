### Generate the targeted images
The targeted images h_ξ(c_tar) can be obtained via Stable Diffusion by reading text prompt from the sampled COCO captions, with the script below and [`txt2img_coco.py`](https://drive.google.com/file/d/1hTHxlgdx97_uEL3g9AmVx-qGNgssJeIy/view?usp=sharing) (please move `txt2img_coco.py` to ```./stable-diffusion/```, note that hyperparameters can be adjusted with your preference):
<!-- $\boldsymbol{h}_\xi(\boldsymbol{c}_\text{tar})$ -->

```bash
CUDA_VISIBLE_DEVICES=4,5 python txt2img_coco.py \
        --ddim_eta 0.0 \
        --n_samples 2 \
        --n_iter 1 \
        --scale 7.5 \
        --ddim_steps 50 \
        --plms \
        --skip_grid \
        --ckpt sd-v1-4-full-ema.ckpt \
        --from-file 'coco_karpathy_val.json' \
        --outdir '/YOUR/OUTPUT/PATH'
```
where the ckpt is provided by [Stable Diffusion v1](https://github.com/CompVis/stable-diffusion#weights:~:text=The%20weights%20are%20available%20via) and can be downloaded here: [sd-v1-4-full-ema.ckpt](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt).

Additional implementation details of text-to-image generation by Stable Diffusion can be found [HERE](https://github.com/CompVis/stable-diffusion#:~:text=active%20community%20development.-,Reference%20Sampling%20Script,-We%20provide%20a).

### MF-ii
Note that hyperparameters can be adjusted with your preference
```bash
CUDA_VISIBLE_DEVICES=0 python MF-ii.py  \
    --batch_size 40 \
    --num_samples 5000 \
    --steps 100 \
    --output '/YOUR/OUTPUT/PATH' \
    --model_name 'ViT-B/32'  #clip model name
    --clean_image 'imagenet-1K'  #path to the clean images \
    --target_image 'stable_generate_image' #path to the generated images by stable difussion \
    --target_caption 'coco_karpathy_val.json'
```

### MF-it
```bash
CUDA_VISIBLE_DEVICES=0 python MF-it.py  \
    --batch_size 40 \
    --num_samples 5000 \
    --steps 100 \
    --output '/YOUR/OUTPUT/PATH' \
    --model_name 'ViT-B/32'  #clip model name
    --clean_image 'imagenet-1K'  #path to the clean images \
    --target_caption 'coco_karpathy_val.json'
```