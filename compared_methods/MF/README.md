The targeted images h_ξ(c_tar) can be obtained via Stable Diffusion by reading text prompt from the sampled COCO captions, with the script below and [`txt2img_coco.py`](https://drive.google.com/file/d/1hTHxlgdx97_uEL3g9AmVx-qGNgssJeIy/view?usp=sharing) (please move `txt2img_coco.py` to ```./stable-diffusion/```, note that hyperparameters can be adjusted with your preference):
<!-- $\boldsymbol{h}_\xi(\boldsymbol{c}_\text{tar})$ -->

```bash
python txt2img_coco.py \
        --ddim_eta 0.0 \
        --n_samples 10 \
        --n_iter 1 \
        --scale 7.5 \
        --ddim_steps 50 \
        --plms \
        --skip_grid \
        --ckpt ./_model_pool/sd-v1-4-full-ema.ckpt \
        --from-file './name_of_your_coco_captions_file.txt' \
        --outdir './path_of_your_targeted_images' 
```
where the ckpt is provided by [Stable Diffusion v1](https://github.com/CompVis/stable-diffusion#weights:~:text=The%20weights%20are%20available%20via) and can be downloaded here: [sd-v1-4-full-ema.ckpt](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt).

Additional implementation details of text-to-image generation by Stable Diffusion can be found [HERE](https://github.com/CompVis/stable-diffusion#:~:text=active%20community%20development.-,Reference%20Sampling%20Script,-We%20provide%20a).