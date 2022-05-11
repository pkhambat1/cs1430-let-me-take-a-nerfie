# let-me-take-a-nerfie

A pytorch implementation of [instant-ngp](https://github.com/NVlabs/instant-ngp), as described in [_Instant Neural Graphics Primitives with a Multiresolution Hash Encoding_](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf).


To run this project, simply open `let-me-take-a-nerfie.ipyny` in Google Colab and hit Runtime > Run all!


# Usage


```bash
### HashNeRF
# train with different backbones (with slower pytorch ray marching)
# for the colmap dataset, the default dataset setting `--mode colmap --bound 2 --scale 0.33` is used.
python main_nerf.py ../instant-ngp/data/nerf/fox --workspace trial_nerf_fox # fp32 mode
python main_nerf.py ../instant-ngp/data/nerf/fox --workspace trial_nerf_fox --fp16 # fp16 mode (pytorch amp)

# test mode
python main_nerf.py ../instant-ngp/data/nerf/fox --workspace trial_nerf_fox --fp16 --test

# The following code only runs on CUDA enabled hardware :( 
# for custom dataset, you should:
# 1. take a video / many photos from different views 
# 2. put the video under a path like ./data/custom/video.mp4 or the images under ./data/custom/images/*.jpg.
# 3. call the preprocess code: (should install ffmpeg and colmap first! refer to the file for more options)
python colmap2nerf.py --video ./data/custom/video.mp4 --run_colmap # if use video
python colmap2nerf.py --images ./data/custom/images/ --run_colmap # if use images


# Acknowledgement

* Credits to [ashawkey](https://github.com/ashawkey) for the amazing [torch-ngp](https://github.com/ashawkey/torch-ngp)
    