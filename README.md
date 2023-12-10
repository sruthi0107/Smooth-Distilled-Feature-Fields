# Smoothing in Distilled Feature Fields

Built on: [distilled feature fields (DFFs)](https://pfnet-research.github.io/distilled-feature-fields/) (Kobayashi et al. NeurIPS 2022).
This is a simpler and faster demo codebase of [distilled feature fields (DFFs)](https://pfnet-research.github.io/distilled-feature-fields/) (Kobayashi et al. NeurIPS 2022).

NOTES:
1) For the 3 techniques that we tested in this project, we have 3 separate branches namely - total_variation, bilateral_filtering and sam_for_conv corresponding to total variation, bilateral filtering and the sam guided smoothing methods. The master branch contains the baseline code for DFFs
2) Each of the three branches is structured similarly. The train.py file in each of these branches contains sections of code responsible for adding regularization (TV and Bilateral) and performing smoothing (SAM guided). These can be found within the training_step() function, specifically in the feature_loss section.

Visualization of feature field before and after additional smoothing. 



<p float="left">
  <img src="https://github.com/umangi-jain/smooth-dff/blob/sam_for_conv/demos/vegetable/005.png" width="30%" />
  <img src="https://github.com/umangi-jain/smooth-dff/blob/sam_for_conv/demos/vegetable/005_f.png" width="30%" />
  <img src="https://github.com/umangi-jain/smooth-dff/blob/sam_for_conv/demos/vegetable/005_s.png" width="30%" />
</p>

Outputs from editing operations:






https://github.com/umangi-jain/smooth-dff/assets/25801418/4f34eef7-09ad-41d6-86a8-450c9f1d255f



https://github.com/umangi-jain/smooth-dff/assets/25801418/d091f6a6-4452-419d-ae79-b0ec081fd923



https://github.com/umangi-jain/smooth-dff/assets/25801418/29e4bf97-6c36-416b-a4b7-7bdb9e2bc0fd


Setup
```
python -m pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1--index-url https://download.pytorch.org/whl/cu121
python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.1+cu121.html

python -m pip install -r requirements.txt
python -m pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
git submodule update --init --recursive
cd apex && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && cd ..
python -m pip install models/csrc/
```

(Download [a sample dataset](https://github.com/pfnet-research/distilled-feature-fields/releases/download/tmp/sample_dataset.zip))

Train
- `--root_dir` is the dataset of images with poses.
- `--feature_directory` is the dataset of feature maps for distillation. `--feature_dim` matches the dimension of them.
```
python train.py --root_dir sample_dataset --dataset_name colmap --exp_name exp_v1 --downsample 0.25 --num_epochs 4 --batch_size 4096 --scale 4.0 --ray_sampling_strategy same_image --feature_dim 512 --random_bg --feature_directory sample_dataset/rgb_feature_langseg
```

Render with Edit
- Modify `--edit_config` or codebase itself for other editings.
- Set `--ckpt_path` with the checkpoint above.
```
python render.py --root_dir sample_dataset --dataset_name colmap --downsample 0.25 --scale 4.0 --ray_sampling_strategy same_image --feature_dim 512 --ckpt_path ckpts/colmap/exp_v1_clip/epoch\=0_slim.ckpt --edit_config query.yaml
# ls ./renderd_*.png
# ffmpeg -framerate 30 -i ./rendered_%03d.png -vcodec libx264 -pix_fmt yuv420p -r 30 video.mp4
```


## With New Scene

### Prepare Posed Images

colmap
```
colmap feature_extractor --ImageReader.camera_model OPENCV --SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pooling=true --ImageReader.single_camera 1 --database_path sample_dataset/database.db --image_path sample_dataset/images --SiftExtraction.use_gpu=false
colmap exhaustive_matcher --SiftMatching.guided_matching=true --database_path sample_dataset/database.db --SiftMatching.use_gpu=false
mkdir sample_dataset/sparse
colmap mapper --database_path sample_dataset/database.db --image_path sample_dataset/images --output_path sample_dataset/sparse
colmap bundle_adjuster --input_path sample_dataset/sparse/0 --output_path sample_dataset/sparse/0 --BundleAdjustment.refine_principal
_point 1
colmap image_undistorter --image_path sample_dataset/images --input_path sample_dataset/sparse/0 --output_path sample_dataset_undis
--output_type COLMAP
```

### Encode Features by Teacher Network

Setup LSeg
```
cd distilled_feature_field/encoders/lseg_encoder
pip install -r requirements.txt
pip install git+https://github.com/zhanghang1989/PyTorch-Encoding/
```

Download the LSeg model file `demo_e200.ckpt` from [the Google drive](https://drive.google.com/file/d/1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb/view?usp=sharing).

Encode and save
```
python -u encode_images.py --backbone clip_vitl16_384 --weights demo_e200.ckpt --widehead --no-scaleinv --outdir ../../sample_dataset_undis/rgb_feature_langseg --test-rgb-dir ../../sample_dataset_undis/images
```
This may produces large feature map files in `--outdir` (100-200MB per file).

Run train.py. If reconstruction fails, change `--scale 4.0` to smaller or larger values, e.g., `--scale 1.0` or `--scale 16.0`.


### Citation
The codebase for this project is derived from [DFFs](https://github.com/pfnet-research/distilled-feature-fields)

The codebase of NeRF is derived from [ngp_pl](https://github.com/kwea123/ngp_pl/commit/6b2a66928d032967551ab98d5cd84c7ef1b83c3d) (6b2a669, Aug 30 2022)

The codebase of `encoders/lseg_encoder` is derived from [lang-seg](https://github.com/isl-org/lang-seg)





