from torch.utils.data import Dataset
import numpy as np
import torch
import matplotlib.pyplot as plt

class BaseDataset(Dataset):
    """
    Define length and sampling method
    """
    def __init__(self, root_dir, split='train', downsample=1.0, len_per_epoch=1000):
        self.root_dir = root_dir
        self.split = split
        self.downsample = downsample
        self.len_per_epoch = len_per_epoch
        self.patch_size = 64  # oom at 128
        self.patch_coverage = 0.9

    def read_intrinsics(self):
        raise NotImplementedError

    def __len__(self):
        if self.split.startswith('train'):
            return self.len_per_epoch
        return len(self.poses)

    def sample_patch(self, h, w):
        skip = int((min(h, w) * self.patch_coverage) / self.patch_size)
        patch_w_skip = self.patch_size * skip
        patch_h_skip = self.patch_size * skip

        left = torch.randint(0, w - patch_w_skip - 1, (1,))[0]
        left_to_right = torch.arange(left, left + patch_w_skip, skip)
        top = torch.randint(0, h - patch_h_skip - 1, (1,))[0]
        top_to_bottom = torch.arange(top, top + patch_h_skip, skip)

        index_hw = (top_to_bottom * w)[:, None] + left_to_right[None, :]
        # 128, 128 is the patch, patch_h_skip, patch_w_skip are skip
        return index_hw.reshape(-1)

    def __getitem__(self, idx):
        if self.split.startswith('train'):
            # training pose is retrieved in train.py
            if self.ray_sampling_strategy == 'all_images': # randomly select images
                img_idxs = np.random.choice(len(self.poses), self.batch_size)
            elif self.ray_sampling_strategy == 'same_image': # randomly select ONE image
                img_idxs = np.random.choice(len(self.poses), 1)[0]
            # randomly select pixels
            if self.patch_size is None:
                pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size)
            else:
                pix_idxs = self.sample_patch(self.img_wh[1], self.img_wh[0])

            rays = self.rays[img_idxs, pix_idxs]
            sample = {'img_idxs': img_idxs, 'pix_idxs': pix_idxs,
                      'rgb': rays[:, :3]}

            if hasattr(self, 'features') and len(self.features):
                if self.ray_sampling_strategy == 'all_images':
                    # TODO
                    raise NotImplementedError
                elif self.ray_sampling_strategy == 'same_image':
                    feature_map = self.features[img_idxs][None].float()  # chw->1chw
                    u = (pix_idxs % self.img_wh[0] / self.img_wh[0]) * 2 - 1
                    v = (pix_idxs // self.img_wh[0] / self.img_wh[1]) * 2 - 1
                    with torch.no_grad():
                        sampler = torch.tensor(np.stack([u, v], axis=-1)[None, None]).float()  # N2->11N2
                        # TODO: sparse supervision
                        feats = torch.nn.functional.grid_sample(feature_map, sampler, mode='bilinear', align_corners=True)  # 1c1N
                        feats = feats[0, :, 0].T  # 1c1N->cN->Nc
                    sample['feature'] = feats
                    # print("feature_map", feature_map.shape) # 1, 512, 360, 480
                    # print("feats", feats.shape) # 32, 512
                    # print("u", u.shape) # 32
                    # print("pix_idxs", pix_idxs.shape, pix_idxs)
                    # print('feat path', self.features[img_idxs])
            if hasattr(self, 'sam_masks') and len(self.sam_masks):
                sam_crops = self.sam_masks[img_idxs]
                # print('sam path', sam_crops)
                u = (pix_idxs % self.img_wh[0] / self.img_wh[0]) * 2 - 1
                v = (pix_idxs // self.img_wh[0] / self.img_wh[1]) * 2 - 1
                with torch.no_grad():
                    sampler = torch.tensor(np.stack([u, v], axis=-1)[None, None]).float()  # N2->11N2
                    for sam_length in range(len(self.sam_masks[img_idxs])):
                        try:
                            if isinstance(self.sam_masks[img_idxs][sam_length]["segmentation"], np.ndarray):
                                seg = torch.from_numpy(self.sam_masks[img_idxs][sam_length]["segmentation"]).unsqueeze(0).unsqueeze(0).float()
                            else:
                                seg = self.sam_masks[img_idxs][sam_length]["segmentation"].unsqueeze(0).unsqueeze(0).float()
                            # breakpoint()
                            
                            sam_crops[sam_length]["segmentation"] = torch.nn.functional.grid_sample(seg, sampler, mode='bilinear', align_corners=True)  # 1c1N
                            sam_crops[sam_length]["use"] = True
                        except:
                            sam_crops[sam_length]["use"] = False
                            # print('sam error', seg.shape, sampler.shape, len(sam_crops))
                        # sam_crops[sam_length]["segmentation"] = sam_crops[sam_length]["segmentation"]
                        # print("sam_crops", sam_crops[sam_length]["segmentation"].shape) # 32, 512
                sample['sam_masks'] = sam_crops
                check_image = sample['sam_masks'][0]["segmentation"].reshape(64, 64)
                # plt.imshow(check_image)
                # plt.savefig('check_sam_crop.png')
                # plt.imshow(sample['rgb'].cpu().numpy().reshape(64, 64, 3))
                # plt.savefig('check_fig_crop.png')

            if self.rays.shape[-1] == 4: # HDR-NeRF data
                sample['exposure'] = rays[:, 3:]
        else:
            sample = {'pose': self.poses[idx], 'img_idxs': idx}
            if len(self.rays)>0: # if ground truth available
                rays = self.rays[idx]
                sample['rgb'] = rays[:, :3]
                if rays.shape[1] == 4: # HDR-NeRF data
                    sample['exposure'] = rays[0, 3] # same exposure for all rays
            if hasattr(self, 'features') and len(self.features):

                feature_map = self.features[idx].float()  # chw->1chw   
                ch, h, w = feature_map.shape 
                sample['feature'] = feature_map.permute(1, 2, 0).reshape(-1, ch)
            # print(sample.keys(), hasattr(self, 'features'), len(self.features))
        return sample
