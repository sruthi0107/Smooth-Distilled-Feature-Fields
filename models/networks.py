import torch
from torch import nn
import tinycudann as tcnn
import vren
from einops import rearrange
from .custom_functions import TruncExp
import numpy as np
from sklearn.decomposition import PCA

from .rendering import NEAR_DISTANCE


class NGP(nn.Module):
    def __init__(self, scale, rgb_act='Sigmoid', feature_out_dim=None):
        super().__init__()

        self.rgb_act = rgb_act
        self.feature_out_dim = feature_out_dim
        self.query_features = None
        self.positive_ids = [0]

        # scene bounding box
        self.scale = scale
        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3)*scale)
        self.register_buffer('xyz_max', torch.ones(1, 3)*scale)
        self.register_buffer('half_size', (self.xyz_max-self.xyz_min)/2)

        # each density grid covers [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1]
        self.cascades = max(1+int(np.ceil(np.log2(2*scale))), 1)
        self.grid_size = 128
        self.register_buffer('density_bitfield',
            torch.zeros(self.cascades*self.grid_size**3//8, dtype=torch.uint8))

        # constants
        L = 16; F = 2; log2_T = 19; N_min = 16

        self.use_smooth_encoding = True
        if self.use_smooth_encoding:
            b = np.exp(np.log(512*scale/N_min)/(L-1))
            encoding_config = {
                "otype": "Composite",
	        "nested": [
                    {
                        "otype": "Grid",
	                "type": "Hash",
                        "n_levels": L,
                        "n_features_per_level": F,
                        "log2_hashmap_size": log2_T,
                        "base_resolution": N_min,
                        "per_level_scale": b,
                        "n_dims_to_encode": 3,
                        "interpolation": "Linear"
                    },
                    {
	                "otype": "Frequency",
	                "n_frequencies": 10,
                        "n_dims_to_encode": 3,
                    }
                ]}
            # TODO: bypass network to rgb
            #"""
            self.bypass_rgb_net = tcnn.NetworkWithInputEncoding(
                n_input_dims=3,
                n_output_dims=3,
                encoding_config={
                    "otype": "Frequency",
                    "n_frequencies": 10,
                },
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                }
            )
            #"""
        else:
            b = np.exp(np.log(2048*scale/N_min)/(L-1))
            encoding_config = {
                "otype": "Grid",
	        "type": "Hash",
                "n_levels": L,
                "n_features_per_level": F,
                "log2_hashmap_size": log2_T,
                "base_resolution": N_min,
                "per_level_scale": b,
                "interpolation": "Linear"
            }

        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')
        self.xyz_encoder = \
            tcnn.NetworkWithInputEncoding(
                n_input_dims=6 if self.use_smooth_encoding else 3,
                n_output_dims=32 if self.use_smooth_encoding else 16,
                encoding_config=encoding_config,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 2 if self.use_smooth_encoding else 1,
                }
            )

        self.dir_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            )

        self.rgb_net = \
            tcnn.Network(
                n_input_dims=48 if self.use_smooth_encoding else 32, n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                }
            )

        if self.rgb_act == 'None': # rgb_net output is log-radiance
            for i in range(3): # independent tonemappers for r,g,b
                tonemapper_net = \
                    tcnn.Network(
                        n_input_dims=1, n_output_dims=1,
                        network_config={
                            "otype": "FullyFusedMLP",
                            "activation": "ReLU",
                            "output_activation": "Sigmoid",
                            "n_neurons": 64,
                            "n_hidden_layers": 1,
                        }
                    )
                setattr(self, f'tonemapper_net_{i}', tonemapper_net)

        if self.feature_out_dim is not None:
            L = 16; F = 2; log2_T = 19; N_min = 16
            # b = np.exp(np.log(2048*scale/N_min)/(L-1))
            b = np.exp(np.log(128*scale/N_min)/(L-1))  # feature is of lower-frequency
            self.feature_encoder = tcnn.NetworkWithInputEncoding(
                n_input_dims=6, n_output_dims=self.feature_out_dim,
                encoding_config={
                    "otype": "Composite",
	            "nested": [
                        {
                            "otype": "Grid",
	                    "type": "Hash",
                            "n_levels": L,
                            "n_features_per_level": F,
                            "log2_hashmap_size": log2_T,
                            "base_resolution": N_min,
                            "per_level_scale": b,
                            "n_dims_to_encode": 3,
                            "interpolation": "Linear"
                        },
                        {
	                    "otype": "Frequency",
	                    "n_frequencies": 8,
                            "n_dims_to_encode": 3,
                        }
                    ]},
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": 2,
                }
            )
            # TODO: separate the final layer and make efficient queried dot product by layer-query fusion
            """
            self.basenet_to_feature = tcnn.Network(
                n_input_dims=32 if self.use_smooth_encoding else 16,
                n_output_dims=self.feature_out_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 32,
                    "n_hidden_layers": 1,
                }
            )
            """
        else:
            self.feature_encoder = None

    def density(self, x, return_feat=False):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs:
            sigmas: (N)
        """
        x = (x-self.xyz_min)/(self.xyz_max-self.xyz_min)
        if self.use_smooth_encoding:
            h = self.xyz_encoder(torch.cat([x, x], dim=-1))
        else:
            h = self.xyz_encoder(x)
        sigmas = TruncExp.apply(h[:, 0])
        if return_feat: return sigmas, h
        return sigmas

    def log_radiance_to_rgb(self, log_radiances, **kwargs):
        """
        Convert log-radiance to rgb as the setting in HDR-NeRF.
        Called only when self.rgb_act == 'None' (with exposure)

        Inputs:
            log_radiances: (N, 3)

        Outputs:
            rgbs: (N, 3)
        """
        if 'exposure' in kwargs:
            log_exposure = torch.log(kwargs['exposure'])
        else: # unit exposure by default
            log_exposure = 0

        out = []
        for i in range(3):
            inp = log_radiances[:, i:i+1]+log_exposure
            out += [getattr(self, f'tonemapper_net_{i}')(inp)]
        rgbs = torch.cat(out, 1)
        return rgbs

    def forward(self, x, d, **kwargs):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions

        Outputs:
            sigmas: (N)
            rgbs: (N, 3)
        """
        sigmas, h = self.density(x, return_feat=True)
        if kwargs.get('detach_geometry', False):
            sigmas = sigmas.detach()
            h = h.detach()
            # .detach() enables the network to completely preserve geometry,
            # because self.xyz_encoder is not updated.
            # However, it limits the degree of freedom of optimization and
            # could make difficulty of large appearance changes.
            # So, instead of the complete .detach(),
            # this straight-through estimator-like trick enables the network
            # to optimize self.xyz_encoder 'a little'.
            # h = h.detach() * 0.999 + h * 0.001
        d = d/torch.norm(d, dim=1, keepdim=True)
        d = self.dir_encoder((d+1)/2)
        rgbs = self.rgb_net(torch.cat([d, h], 1))

        if self.use_smooth_encoding:
            rgbs = rgbs + self.bypass_rgb_net((x-self.xyz_min)/(self.xyz_max-self.xyz_min))

        if self.rgb_act == 'None': # rgbs is log-radiance
            if kwargs.get('output_radiance', False): # output HDR map
                rgbs = TruncExp.apply(rgbs)
            else: # convert to LDR using tonemapper networks
                rgbs = self.log_radiance_to_rgb(rgbs, **kwargs)
        else:
            assert self.rgb_act == 'Sigmoid'
            rgbs = torch.sigmoid(rgbs)

        if self.feature_encoder is not None and not kwargs.get('skip_feature', False):
            features = self.encode_feature(x)
            # features = features + self.basenet_to_feature(h.detach())
            return sigmas, rgbs, features
        else:
            return sigmas, rgbs

    def encode_feature(self, x):
        x = (x-self.xyz_min)/(self.xyz_max-self.xyz_min)
        if self.use_smooth_encoding:
            features = self.feature_encoder(torch.cat([x, x], dim=-1))
        else:
            features = self.feature_encoder(x)
        return features

    @torch.no_grad()
    def get_all_cells(self):
        """
        Get all cells from the density grid.
        
        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        indices = vren.morton3D(self.grid_coords).long()
        cells = [(indices, self.grid_coords)] * self.cascades

        return cells

    @torch.no_grad()
    def sample_uniform_and_occupied_cells(self, M, density_threshold):
        """
        Sample both M uniform and occupied cells (per cascade)
        occupied cells are sample from cells with density > @density_threshold
        
        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        cells = []
        for c in range(self.cascades):
            # uniform cells
            coords1 = torch.randint(self.grid_size, (M, 3), dtype=torch.int32,
                                    device=self.density_grid.device)
            indices1 = vren.morton3D(coords1).long()
            # occupied cells
            indices2 = torch.nonzero(self.density_grid[c]>density_threshold)[:, 0]
            if len(indices2)>0:
                rand_idx = torch.randint(len(indices2), (M,),
                                         device=self.density_grid.device)
                indices2 = indices2[rand_idx]
            coords2 = vren.morton3D_invert(indices2.int())
            # concatenate
            cells += [(torch.cat([indices1, indices2]), torch.cat([coords1, coords2]))]

        return cells

    @torch.no_grad()
    def mark_invisible_cells(self, K, poses, img_wh, chunk=64**3):
        """
        mark the cells that aren't covered by the cameras with density -1
        only executed once before training starts

        Inputs:
            K: (3, 3) camera intrinsics
            poses: (N, 3, 4) camera to world poses
            img_wh: image width and height
            chunk: the chunk size to split the cells (to avoid OOM)
        """
        N_cams = poses.shape[0]
        self.count_grid = torch.zeros_like(self.density_grid)

        # w2c_R = poses[:, :3, :3].mT # (N_cams, 3, 3) batch transpose
        w2c_R = poses[:, :3, :3].transpose(1, 2) # (N_cams, 3, 3) batch transpose
        #w2c_R = rearrange(poses[:, :3, :3], 'n a b -> n b a') # (N_cams, 3, 3)

        w2c_T = -w2c_R@poses[:, :3, 3:] # (N_cams, 3, 1)
        cells = self.get_all_cells()
        for c in range(self.cascades):
            indices, coords = cells[c]
            for i in range(0, len(indices), chunk):
                xyzs = coords[i:i+chunk]/(self.grid_size-1)*2-1
                s = min(2**(c-1), self.scale)
                half_grid_size = s/self.grid_size
                xyzs_w = (xyzs*(s-half_grid_size)).T # (3, chunk)
                xyzs_c = w2c_R @ xyzs_w + w2c_T # (N_cams, 3, chunk)
                uvd = K @ xyzs_c # (N_cams, 3, chunk)
                uv = uvd[:, :2]/uvd[:, 2:] # (N_cams, 2, chunk)
                in_image = (uvd[:, 2]>=0)& \
                           (uv[:, 0]>=0)&(uv[:, 0]<img_wh[0])& \
                           (uv[:, 1]>=0)&(uv[:, 1]<img_wh[1])
                covered_by_cam = (uvd[:, 2]>=NEAR_DISTANCE)&in_image # (N_cams, chunk)
                # if the cell is visible by at least one camera
                self.count_grid[c, indices[i:i+chunk]] = \
                    count = covered_by_cam.sum(0)/N_cams

                too_near_to_cam = (uvd[:, 2]<NEAR_DISTANCE)&in_image # (N, chunk)
                # if the cell is too close (in front) to any camera
                too_near_to_any_cam = too_near_to_cam.any(0)
                # a valid cell should be visible by at least one camera and not too close to any camera
                valid_mask = (count>0)&(~too_near_to_any_cam)
                self.density_grid[c, indices[i:i+chunk]] = \
                    torch.where(valid_mask, 0., -1.)

    @torch.no_grad()
    def update_density_grid(self, density_threshold, warmup=False, decay=0.95, erode=False):
        density_grid_tmp = torch.zeros_like(self.density_grid)
        if warmup: # during the first steps
            cells = self.get_all_cells()
        else:
            cells = self.sample_uniform_and_occupied_cells(self.grid_size**3//4,
                                                           density_threshold)
        # infer sigmas
        for c in range(self.cascades):
            indices, coords = cells[c]
            s = min(2**(c-1), self.scale)
            half_grid_size = s/self.grid_size
            xyzs_w = (coords/(self.grid_size-1)*2-1)*(s-half_grid_size)
            # pick random position in the cell by adding noise in [-hgs, hgs]
            xyzs_w += (torch.rand_like(xyzs_w)*2-1) * half_grid_size
            density_grid_tmp[c, indices] = self.density(xyzs_w)

        if erode:
            # My own logic. decay more the cells that are visible to few cameras
            decay = torch.clamp(decay**(1/self.count_grid), 0.1, 0.95)
        self.density_grid = \
            torch.where(self.density_grid<0,
                        self.density_grid,
                        torch.maximum(self.density_grid*decay, density_grid_tmp))

        mean_density = self.density_grid[self.density_grid>0].mean().item()

        vren.packbits(self.density_grid, min(mean_density, density_threshold),
                      self.density_bitfield)

    def calculate_selection_score_from_xyz(self, x, query_features=None):
        features = self.encode_feature(x)
        return self.calculate_selection_score(features, query_features)

    def calculate_selection_score(self, features, query_features=None):
        features /= features.norm(dim=-1, keepdim=True)
        if query_features is None:
            query_features = self.query_features
        query_features /= query_features.norm(dim=-1, keepdim=True)
        scores = features.half() @ query_features.T.half()  # (N_points, n_texts) # 
        # features_r = features.reshape(-1, 512)
        # features_r = features_r.cpu().numpy()
        # features_r = np.nan_to_num(features_r)
        # pca = PCA(n_components=3)
        # pca.fit(features_r)
        # PCA(n_components=3)
        # fes = pca.transform(features_r)
        # fes = fes.reshape(360, 480, 4, 3)
        # feso = fes[:, :, 0, 3]
        # feso = fes[:, :, 0, :]
        # feso = (feso - np.min(feso))/(np.max(feso)-np.min(feso))
        # feso = (feso * 255).astype(np.uint8)
        # imageio.imsave('features_nomask.png', feso)

        # scores_apple = scores[:, 0]
        # scores_apple = _scores.reshape(360, 480, 4)
        # scores_apple = scores_apple[:, :, 0]
        # scores_apple = scores_apple.cpu().numpy()
        # scores_apple = np.nan_to_num(scores_apple)
        # scores_apple = (scores_apple - np.min(scores_apple))/(np.max(scores_apple)-np.min(scores_apple))
        # scores_apple = (scores_apple * 255).astype(np.uint8)
        # imageio.imsave('scores_apple_aftersam.png', scores_apple)

        if scores.shape[-1] == 1:
            score_threshold = self.score_threshold if self.score_threshold is not None else 0.4
            scores = scores[:, 0]  # (N_points,)
            scores = (scores >= score_threshold).float()
        else:
            scores = torch.nn.functional.softmax(scores, dim=-1)  # (N_points, n_texts)
            # if False: #self.score_threshold is not None:
            if self.score_threshold is not None:
                # if sam_features is not None:
                    # breakpoint()
                self.score_threshold = 0.25
                scores = scores[:, self.positive_ids].sum(-1)  # (N_points, )
                scores = (scores >= self.score_threshold).float()
            else:
                scores[:, self.positive_ids[0]] = scores[:, self.positive_ids].sum(-1)  # (N_points, )
                scores = torch.isin(torch.argmax(scores, dim=-1), torch.tensor(self.positive_ids).cuda()).float()
        return scores

    def sample_density(self, density_threshold, warmup=False):
        density_grid_tmp = 0
        if warmup: # during the first steps
            cells = self.get_all_cells()
        else:
            cells = self.sample_uniform_and_occupied_cells(self.grid_size**3//4,
                                                           density_threshold)
        # infer sigmas
        for c in range(self.cascades):
            indices, coords = cells[c]
            s = min(2**(c-1), self.scale)
            half_grid_size = s/self.grid_size
            xyzs_w = (coords/(self.grid_size-1)*2-1)*(s-half_grid_size)
            # pick random position in the cell by adding noise in [-hgs, hgs]
            xyzs_w += (torch.rand_like(xyzs_w)*2-1) * half_grid_size
            density_grid_tmp = density_grid_tmp + self.density(xyzs_w).mean()

        return density_grid_tmp
