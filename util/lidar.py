import os

import einops
import torch
import torch.nn.functional as F
from torch import nn
import numba

import numpy as np




labelmap = {
    0: 0,  # "unlabeled"
    1: 0,  # "outlier" mapped to "unlabeled" --------------------------mapped
    10: 1,  # "car"
    11: 2,  # "bicycle"
    13: 5,  # "bus" mapped to "other-vehicle" --------------------------mapped
    15: 3,  # "motorcycle"
    16: 5,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18: 4,  # "truck"
    20: 5,  # "other-vehicle"
    30: 6,  # "person"
    31: 7,  # "bicyclist"
    32: 8,  # "motorcyclist"
    40: 9,  # "road"
    44: 10,  # "parking"
    48: 11,  # "sidewalk"
    49: 12,  # "other-ground"
    50: 13,  # "building"
    51: 14,  # "fence"
    52: 0,  # "other-structure" mapped to "unlabeled" ------------------mapped
    60: 9,  # "lane-marking" to "road" ---------------------------------mapped
    70: 15,  # "vegetation"
    71: 16,  # "trunk"
    72: 17,  # "terrain"
    80: 18,  # "pole"
    81: 19,  # "traffic-sign"
    99: 0,  # "other-object" to "unlabeled" ----------------------------mapped
    252: 1,  # "moving-car" to "car" ------------------------------------mapped
    253: 7,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
    254: 6,  # "moving-person" to "person" ------------------------------mapped
    255: 8,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    256: 5,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    257: 5,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
    258: 4,  # "moving-truck" to "truck" --------------------------------mapped
    259: 5,  # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}

# @numba.jit
def scatter(arrary, index, value):
    for (h, w), v in zip(index, value):
        arrary[h, w] = v
    return arrary


def projection(source, grid, order, H, W):
    assert source.ndim == 2, source.ndim
    C = source.shape[1]
    proj = np.zeros((H, W, C))
    proj = np.asarray(proj, dtype=source.dtype)
    proj = scatter(proj, grid[order], source[order])
    return proj


def point_cloud_to_xyz_image(points, H = 64, W=2048, fov_up=3.0, fov_down=-25.0, is_sorted=True):
    xyz = points[:, :3]  # xyz
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    depth = np.linalg.norm(xyz, ord=2, axis=1)
    order = np.argsort(-depth)
    if not is_sorted:
        fov_up = fov_up / 180.0 * np.pi      # field of view up in rad
        fov_down = fov_down/ 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)
        pitch = np.arcsin(z / depth)
        grid_h = 1.0 - (pitch + abs(fov_down)) / fov
        grid_h = np.clip(np.floor(grid_h * H), 0, H-1)
    else:
        # the i-th quadrant
        # suppose the points are ordered counterclockwise
        quads = np.zeros_like(x)
        quads[(x >= 0) & (y >= 0)] = 0  # 1st
        quads[(x < 0) & (y >= 0)] = 1  # 2nd
        quads[(x < 0) & (y < 0)] = 2  # 3rd
        quads[(x >= 0) & (y < 0)] = 3  # 4th
        diff = np.roll(quads, 1) - quads
        (start_inds,) = np.where(diff == 3)  # number of lines
        inds = list(start_inds) + [len(quads)]  # add the last index
        line_idx = 63  # ...0
        grid_h = np.zeros_like(x)
        for i in reversed(range(len(start_inds))):
            grid_h[inds[i] : inds[i + 1]] = line_idx
            line_idx -= 1
        
    # horizontal grid
    yaw = -np.arctan2(y, x)  # [-pi,pi]
    grid_w = (yaw / np.pi + 1) / 2  # [0,1]
    grid_w = np.clip(np.floor(grid_w * W), 0, W - 1)
    grid = np.stack((grid_h, grid_w), axis=-1).astype(np.int32)
    proj = projection(points, grid, order, H, W)
    return proj, grid

class Coordinate(nn.Module):
    def __init__(self, min_depth, max_depth, shape, drop_const=0) -> None:
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.H, self.W = shape
        self.drop_const = drop_const
        self.register_buffer("angle", self.init_coordmap(self.H, self.W))

    def init_coordmap(self, H, W):
        raise NotImplementedError

    @staticmethod
    def normalize_minmax(tensor, vmin: float, vmax: float):
        return (tensor - vmin) / (vmax - vmin)

    @staticmethod
    def denormalize_minmax(tensor, vmin: float, vmax: float):
        return tensor * (vmax - vmin) + vmin

    def invert_depth(self, norm_depth):
        # depth to inverse depth
        depth = self.denormalize_minmax(norm_depth, self.min_depth, self.max_depth)
        disp = 1 / depth
        norm_disp = self.normalize_minmax(disp, 1 / self.max_depth, 1 / self.min_depth)
        return norm_disp

    def revert_depth(self, norm_disp, norm=True):
        # inverse depth to depth
        disp = self.denormalize_minmax(
            norm_disp, 1 / self.max_depth, 1 / self.min_depth
        )
        depth = 1 / disp
        if norm:
            return self.normalize_minmax(depth, self.min_depth, self.max_depth)
        else:
            return depth

    def pol_to_xyz(self, polar):
        assert polar.dim() == 4
        grid_cos = torch.cos(self.angle)
        grid_sin = torch.sin(self.angle)
        grid_x = polar * grid_cos[:, [0]] * grid_cos[:, [1]]
        grid_y = polar * grid_cos[:, [0]] * grid_sin[:, [1]]
        grid_z = polar * grid_sin[:, [0]]
        return torch.cat((grid_x, grid_y, grid_z), dim=1)

    def xyz_to_pol(self, xyz):
        return torch.norm(xyz, p=2, dim=1, keepdim=True)

    def inv_to_xyz(self, inv_depth, tol=1e-8): # inv_depth [0, 1]
        valid = torch.abs(inv_depth - self.drop_const) > tol
        depth = self.revert_depth(inv_depth)  # [0,1] depth
        depth = depth * (self.max_depth - self.min_depth) + self.min_depth
        depth /= self.max_depth
        depth *= valid
        points = self.pol_to_xyz(depth)
        return points

    def points_to_depth(self, xyz, drop_value=1, tol=1e-8, tau=2):
        assert xyz.ndim == 3
        device = xyz.device

        x = xyz[..., [0]]
        y = xyz[..., [1]]
        z = xyz[..., [2]]
        r = torch.norm(xyz[..., :2], p=2, dim=2, keepdim=True)
        depth_1d = torch.norm(xyz, p=2, dim=2, keepdim=True)
        weight = 1.0 / torch.exp(tau * depth_1d)
        depth_1d = depth_1d * self.max_depth
        weight *= ((depth_1d > self.min_depth) & (depth_1d < self.max_depth)).detach()

        angle_u = torch.atan2(z, r)  # elevation
        angle_v = torch.atan2(y, x)  # azimuth
        angle_uv = torch.cat([angle_u, angle_v], dim=2)
        angle_uv = einops.rearrange(angle_uv, "b n c -> b n 1 c")
        angle_uv_ref = einops.rearrange(self.angle, "b c h w -> b 1 (h w) c")

        _, ids = torch.norm(angle_uv - angle_uv_ref, p=2, dim=3).min(dim=2)
        id_to_uv = einops.rearrange(
            torch.stack(
                torch.meshgrid(
                    torch.arange(self.H, device=device),
                    torch.arange(self.W, device=device),
                ),
                dim=-1,
            ),
            "h w c -> (h w) c",
        )
        uv = F.embedding(ids, id_to_uv).float()
        depth_2d = render.bilinear_rasterizer(uv, weight * depth_1d, (self.H, self.W))
        depth_2d /= render.bilinear_rasterizer(uv, weight, (self.H, self.W)) + 1e-8
        valid = depth_2d != 0
        depth_2d = self.minmax_norm(depth_2d, self.min_depth, self.max_depth)
        depth_2d[~valid] = drop_value

        return depth_2d, valid


class LiDAR(Coordinate):
    def __init__(
        self,
        num_ring,
        num_points,
        angle_file,
        dataset_name=None
    ):
        assert os.path.exists(angle_file), angle_file
        self.angle_file = angle_file
        if dataset_name == 'nuscene':
            from dataset.nuscene import MIN_DEPTH
            from dataset.nuscene import MAX_DEPTH
            min_depth, max_depth = MIN_DEPTH, MAX_DEPTH
        elif dataset_name == 'kitti':
            from dataset.kitti_odometry import MIN_DEPTH
            from dataset.kitti_odometry import MAX_DEPTH
            min_depth, max_depth = MIN_DEPTH, MAX_DEPTH

        super().__init__(
            min_depth=min_depth,
            max_depth=max_depth,
            shape=(num_ring, num_points),
        )

    def init_coordmap(self, H, W):
        angle = torch.load(self.angle_file)[None]
        angle = F.interpolate(angle, size=(H, W), mode="bilinear")
        # fov_up = 3.0 / 180.0 * np.pi     
        # fov_down = -25.0 / 180.0 * np.pi  
        # fov = abs(fov_down) + abs(fov_up)
        # pitch = (1 - (torch.arange(H) / H)) * fov - abs(fov_down)
        # yaw = (1 - (torch.arange(W) / W) * 2) * np.pi
        # pitch_grid, yaw_grid = torch.meshgrid(pitch, yaw)
        # angle = torch.stack([pitch_grid, yaw_grid], dim=0)[None]
        return angle



def range_image_to_point_cloud(depth_range, lidar_intensity, H, W):
    # convert range to euclidean xyz
    fov_up=3.0
    fov_down=-25.0
    fov_up = fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad
    x, y = np.meshgrid(np.arange(0, W), np.arange(0, H))
    x,y= x.astype('float'), y.astype('float')
    x *= 1/W
    y *= 1/H
    yaw = np.pi*(x * 2 - 1)
    pitch = (1.0 - y)*fov - abs(fov_down)
    yaw = yaw.flatten()
    pitch = pitch.flatten()
    depth = depth_range.flatten()
    
    pts = np.zeros((len(yaw), 4))
    pts[:, 0] =  np.cos(yaw) * np.cos(pitch) * depth
    pts[:, 1] =  -np.sin(yaw) * np.cos(pitch) * depth
    pts[:, 2] =  np.sin(pitch) * depth
    if lidar_intensity is not None:
        pts[:, 3] = lidar_intensity.flatten()
    mask = np.logical_and(depth>0.5, depth < 63.0)

    return pts, mask
