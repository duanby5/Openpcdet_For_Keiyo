import torch
import torch.nn as nn
from kornia.geometry.linalg import transform_points
from . import transform_utils

# def bilinear_interpolate_torch(im, x, y):
#     """
#     Args:
#         im: (B, H, W, C) [y, x]
#         x: (B, N)
#         y: (B, N)
#
#     Returns:
#
#     """
#     B = im.shape[0]
#
#     x0 = torch.floor(x).long()
#     x1 = x0 + 1
#
#     y0 = torch.floor(y).long()
#     y1 = y0 + 1
#
#     x0 = torch.clamp(x0, 0, im.shape[2] - 1)
#     x1 = torch.clamp(x1, 0, im.shape[2] - 1)
#     y0 = torch.clamp(y0, 0, im.shape[1] - 1)
#     y1 = torch.clamp(y1, 0, im.shape[1] - 1)
#
#     Ia_list = []
#     Ib_list = []
#     Ic_list = []
#     Id_list = []
#     for i in range(B):
#         Ia_list.append(im[i, y0[i], x0[i]])
#         Ib_list.append(im[i, y1[i], x0[i]])
#         Ic_list.append(im[i, y0[i], x1[i]])
#         Id_list.append(im[i, y1[i], x1[i]])
#
#     Ia = torch.stack(Ia_list, dim=0)
#     Ib = torch.stack(Ib_list, dim=0)
#     Ic = torch.stack(Ic_list, dim=0)
#     Id = torch.stack(Id_list, dim=0)
#
#     wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
#     wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
#     wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
#     wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
#     ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
#     return ans

def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans

def image_feature_extractor(point, image_feature, image_stride, lidar_to_cam, cam_to_img):
    '''
    :param point: [B, N1 N2..., 3]
    :param image_feature: [B, C, H, W]
    :param image_stride: S, input image stride relate to origin image
    :param lidar_to_cam: (B, 4, 4), LiDAR to camera frame transformation
    :param cam_to_img: (B, 3, 4), Camera projection matrix
    :return:
        point_feature_in_image: [B, N1 N2..., C]
    '''
    point_coord = point[..., :3]
    B = lidar_to_cam.shape[0]

    # Create transformation matricies
    C_V = lidar_to_cam  # LiDAR -> Camera (B, 4, 4)
    I_C = cam_to_img  # Camera -> Image (B, 3, 4)
    trans = C_V

    # Reshape to match dimensions
    unsqueeze_time = len(point_coord.shape) - 3
    for _ in range(unsqueeze_time):
        trans.unsqueeze(1)
    # trans = trans.reshape(B, 1, 1, 4, 4)

    # Transform to camera frame
    camera_grid = transform_points(trans_01=trans, points_1=point_coord)

    # Project to image
    for _ in range(unsqueeze_time):
        I_C.unsqueeze(1)

    # I_C = I_C.reshape(B, 1, 1, 3, 4)
    image_grid, _ = transform_utils.project_to_image(project=I_C, points=camera_grid)

    image_grid = image_grid / image_stride

    point_features_in_image_sqe = []
    for k in range(B):
        cur_x_idxs = image_grid[k, :, 0]
        cur_y_idxs = image_grid[k, :, 1]
        cur_image_features = image_feature[k].permute(1, 2, 0)  # (H, W, C)
        point_features_in_image_per_channel = bilinear_interpolate_torch(cur_image_features, cur_x_idxs, cur_y_idxs)
        point_features_in_image_sqe.append(point_features_in_image_per_channel.unsqueeze(dim=0))
    point_features_in_image = torch.cat(point_features_in_image_sqe, dim=0)  # (B, N, C0)
    # # Convert depths to depth bins
    # image_depths = transform_utils.bin_depths(depth_map=image_depths, **self.disc_cfg)
    #
    # # Stack to form frustum grid
    # image_depths = image_depths.unsqueeze(-1)
    # frustum_grid = torch.cat((image_grid, image_depths), dim=-1)
    # return frustum_grid
    return point_features_in_image