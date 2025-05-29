import torch

from . import common_utils, voxel_aggregation_utils
from al3d_utils.ops.roiaware_pool3d import roiaware_pool3d_utils

# 计算每个部分中点的数量
def find_num_points_per_part(batch_points, batch_boxes, grid_size):
    """
    Args:
        batch_points: (N, 4)
        batch_boxes: (B, O, 7)
        grid_size: G
    Returns:
        points_per_parts: (B, O, G, G, G)
    """
    # 就是计算每个ROI的每个网格
    assert grid_size > 0

    # 取出当前索引批次和当前点坐标
    batch_idx = batch_points[:, 0]
    batch_points = batch_points[:, 1:4]

    # 存储每部分的点数
    points_per_parts = []

    # 遍历每个batch
    for i in range(batch_boxes.shape[0]):
        # box数量 ---- O * 7
        boxes = batch_boxes[i]
        bs_mask = (batch_idx == i)
        # 取出当前batch的点特征  N * 4
        points = batch_points[bs_mask]
        # 使用GPU加速的方法判断哪些点在当前框内
        # N' * 1;代表每个点对应的框，-1表示不在
        box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(points.unsqueeze(0), boxes.unsqueeze(0)).squeeze(0)
        # 标记在框里的点
        points_in_boxes_mask = box_idxs_of_pts != -1
        # 有效点数量 * 7 ；获取的是每个有效的点对应的框
        box_for_each_point = boxes[box_idxs_of_pts.long()][points_in_boxes_mask]
        # 点坐标转化为相对框的局部坐标
        xyz_local = points[points_in_boxes_mask] - box_for_each_point[:, 0:3]
        # 角度旋转--这样可以保证每个框的处理都是一致的
        xyz_local = common_utils.rotate_points_along_z(
            xyz_local[:, None, :], -box_for_each_point[:, 6]
        ).squeeze(dim=1)
        # Change coordinate frame to corner instead of center of box
        # 转换到角落
        xyz_local += box_for_each_point[:, 3:6] / 2
        # points_in_boxes_gpu gets points slightly outside of box, clamp values to make sure no out of index values
        # 防止坐标越界
        xyz_local = torch.min(xyz_local, box_for_each_point[:, 3:6] - 1e-5)
        # 局部坐标转化为网格坐标
        xyz_local_grid = (xyz_local // (box_for_each_point[:, 3:6] / grid_size))
        # 每个点的框索引与网格坐标相连接  N * 4
        xyz_local_grid = torch.cat((box_idxs_of_pts[points_in_boxes_mask].unsqueeze(-1),
                                    xyz_local_grid), dim=-1).long()
        # 获取网格坐标和对应的网格点的数量--去除重复的框坐标;得到每个重复坐标的计数
        # part-idx-- 4 * X  ; per_part - X 
        part_idxs, points_per_part = xyz_local_grid.unique(dim=0, return_counts=True)
        # 计算每个部分的点数--每个框在每个网格的点数
        points_per_part_dense = torch.sparse_coo_tensor(part_idxs.T, points_per_part, size=(boxes.shape[0], grid_size, grid_size, grid_size)).to_dense()
        points_per_parts.append(points_per_part_dense)

    return torch.stack(points_per_parts)


def find_num_points_per_part_multi(batch_points, batch_boxes, grid_size, max_num_boxes, return_centroid=False):
    """
    Args:
        batch_points: (N, 4)
        batch_boxes: (B, O, 7)
        grid_size: G
        max_num_boxes: M
    Returns:
        points_per_parts: (B, O, G, G, G)
    """
    assert grid_size > 0

    batch_idx = batch_points[:, 0]
    batch_points = batch_points[:, 1:4]

    points_per_parts = []
    for i in range(batch_boxes.shape[0]):
        # import pdb; pdb.set_trace()
        boxes = batch_boxes[i]
        bs_mask = (batch_idx == i)
        points = batch_points[bs_mask]
        box_idxs_of_pts = roiaware_pool3d_utils.points_in_multi_boxes_gpu(points.unsqueeze(0), boxes.unsqueeze(0), max_num_boxes).squeeze(0)
        box_for_each_point = boxes[box_idxs_of_pts.long()]
        xyz_local = points.unsqueeze(1) - box_for_each_point[..., 0:3]
        xyz_local_original_shape = xyz_local.shape
        xyz_local = xyz_local.reshape(-1, 1, 3)
        # Flatten for rotating points
        xyz_local = common_utils.rotate_points_along_z(
            xyz_local, -box_for_each_point.reshape(-1, 7)[:, 6]
        )
        # Change coordinate frame to corner instead of center of box
        xyz_local_corner = xyz_local.reshape(xyz_local_original_shape) + box_for_each_point[..., 3:6] / 2
        # points_in_boxes_gpu gets points slightly outside of box, clamp values to make sure no out of index values
        xyz_local_grid = (xyz_local_corner / (box_for_each_point[..., 3:6] / grid_size))
        points_out_of_range = ((xyz_local_grid < 0) | (xyz_local_grid >= grid_size) | (xyz_local_grid.isnan())).any(-1).flatten()
        xyz_local_grid = torch.cat((box_idxs_of_pts.unsqueeze(-1),
                                    xyz_local_grid), dim=-1).long()
        xyz_local_grid = xyz_local_grid.reshape(-1, xyz_local_grid.shape[-1])
        # Filter based on valid box_idxs
        valid_points_mask = (xyz_local_grid[:, 0] != -1) & (~points_out_of_range)
        xyz_local_grid = xyz_local_grid[valid_points_mask]

        if return_centroid:
            xyz_local = xyz_local[valid_points_mask].squeeze(1)
            centroids, part_idxs, points_per_part = voxel_aggregation_utils.get_centroid_per_voxel(xyz_local, xyz_local_grid)
            points_per_part = torch.cat((points_per_part.unsqueeze(-1), centroids), dim=-1)
            # Sometimes no points in boxes, usually in the first few iterations. Return empty tensor in that case
            if part_idxs.shape[0] == 0:
                points_per_part_dense = torch.zeros((boxes.shape[0], grid_size, grid_size, grid_size, points_per_part.shape[-1]), dtype=points_per_part.dtype, device=points.device)
            else:
                points_per_part_dense = torch.sparse_coo_tensor(part_idxs.T, points_per_part, size=(boxes.shape[0], grid_size, grid_size, grid_size, points_per_part.shape[-1])).to_dense()
        else:
            part_idxs, points_per_part = xyz_local_grid.unique(dim=0, return_counts=True)
            # Sometimes no points in boxes, usually in the first few iterations. Return empty tensor in that case
            if part_idxs.shape[0] == 0:
                points_per_part_dense = torch.zeros((boxes.shape[0], grid_size, grid_size, grid_size), dtype=points_per_part.dtype, device=points.device)
            else:
                points_per_part_dense = torch.sparse_coo_tensor(part_idxs.T, points_per_part, size=(boxes.shape[0], grid_size, grid_size, grid_size)).to_dense()

        points_per_parts.append(points_per_part_dense)

    return torch.stack(points_per_parts)