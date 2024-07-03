import numpy as np
import open3d as o3d
import torch
from chamferdist import ChamferDistance
import time

def o3d_vis(proj_points):
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    proj_points_pcd = o3d.geometry.PointCloud()
    proj_points_pcd.points = o3d.utility.Vector3dVector(proj_points)
    o3d.visualization.draw_geometries([proj_points_pcd])
    return 0
def cal_error(points0,points1):
    p1 = torch.tensor([points0]).cuda()
    p2 = torch.tensor([points1]).cuda()

    # p1 = torch.randn(1, 100, 3).cuda()
    # p2 = torch.randn(1, 50, 3).cuda()
    s = time.time()
    chamferDist = ChamferDistance()
    dist_forward = chamferDist(p1, p2)
    dist_backward = chamferDist(p2, p1)
    print(0.5 * dist_forward.detach().cpu().item() + 0.5 * dist_backward.detach().cpu().item())
    print(f"Time: {time.time() - s} seconds")
    return 0.5 * dist_forward.detach().cpu().item() + 0.5 * dist_backward.detach().cpu().item()


# 读取两个点云
pcd1 = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/sat_cloud_origin.ply")
# pcd2 = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/gaussian_semantics/output/sat_pose_instance4/point_cloud/iteration_1000/point_cloud.ply")
pcd2 = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting/logs/save_offical&sigmod/best_scene_offical.ply")
pcd_apper = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting/logs/save_offical&sigmod/scene_apperance_net.ply")
pcd3 = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting/logs/scene_instance_less.ply")



pcd1.paint_uniform_color([0.5, 0.5, 0])

pcd2.paint_uniform_color([0.5, 0, 0.5])

# o3d.visualization.draw_geometries([pcd1,pcd2])
print("官方的误差为：")
cal_error(pcd1.points,pcd2.points)
print("加入外观嵌入之后的误差为：")
cal_error(pcd1.points,pcd_apper.points)
print("语义之后的误差为")
cal_error(pcd1.points,pcd3.points)
# cal_error(pcd1.points,pcd4.points)

# print("剪枝")
# cal_error(pcd1.points,pcd2_gs1.points)

