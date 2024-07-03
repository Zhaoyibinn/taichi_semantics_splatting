import argparse
import pandas as pd
from taichi_3d_gaussian_splatting.GaussianPointCloudScene import GaussianPointCloudScene
from plyfile import PlyData, PlyElement
import numpy as np
import torch
def save_ply(pointcloud):
    print(pointcloud.head())



def to_ply(ply_path,point_cloud,instance,valid_point_cloud_features):
    valid_point_cloud = point_cloud
    valid_point_cloud_features = valid_point_cloud_features
    xyz = valid_point_cloud
    normals = np.zeros_like(xyz)
    f_sh = valid_point_cloud_features[:, 8:].reshape(-1, 3, 16)
    f_dc = f_sh[..., 0].numpy()
    f_rest = f_sh[..., 1:].reshape(-1, 45).numpy()
    opacities = valid_point_cloud_features[:, 7:8].numpy()
    scale = valid_point_cloud_features[:, 4:7].numpy()
    rotation = valid_point_cloud_features[:, [3, 0, 1, 2]].numpy()

    def construct_list_of_attributes():
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        for i in range(f_rest.shape[1]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(scale.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotation.shape[1]):
            l.append('rot_{}'.format(i))

        l.append("instance")
        return l

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)

    instance = np.array(torch.sigmoid(torch.tensor(instance)).cpu().detach())
    instance = np.argmax(instance, axis=1).reshape((-1, 1))

    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation,instance/2), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(ply_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", type=str, required=True)
    parser.add_argument("--ply_path", type=str, required=True)
    args = parser.parse_args()
    # scene = GaussianPointCloudScene.from_parquet(
    #     args.parquet_path, config=GaussianPointCloudScene.PointCloudSceneConfig(max_num_points_ratio=None))

    scene_df = pd.read_parquet(args.parquet_path)
    feature_columns = [f"cov_q{i}" for i in range(4)] + \
                      [f"cov_s{i}" for i in range(3)] + \
                      [f"alpha{i}" for i in range(1)] + \
                      [f"r_sh{i}" for i in range(16)] + \
                      [f"g_sh{i}" for i in range(16)] + \
                      [f"b_sh{i}" for i in range(16)]

    df_has_color = "r" in scene_df.columns and "g" in scene_df.columns and "b" in scene_df.columns
    point_cloud = scene_df[["x", "y", "z"]].to_numpy()
    instance = scene_df[[f"instance_{i}" for i in range(3)]].to_numpy()
    valid_point_cloud_features = torch.from_numpy(scene_df[feature_columns].to_numpy())

    to_ply(args.ply_path,point_cloud,instance,valid_point_cloud_features)