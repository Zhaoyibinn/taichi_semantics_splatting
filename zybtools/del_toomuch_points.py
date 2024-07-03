import numpy as np
import open3d as o3d
all = np.loadtxt("/home/zhaoyibin/3DRE/3DGS/sat_blender_pose_sem_badlight_taichi/sparse/less_points/points3D.txt",usecols=[0,1,2,3])
ply_origin = o3d.geometry.PointCloud()
ply_origin.points = o3d.utility.Vector3dVector(np.array(all[:,1:4]))
cl, ind = ply_origin.remove_radius_outlier(nb_points=3, radius=0.1)
ok_idx = all[:,0][ind]
# o3d.visualization.draw_geometries([cl])

lines = []
with open('/home/zhaoyibin/3DRE/3DGS/sat_blender_pose_sem_badlight_taichi/sparse/less_points/points3D.txt', 'r') as file:
    for line in file:
        lines.append(line)



less_points = []
i = 0
for line in lines:
    if i<=4:
        less_points.append(line)
    elif int(line.split()[0]) in ok_idx:
        less_points.append(line)
    i = i + 1

with open('/home/zhaoyibin/3DRE/3DGS/sat_blender_pose_sem_badlight_taichi/sparse/less_points/points3D_less.txt', 'w') as file:
    file.writelines(less_points)
print("end")