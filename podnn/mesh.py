import os
import sys
import time
import pandas as pd
import numpy as np
import meshio
import re


def create_linear_mesh(x_min, x_max, n_x,
                       y_min=0, y_max=0, n_y=0,
                       z_min=0, z_max=0, n_z=0):
    dim = 1
    n_xyz = n_x

    x = np.linspace(x_min, x_max, n_x).reshape((n_x, 1))

    if n_y > 0:
        dim += 1
        n_xyz *= n_y
        y = np.linspace(y_min, y_max, n_y).reshape((n_y, 1))

        if n_z > 0:
            dim += 1
            n_xyz *= n_z
            z = np.linspace(z_min, z_max, n_z).reshape((n_z, 1))

            X, Y, Z = np.meshgrid(x, y, z)
            Xflat = X.reshape((n_xyz, 1))
            Yflat = Y.reshape((n_xyz, 1))
            Zflat = Z.reshape((n_xyz, 1))
            idx = np.array(range(1, n_xyz + 1)).reshape((n_xyz, 1))
            return np.hstack((idx, Xflat, Yflat, Zflat))

        X, Y = np.meshgrid(x, y)
        Xflat, Yflat = X.reshape((n_xyz, 1)), Y.reshape((n_xyz, 1))
        idx = np.array(range(1, n_xyz + 1)).reshape((n_xyz, 1))
        return np.hstack((idx, Xflat, Yflat))

    idx = np.array(range(1, n_xyz + 1)).reshape((n_xyz, 1))
    return np.hstack((idx, x))


# From https://stackoverflow.com/a/5967539
def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def read_vtk(filename):
    vtk = meshio.read(filename, file_format="vtk")
    U = np.zeros((vtk.points.shape[0], 3))
    U[:, 0] = vtk.point_data["h"]
    U[:, 1] = vtk.point_data["velocity"][:, 0]
    U[:, 2] = vtk.point_data["velocity"][:, 1]
    return U.T, vtk.points

def read_multi_space_sol_input_mesh(n_s, n_t, d_t, idx, x_u_mesh_path,
                                    mu_mesh_path, mu_mesh_idx,
                                    n_s_0=0):
    st = time.time()
    x_mesh = None
    U = None
    X_v = None
    # Number of parameters, 1+others
    n_p = 1 + len(mu_mesh_idx)
    mu = np.loadtxt(mu_mesh_path, skiprows=1)[:, mu_mesh_idx]
    # Get dirs
    for root, dirs, _ in os.walk(x_u_mesh_path):
        # Sort them naturally, 0-9, 10-19, ...
        picked_dirs = sorted(dirs, key=natural_keys)[n_s_0:n_s+n_s_0]
        # Pick the ones associated with results
        picked_dirs = filter(lambda x: x.startswith("multi_"), picked_dirs) 
        # For filtered/sorted dir (each sample)
        for i, name in enumerate(picked_dirs):
            print(f"Loading sample #{i+1}")
            # Get files of directories
            for sub_root, _, files in os.walk(os.path.join(root, name)):
                t_0 = 0.
                # Sorting and picking the righ ones
                picked_files = filter(lambda file: file.startswith("0_FV-Paraview"), files)
                picked_files = sorted(picked_files, key=natural_keys)
                # For filtered/sorted files
                for j, file in enumerate(picked_files[:n_t]):
                    t_j  = t_0 + j*d_t
                    # Parse the file
                    U_ij, points = read_vtk(os.path.join(sub_root, file))
                    # For the first file, initialize the constant mesh and size
                    if i == 0:
                        U = np.zeros((U_ij.shape[0], U_ij.shape[1], n_t, n_s))
                        x_mesh = points
                        X_v = np.zeros((n_s*n_t, n_p))
                    # Append to the fat matrix
                    U[:, :, j, i] = U_ij
                    X_v[i*n_t + j, :] = np.array([np.array(t_j), mu[i, 0]]).T
    return x_mesh, U, X_v



    print("Loading " + mu_mesh_path + "")
    X_v = np.loadtxt(mu_mesh_path)[:, 0:1]

    print("Loading " + x_u_mesh_path + "")
    x_u_mesh = pd.read_table(x_u_mesh_path,
                             header=None,
                             delim_whitespace=True).to_numpy()
    print(f"Loaded in {time.time() - st} sec.")

    idx_i = idx[0]
    idx_x = idx[1]
    idx_u = idx[2]
    n_xyz = int(x_u_mesh.shape[0] / n_s)
    x_mesh = x_u_mesh[:n_xyz, idx_i + idx_x]
    u_mesh = x_u_mesh[:, idx_u]

    return x_mesh, u_mesh, X_v


def read_space_sol_input_mesh(n_s, idx, x_u_mesh_path, mu_mesh_path):
    st = time.time()
    print("Loading " + mu_mesh_path + "")
    X_v = np.loadtxt(mu_mesh_path)[:, 0:1]

    print("Loading " + x_u_mesh_path + "")
    x_u_mesh = pd.read_table(x_u_mesh_path,
                             header=None,
                             delim_whitespace=True).to_numpy()
    print(f"Loaded in {time.time() - st} sec.")

    idx_i = idx[0]
    idx_x = idx[1]
    idx_u = idx[2]
    n_xyz = int(x_u_mesh.shape[0] / n_s)
    x_mesh = x_u_mesh[:n_xyz, idx_i + idx_x]
    u_mesh = x_u_mesh[:, idx_u]

    return x_mesh, u_mesh, X_v


if __name__ == "__main__":
    print(create_linear_mesh(0, 1, 10))
    print(create_linear_mesh(0, 1, 10, 1, 2, 5))
    print(create_linear_mesh(0, 1, 2, 1, 2, 5, 2, 3, 3))
