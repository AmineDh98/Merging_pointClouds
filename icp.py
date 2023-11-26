import numpy as np
from sklearn.neighbors import NearestNeighbors
#import drawing
from sklearn.metrics import mean_squared_error
import math
import cv2
import numpy as np

from scipy.spatial import cKDTree


# Function to get color from image based on pixel coordinates
def get_color_from_image(image_path, pixel_coords):
    image = cv2.imread(image_path)
    color = image[int(pixel_coords[1]), int(pixel_coords[0])]  # Assuming pixel_coords is (x, y)
    return color

def create_kd_tree(target):
    """
    Creates a KD-tree data structure to speed-up closest points search.

    Parameters:
        - target: the target point set, used to create the KD-Tree. A Numpy array of size num_pts x 3.
    Returns:
        - kdtree: the KD-tree search structure.
    """
    kdtree = cKDTree(target)
    return kdtree

def closest_points_kd_tree(kdtree, source):
    """
    Finds the closest points between the source and target point sets using an already-constructed KD-tree search structure.

    Parameters:
        - kdtree: the KD-tree search structure, built from the target point set.
        - source: the source point set. A Numpy array of size num_pts x 3. Distances will be computed from the source to the target.
    Returns:
        - distances: the distances from each point in the source point set to its closest point in the target point set.
        - indices: the indices of the closest points in the target point set.
    """
    distances, indices = kdtree.query(source, k=1)
    return distances, indices

def icp(source, target, max_iters = 100, max_overlap_dist = 1, relative_error_tolerance = 1e-6):
    """
    The Iterative Closest Point registration method.

    Parameters:
        - source: the source point set, that will be "moved" rigidly towards the target point set. A Numpy array of size num_pts_source x 3.
        - target: the target point set, that will not be moved. A Numpy array of size num_pts_target x 3.
        - max_iters: maximum number of iterations.
        - max_overlap_dist: each point in the source point cloud will try to find a corresponding point in the target point cloud that is, at most, at this distance.
        - relative_error_tolerance: if the change of RMSE from the last iteration is less than this value, the method will stop.  
    Returns:
        - source_tf: the transformed source point set.
    """
    for iterations_counter in range(max_iters):
        kdtree = create_kd_tree(target)
        distances, indices = closest_points_kd_tree(kdtree, source)
       
       
        #just consider closest points that are at most at a given distance
        filtered_source = source[np.squeeze(distances < max_overlap_dist)]
        filtered_target = target[indices[distances < max_overlap_dist]]

        R, t = rigid_transform_3d(filtered_source, filtered_target)

        source_tf = (R @ source.T + t).T
        source = source_tf
        rmsd = math.sqrt(mean_squared_error(filtered_source, filtered_target))
        if rmsd < relative_error_tolerance or iterations_counter == max_iters - 1:
            break

    return source_tf

def rigid_transform_3d(A, B):
    """
    Computes the rigid transformation between two sets of corresponding points.

    Parameters:
        - A, B: the two point sets to register, Numpy arrays of shape num_pts x 3.
   
    Returns:
        - R, t: rotation (3x3) and translation (3x1) matrices transforming the point set A to the point set B (B=R@A+t)
    """
    
    c_a = np.mean(A, axis = 0)
    c_b = np.mean(B, axis = 0)
    P_AC = A - c_a
    P_BC = B - c_b
   
    H = P_AC.T @ P_BC
   
    U, _, V_T = np.linalg.svd(H)
    V = V_T.T
    R =  V @ U.T
    if np.linalg.det(R) < 0:
        V[2] *= -1
        R =  V @ U.T
       
    t = c_b - R @ c_a
    return R, t.reshape((3, 1))

def depth_to_3d(depth_img, camera_matrix, depth_units=0.001, color_img=None):
    """
    Converts a depth image and a color image from an RGBD camera into a colored point set.

    Parameters:
        - depth_img: the depth image (a Numpy array with elements of type uint16).
        - camera_matrix: the pinhole camera matrix with the intrinsics of the camera.
        - depth_units: the scale factor converting depth_image units (i.e., uint16) to meters.
        - color_img: the color image, aligned with the depth_image.
    Return:
        - pts_3d: 3D point set, a Numpy array of size num_pts x 3.
        - pts_colors: the RGB colors for the point set, a Numpy array of size num_pts x 3, values in the range 0..1.
    """
    c = []
    d = []

    for y in range(depth_img.shape[0]):
        for x in range(depth_img.shape[1]):
            if depth_img[y, x] > 0:
                c.append([x, y])
                d.append(depth_img[y, x])

    c = np.array(c)
    d = np.array(d)

    coords = np.hstack((c, np.ones((c.shape[0], 1))))
    coords = coords.T

    coords = np.linalg.inv(camera_matrix) @ coords

    coords = (coords * depth_units * d)

    pts_3d = coords[:3, :].T
    pts_colors = color_img[c[:, 1], c[:, 0]] / 255.0
   
    pts_colors[:, [0, 2]] = pts_colors[:, [2, 0]]
    return pts_3d, pts_colors


