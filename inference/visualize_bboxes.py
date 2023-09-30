"""
Load bboxes from a pickle file. Each entry is {instance_id: {"bbox": bbox, "orientation": rotation, "position": center}}
Visualize all bboxes in 3D with different colour for each instance
"""

import sys
sys.path.append(".")
import argparse
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from scipy.linalg import eigh

from inference.mbr import MinimumBoundingRectangle


def get_bbox_vertices(min_vert, max_vert):
    x000 = min_vert
    x111 = max_vert
    x001 = np.array([min_vert[0], min_vert[1], max_vert[2]])
    x010 = np.array([min_vert[0], max_vert[1], min_vert[2]])
    x011 = np.array([min_vert[0], max_vert[1], max_vert[2]])
    x100 = np.array([max_vert[0], min_vert[1], min_vert[2]])
    x101 = np.array([max_vert[0], min_vert[1], max_vert[2]])
    x110 = np.array([max_vert[0], max_vert[1], min_vert[2]])
    return np.array([x000, x001, x010, x011, x100, x101, x110, x111])


def plot_cube(ax, bbox_vertices_world, color):
    x000, x001, x010, x011, x100, x101, x110, x111 = bbox_vertices_world
    # bottom face edges
    ax.plot([x000[0], x001[0]], [x000[1], x001[1]], [x000[2], x001[2]], color=color)
    ax.plot([x000[0], x010[0]], [x000[1], x010[1]], [x000[2], x010[2]], color=color)
    ax.plot([x001[0], x011[0]], [x001[1], x011[1]], [x001[2], x011[2]], color=color)
    ax.plot([x010[0], x011[0]], [x010[1], x011[1]], [x010[2], x011[2]], color=color)
    # top face edges
    ax.plot([x100[0], x101[0]], [x100[1], x101[1]], [x100[2], x101[2]], color=color)
    ax.plot([x100[0], x110[0]], [x100[1], x110[1]], [x100[2], x110[2]], color=color)
    ax.plot([x101[0], x111[0]], [x101[1], x111[1]], [x101[2], x111[2]], color=color)
    ax.plot([x110[0], x111[0]], [x110[1], x111[1]], [x110[2], x111[2]], color=color)
    # vertical edges
    ax.plot([x000[0], x100[0]], [x000[1], x100[1]], [x000[2], x100[2]], color=color)
    ax.plot([x001[0], x101[0]], [x001[1], x101[1]], [x001[2], x101[2]], color=color)
    ax.plot([x010[0], x110[0]], [x010[1], x110[1]], [x010[2], x110[2]], color=color)
    ax.plot([x011[0], x111[0]], [x011[1], x111[1]], [x011[2], x111[2]], color=color)

def filter_pointcloud(points):
    '''
    Filter points that have too few points in the neighborhood radius
    points: N x 3
    return: N x 3
    '''
    # create a KDTree
    tree = KDTree(points)
    # query the tree
    dist, _ = tree.query(points, k=10)
    # filter points
    max_dist = dist[..., -1]
    # keep 70% of the points
    keep = max_dist < np.percentile(max_dist, 70)
    points = points[keep]

    # now diagonal gaussian assumption to filter out outliers
    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)
    # filter points
    keep = np.all(np.abs(points - mean) < 3 * std, axis=-1)
    points = points[keep]
    return points



def get_tight_bbox(points, all_points_instances, method="ellipsoid"):
    '''
    For each instance, get the tightest bounding box, which need not be axis-aligned.
    `method` can be "ellipsoid" or "pca" or "simple"
    Returns: 
    instance_id: {
        bbox: ((min_x, min_y, min_z), (max_x, max_y, max_z)),
        orientation: rotation_matrix of unit direction vectors (x, y, z) in world coordinates
        position: center of bbox in world coordinates
    }
    '''
    all_bboxes = {}
    unique_instances = np.unique(all_points_instances)
    for instance in unique_instances:
        if instance == 0: # background
            continue
        instance_points = points[all_points_instances == instance]
        # subsample 10000 points
        if instance_points.shape[0] > 50000:
            instance_points = instance_points[np.random.choice(instance_points.shape[0], 50000, replace=False)]
        # filter points
        instance_points = filter_pointcloud(instance_points)
        
        ### Ellipsoid fitting method
        if method == "ellipsoid":
            # get the minimum volume ellipsoid
            center, radii, rotation = getMinVolEllipse(instance_points)
            # get the bounding box in local coordinates. So, we only need radii
            bbox = ((-radii[0], -radii[1], -radii[2]), (radii[0], radii[1], radii[2]))
            all_bboxes[instance] = {"bbox": bbox, "orientation": rotation, "position": center}

        ### PCA method
        elif method == "pca":
            pca = PCA(n_components=3)
            pca.fit(instance_points)
            # get the bounding box in local coordinates
            projected_points = pca.transform(instance_points)
            bbox = (np.min(projected_points, axis=0), np.max(projected_points, axis=0))
            all_bboxes[instance] = {"bbox": bbox, "orientation": pca.components_, "position": pca.mean_}

        ### Axis-aligned method
        elif method == "simple":
            center = np.mean(instance_points, axis=0)
            bbox = (np.min(instance_points, axis=0), np.max(instance_points, axis=0))
            # in local coordinates
            bbox = (bbox[0] - center, bbox[1] - center)
            all_bboxes[instance] = {"bbox": bbox, "orientation": np.eye(3), "position": center}

        elif method == "oriented":
            center_3d, bbox, rotation = getMinVolBox(instance_points)
            all_bboxes[instance] = {"bbox": bbox, "orientation": rotation, "position": center_3d}


    return all_bboxes



def getMinVolEllipse(P=None, tolerance=0.01):
    """ Find the minimum volume ellipsoid which holds all the points
    
    Based on work by Nima Moshtagh
    http://www.mathworks.com/matlabcentral/fileexchange/9542
    and also by looking at:
    http://cctbx.sourceforge.net/current/python/scitbx.math.minimum_covering_ellipsoid.html
    Which is based on the first reference anyway!
    
    Here, P is a numpy array of N dimensional points like this:
    P = [[x,y,z,...], <-- one point per line
            [x,y,z,...],
            [x,y,z,...]]
    
    Returns:
    (center, radii, rotation)
    
    """
    (N, d) = np.shape(P)
    d = float(d)

    # Q will be our working array
    Q = np.vstack([np.copy(P.T), np.ones(N)]) 
    QT = Q.T
    
    # initializations
    err = 1.0 + tolerance
    u = (1.0 / N) * np.ones(N)

    # Khachiyan Algorithm
    while err > tolerance:
        V = np.dot(Q, np.dot(np.diag(u), QT))
        M = np.diag(np.dot(QT , np.dot(np.linalg.inv(V), Q)))    # M the diagonal vector of an NxN matrix
        j = np.argmax(M)
        maximum = M[j]
        step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
        new_u = (1.0 - step_size) * u
        new_u[j] += step_size
        err = np.linalg.norm(new_u - u)
        u = new_u

    # center of the ellipse 
    center = np.dot(P.T, u)

    # the A matrix for the ellipse
    A = np.linalg.inv(
                    np.dot(P.T, np.dot(np.diag(u), P)) - 
                    np.array([[a * b for b in center] for a in center])
                    ) / d
                    
    # Get the values we'd like to return
    U, s, rotation = np.linalg.svd(A)
    radii = 1.0/np.sqrt(s)
    
    return (center, radii, rotation)


def getMinVolBox(P):
    hull = ConvexHull(P)

    max_volume = float('inf')
    
    # Iterate through each face of the convex hull
    for face in hull.simplices:
        # points forming the face
        face_points = P[face]

        # find the center of the face
        center = np.mean(face_points, axis=0)

        # find the normal of the face
        normal = np.cross(face_points[1] - face_points[0], face_points[2] - face_points[0])
        normal = normal / np.linalg.norm(normal)

        # two orthogonal vectors in the plane of the face
        v1 = face_points[1] - face_points[0]
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(normal, v1)

        # project all the points onto the plane of the face
        projected_points = np.dot(P - center, np.array([v1, v2]).T) # Nx2
        projected_points_normal = np.dot(P - center, normal) # Nx1

        # find the bounding box in the plane of the face
        bounding_rect = MinimumBoundingRectangle(projected_points)
        corners = np.array(list(bounding_rect.corner_points))

        # find the volume of the box
        volume = bounding_rect.area * (np.max(projected_points_normal) - np.min(projected_points_normal))

        # if this is the smallest box we've found so far, store its info
        if volume < max_volume:
            max_volume = volume
            # get 3d coordinates of the corners of the 3d box
            # corners is 4x2 in the plane of the face defined by v1 and v2
            corners_3d = v1 * corners[:, 0].reshape(-1, 1) + v2 * corners[:, 1].reshape(-1, 1) + center
            # get the other 4 corners by adding or subtracting the normal times the height
            corners_3d = np.vstack([corners_3d + normal * np.min(projected_points_normal), corners_3d + normal * np.max(projected_points_normal)])
            # get the center of the box
            center_3d = np.mean(corners_3d, axis=0)
            # get the rotation of the box, which is not necessarily the same as the v1, v2, normal
            # so, we need to get this from the corners
            # corners_3d is 8x3. Use first 4 corners
            rotation = np.zeros((3, 3))
            rotation[:, 0] = corners_3d[1] - corners_3d[0]
            rotation[:, 1] = corners_3d[3] - corners_3d[0]
            rotation[:, 2] = normal
            rotation = rotation / np.linalg.norm(rotation, axis=0)

            # (min_x, min_y, min_z), (max_x, max_y, max_z) in local coordinates
            corners_local = np.dot(corners_3d - center_3d, rotation.T)
            min_local = np.min(corners_local, axis=0)
            max_local = np.max(corners_local, axis=0)
            bbox = (min_local, max_local)

        return center_3d, bbox, rotation


def getEllipsoidVolume(radii):
    """Calculate the volume of ellipsoid."""
    return 4./3.*np.pi*radii[0]*radii[1]*radii[2]



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bboxes", type=str, required=True, help="Path to the pickle file containing bboxes")
    parser.add_argument("--method", type=str, default="ellipsoid", 
                        # possible values: ellipsoid, pca, axis_aligned
                        required=False)
    args = parser.parse_args()

    # get parent directory of bboxes file
    parent_dir = "/".join(args.bboxes.split("/")[:-1])
    # load pointcloud
    with open(parent_dir + "/pointcloud.pkl", "rb") as f:
        data = pickle.load(f)
    points = data["points"]
    if type(points) == list:
        points = torch.cat(points, dim=0).cpu().numpy()
    instances = data["instances"]

    instance_count = {}
    for instance_id in np.unique(instances):
        instance_count[instance_id] = np.sum(instances==instance_id)
    smallest_instance = min(instance_count, key=instance_count.get)
    print("Smallest instance:", smallest_instance)

    with open(args.bboxes, "rb") as f:
        bboxes = pickle.load(f)
    # bboxes = bboxes[args.method]

    ##### DEBUGGING #####
    # bboxes = get_tight_bbox(points, instances, method="simple")
    # with open(args.bboxes, "wb") as f:
    #     pickle.dump(bboxes, f)
    #####################

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('ortho')
    colors = np.random.rand(max(bboxes.keys())+1, 3)

    for instance_id, box in bboxes.items():
        if instance_id == smallest_instance:
            continue
        bbox, orientation, position = np.array(box["bbox"]), box["orientation"], box["position"]
        bbox_vertices = get_bbox_vertices(bbox[0], bbox[1])
        
        # bbox_worldspace = (np.matmul(orientation, bbox[0].T).T + position, np.matmul(orientation, bbox[1].T).T + position) # (min_x, min_y, min_z), (max_x, max_y, max_z)
        bbox_vertices_world = np.matmul(orientation, bbox_vertices.T).T + position
        instance_points = points[instances==instance_id]
        if instance_points.shape[0] > 10000:
            instance_points = instance_points[np.random.choice(instance_points.shape[0], 10000, replace=False)]
        instance_points = filter_pointcloud(instance_points)

        #### DEBUGGING ####
        instance_points_local = instance_points - position # translate to origin
        instance_points_local = np.matmul(orientation.T, instance_points_local.T).T # rotate to local frame
        # check if points are inside bbox
        print(np.all(np.logical_and(instance_points_local >= bbox[0], instance_points_local <= bbox[1])))
        # breakpoint()
        ###################

        # plot positions and bounding boxes in same colour
        ax.scatter(position[0], position[1], position[2], color=colors[instance_id])
        plot_cube(ax, bbox_vertices_world, colors[instance_id])
        # plot points in black colour
        ax.scatter(instance_points[:,0], instance_points[:,1], instance_points[:,2], color=colors[instance_id], s=0.1)
        
    plt.show()