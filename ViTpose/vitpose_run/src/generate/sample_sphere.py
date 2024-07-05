import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance_matrix
import vedo

"""
Author： Wanqing Xia
Email: wxia612@aucklanduni.ac.nz

This is the script to calculate the min, mean, max angular distance between camera points sampled 
on a sphere surrounding the object, helps us to determine the sampling density
"""


def fibonacci_sphere(samples=1, radius=1):
    """
    Generates points on the surface of a sphere using the Fibonacci method.
    :param samples: Number of points to generate
    :param radius: Radius of the sphere
    :return: List of points on the sphere surface
    """
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius_at_y = np.sqrt(1 - y * y) * radius  # radius at y, scaled by the desired radius

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y
        y *= radius  # scale y coordinate by the desired radius

        points.append((x, y, z))

    return points


def angular_distance(point1, point2):
    """
    Calculate the angular distance in degrees between two points on a sphere.
    """
    inner_product = np.dot(point1, point2) / (np.linalg.norm(point1) * np.linalg.norm(point2))
    angle_rad = np.arccos(np.clip(inner_product, -1.0, 1.0))
    return np.degrees(angle_rad)


if __name__ == "__main__":
    # Generate 4000 points
    radius = 5
    points = fibonacci_sphere(42, radius)  # radius is 1 for simplicity

    # Calculate distance matrix
    dist_matrix = distance_matrix(points, points)

    # Sort each row in the distance matrix and take the distances to the 5 nearest neighbors
    nearest_dists = np.sort(dist_matrix, axis=1)[:, 1:6]

    # Calculate the angular distances for each point to its 5 nearest neighbors
    angular_dists = []
    for i in range(len(points)):
        for j in range(5):
            neighbor_idx = np.where(dist_matrix[i] == nearest_dists[i, j])[0][0]
            angular_dists.append(angular_distance(points[i], points[neighbor_idx]))

    # Calculate min, max, and mean of the angular distances
    min_angular_dist = np.min(angular_dists)
    max_angular_dist = np.max(angular_dists)
    mean_angular_dist = np.mean(angular_dists)
    print("min angular distance: ", min_angular_dist)
    print("max angular distance:", max_angular_dist)
    print("mean angular distance: ", mean_angular_dist)

    # Load 3D model using vedo
    model_path = '/media/iai-lab/wanqing/YCB_Video_Dataset/models/035_power_drill/textured.obj'
    texture_path = '/media/iai-lab/wanqing/YCB_Video_Dataset/models/035_power_drill/texture_map.png'

    # Load the model
    model = vedo.load(model_path)

    # Load the texture
    texture = vedo.load(texture_path)

    # Apply the texture manually
    model.texture(texture)

    # Create a Plotter, add the model, and display
    plot = vedo.Plotter()
    plot.add(model)
    model.scale(15)  # adjust scaling to fit your scene

    # Define axis lines (simple)
    x_line = vedo.shapes.Line([-5, 0, 0], [5, 0, 0], c='red')  # Red line for the X-axis
    y_line = vedo.shapes.Line([0, -5, 0], [0, 5, 0], c='green')  # Green line for the Y-axis
    z_line = vedo.shapes.Line([0, 0, -5], [0, 0, 5], c='blue')  # Blue line for the Z-axis

    # Add the axis lines to the plot
    plot.add(x_line)
    plot.add(y_line)
    plot.add(z_line)

    # Add points
    for point in points:
        plot += vedo.shapes.Sphere(pos=point, r=0.1, c='b')
    plot.show()
