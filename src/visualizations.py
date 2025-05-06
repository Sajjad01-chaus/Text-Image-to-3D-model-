import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
from IPython.display import Image, display 

 #visualization code in matplotlib
def visualize_simple(mesh, visualize_type="pointcloud"):
   
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    vertices = mesh.vertices
    faces = mesh.faces

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    if visualize_type == "wireframe":
        max_faces = min(len(faces), 15000)  
        for j in range(max_faces):
            face = faces[j]
            x = vertices[face, 0]
            y = vertices[face, 1]
            z = vertices[face, 2]
            ax.plot(x[[0, 1]], y[[0, 1]], z[[0, 1]], color='black', linewidth=0.5, alpha=0.7)
        ax.set_title("3D Model Wireframe Visualization (matplotlib)")
    elif visualize_type == "pointcloud":
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='blue', s=1)
        ax.set_title("3D Model Point Cloud Visualization (matplotlib)")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=30, azim=45)
    output_path = "matplotlib_visualization.png"
    plt.savefig(output_path)
    plt.close(fig)

    display(Image(filename=output_path))  # Displaying the saved image in Colab
    return output_path

def visualize_3d_model(mesh):
    visualize_simple(mesh) 