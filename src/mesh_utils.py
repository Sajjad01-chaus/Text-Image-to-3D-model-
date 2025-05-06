import os
import trimesh
import numpy as np

def clean_mesh(mesh):

    mesh = mesh.copy()

    # Removing duplicate vertices and degenerate faces
    mesh.merge_vertices()
    mesh.remove_degenerate_faces()
 
    if not mesh.is_watertight:
        try:
            mesh.fill_holes()
        except Exception as e:
            print(f"Warning: Could not fill holes in mesh: {e}")
    
    mesh.fix_normals()
    
    return mesh

def simplify_mesh(mesh, target_faces=10000):
 
    if len(mesh.faces) <= target_faces:
        return mesh
    
    try:
        mesh_simplified = mesh.simplify_quadratic_decimation(target_faces)
        return mesh_simplified
    except Exception as e:
        print(f"Warning: Mesh simplification failed: {e}")
        return mesh

def save_mesh(mesh, output_path, file_format='obj'):

    try:
        mesh = clean_mesh(mesh)
        
        if len(mesh.faces) > 10000:
            mesh = simplify_mesh(mesh)
        
        # Ensuring the directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', 
                   exist_ok=True)
        if file_format.lower() == 'obj':
            mesh.export(output_path, file_type='obj')
        elif file_format.lower() == 'stl':
            mesh.export(output_path, file_type='stl')
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        print(f"Mesh saved successfully: {output_path}")
        print(f"Mesh stats: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        return True
        
    except Exception as e:
        print(f"Error saving mesh: {e}")
        return False