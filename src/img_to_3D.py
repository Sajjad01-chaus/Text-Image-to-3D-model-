
import torch
import numpy as np
from PIL import Image
import trimesh

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.image_util import load_image

def load_shap_e_models():
  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    xm = load_model('transmitter', device=device)
    model = load_model('image300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    
    return xm, model, diffusion

def image_to_latents(image_array, model, diffusion):

    pil_image = Image.fromarray(image_array)
    image_data = load_image(pil_image)
    
    # latents
    guidance_scale = 3.0  # Adjust as needed
    latents = sample_latents(
        batch_size=1,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(images=[image_data]),
        progress=True,
        use_fp16=True,
        clip_denoised=True,
    )
    
    return latents

def latents_to_mesh(latents, xm):
    
    # Converting latent to mesh
    t = 0.0  #adjusted as needed
    
    # latent at the first batch index
    latent = latents[0]

    with torch.no_grad():
        mesh = xm.decode_to_mesh(latent, t)
    
    vertices = mesh.verts.cpu().numpy()
    faces = mesh.faces.cpu().numpy()
    
    # Trimesh object
    mesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    return mesh_obj

def generate_3d_from_image(image_array):

    try:
        # Load models
        print("Loading Shap-E models")
        xm, model, diffusion = load_shap_e_models()
        
        print("Generating latent representation")
        latents = image_to_latents(image_array, model, diffusion)
        print("Converting to 3D mesh...")
        mesh = latents_to_mesh(latents, xm)
        
        # mesh cleanup
        print("Finalizing mesh")
        if not mesh.is_watertight:
            print("Warning: Mesh is not watertight. Attempting to repair")
            mesh = mesh.fill_holes()
        
        return mesh
        
    except Exception as e:
        print(f"Error generating 3D model from image: {e}")
        return trimesh.creation.box(extents=[1, 1, 1])