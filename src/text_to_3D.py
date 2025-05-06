import torch
import numpy as np
import trimesh

# Import Shap-E components
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

def load_text_models():
  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Models
    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    
    return xm, model, diffusion

def text_to_latents(text_prompt, model, diffusion):
   
    #  latents from text
    guidance_scale = 15.0  # Adjust as needed
    latents = sample_latents(
        batch_size=1,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[text_prompt]),
        progress=True,
        use_fp16=True,
        clip_denoised=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,      
        sigma_max=160,        
        s_churn=0,
    )
    
    return latents

def latents_to_mesh(latents, xm):
   
    # latent to mesh
    t = 0.0  # Adjust as needed
    latent = latents[0]

    # Decoding the latent to a 3D representation
    with torch.no_grad():
        mesh_utility = decode_latent_mesh(xm, latent)
        verts = mesh_utility.verts.cpu().numpy()
        faces = mesh_utility.faces.cpu().numpy()
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        return mesh

def generate_3d_from_text(text_prompt):
  
    try:
        # Loading models
        print("Loading text-to-3D models...")
        xm, model, diffusion = load_text_models()
        
        print(f"Processing text prompt: '{text_prompt}'")
        print("Generating latent representation...")
        latents = text_to_latents(text_prompt, model, diffusion)
        
        print("Converting to 3D mesh...")
        mesh = latents_to_mesh(latents, xm)
        print("Finalizing mesh...")
        if not mesh.is_watertight:
            print("Warning: Mesh is not watertight. Attempting to repair...")
            mesh = mesh.fill_holes()
        
        return mesh
        
    except Exception as e:
        print(f"Error generating 3D model from text: {e}")
        return trimesh.creation.box(extents=[1, 1, 1])