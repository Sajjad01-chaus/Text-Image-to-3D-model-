
import os
import argparse
from src.input_handler import process_input
from src.img_to_3D import generate_3d_from_image
from src.text_to_3D import generate_3d_from_text
from src.mesh_utils import save_mesh
from src.visualizations import visualize_3d_model

def main():
    
    parser = argparse.ArgumentParser(description='Convert a photo or text prompt to a 3D model')
    parser.add_argument('input', help='Path to image file or text prompt')
    parser.add_argument('--output', default='output', help='Output filename (without extension)')
    parser.add_argument('--format', choices=['obj', 'stl'], default='obj', help='Output format')
    parser.add_argument('--visualize', action='store_true', help='Visualize the 3D model')
    parser.add_argument('--skip_background_removal', action='store_true', 
                        help='Skip background removal for images')
    args = parser.parse_args()
    
    print(f"Processing input: {args.input}")
    
    # Determining input type and process
    input_type, processed_input = process_input(args.input, 
                                              skip_background_removal=args.skip_background_removal)
    
    print(f"Input identified as: {input_type}")
    
    # Generate 3D model based on input type
    if input_type == 'image':
        print("Generating 3D model from image...")
        mesh = generate_3d_from_image(processed_input)
    else:  # text
        print(f"Generating 3D model from text prompt: '{processed_input}'")
        mesh = generate_3d_from_text(processed_input)
    
    # Saving the mesh
    output_path = f"{args.output}.{args.format}"
    save_mesh(mesh, output_path, file_format=args.format)
    print(f"3D model saved to: {output_path}")
    
    # Visualize 
    if args.visualize:
        print("Visualizing 3D model...")
        visualize_3d_model(mesh)

if __name__ == "__main__":
    main()