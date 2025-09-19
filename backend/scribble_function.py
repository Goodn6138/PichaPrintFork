import os
import numpy as np
import torch
import trimesh
from huggingface_hub import snapshot_download
from PIL import Image
from triposg.pipelines.pipeline_triposg_scribble import TripoSGScribblePipeline


@torch.no_grad()
def generate_glb_from_scribble(
    image_path: str,
    prompt: str,
    output_path: str = "./output.glb",
    seed: int = 42,
    num_inference_steps: int = 16,
    scribble_confidence: float = 0.4,
    prompt_confidence: float = 1.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """
    Generate a 3D mesh (.glb) file from an input image and text prompt using TripoSG-Scribble.

    Args:
        image_path (str): Path to input sketch/image.
        prompt (str): Text description for the model.
        output_path (str): Where to save the generated .glb file.
        seed (int): Random seed for reproducibility.
        num_inference_steps (int): Number of inference steps (default: 16).
        scribble_confidence (float): Confidence for scribble input.
        prompt_confidence (float): Confidence for prompt text.
        device (str): "cuda" or "cpu".
        dtype (torch.dtype): torch.float16 for speed on GPU, torch.float32 for CPU.
    """

    # Load pretrained weights if not already present
    weights_dir = "pretrained_weights/TripoSG-scribble"
    if not os.path.exists(weights_dir):
        snapshot_download(repo_id="VAST-AI/TripoSG-scribble", local_dir=weights_dir)

    # Init pipeline
    pipe: TripoSGScribblePipeline = TripoSGScribblePipeline.from_pretrained(weights_dir).to(device, dtype)

    # Load image
    img_pil = Image.open(image_path).convert("RGB")

    # Run inference
    outputs = pipe(
        image=img_pil,
        prompt=prompt,
        generator=torch.Generator(device=device).manual_seed(seed),
        num_inference_steps=num_inference_steps,
        guidance_scale=0,  # CFG-distilled model
        attention_kwargs={
            "cross_attention_scale": prompt_confidence,
            "cross_attention_2_scale": scribble_confidence,
        },
        use_flash_decoder=False,
        dense_octree_depth=8,
        hierarchical_octree_depth=8,
    ).samples[0]

    # Convert to trimesh object
    mesh = trimesh.Trimesh(outputs[0].astype(np.float32), np.ascontiguousarray(outputs[1]))

    # Export GLB
    mesh.export(output_path)
    print(f"✅ Mesh saved to {output_path}")

    return output_path