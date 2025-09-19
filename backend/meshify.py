import torch
import numpy as np
import trimesh
from PIL import Image
import pymeshlab

from triposg.pipelines.pipeline_triposg import TripoSGPipeline
from image_process import prepare_image
from briarmbg import BriaRMBG


def mesh_to_pymesh(vertices, faces):
    mesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(mesh)
    return ms

def pymesh_to_trimesh(mesh):
    verts = mesh.vertex_matrix()
    faces = mesh.face_matrix()
    return trimesh.Trimesh(vertices=verts, faces=faces)

def simplify_mesh(mesh: trimesh.Trimesh, n_faces: int):
    if mesh.faces.shape[0] > n_faces:
        ms = mesh_to_pymesh(mesh.vertices, mesh.faces)
        ms.meshing_merge_close_vertices()
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=n_faces)
        return pymesh_to_trimesh(ms.current_mesh())
    else:
        return mesh


@torch.no_grad()
def convert_image_to_glb(
    image_input: str,
    output_path: str = "./output.glb",
    seed: int = 42,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    faces: int = -1,
    device: str = "cuda",
    dtype=torch.float16,
):
    """
    Convert an image into a 3D mesh and export it as a .glb file.
    """

    # Load pretrained models (assumes weights already downloaded)
    triposg_weights_dir = "pretrained_weights/TripoSG"
    rmbg_weights_dir = "pretrained_weights/RMBG-1.4"

    # Init models
    rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(device)
    rmbg_net.eval()
    pipe: TripoSGPipeline = TripoSGPipeline.from_pretrained(triposg_weights_dir).to(device, dtype)

    # Preprocess input
    img_pil = prepare_image(image_input, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)

    # Run pipeline
    outputs = pipe(
        image=img_pil,
        generator=torch.Generator(device=pipe.device).manual_seed(seed),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).samples[0]

    mesh = trimesh.Trimesh(outputs[0].astype(np.float32), np.ascontiguousarray(outputs[1]))

    if faces > 0:
        mesh = simplify_mesh(mesh, faces)

    # Export mesh
    mesh.export(output_path)
    print(f"✅ Mesh saved to {output_path}")

    return mesh