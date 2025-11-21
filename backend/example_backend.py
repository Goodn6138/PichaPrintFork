import os
from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
from pyngrok import ngrok
from flask_cors import CORS  # allow frontend (Render) to call backend

# Your existing imports
from backend import scribble_func as scribbler
from backend import translate
from huggingface_hub import snapshot_download
from triposg.pipelines.pipeline_triposg import TripoSGPipeline
from scripts.briarmbg import BriaRMBG
from PIL import Image
import torch
from scripts.inference_triposg import *


# 1. Load background removal model
rmbg_net = BriaRMBG.from_pretrained("briaai/RMBG-1.4")
rmbg_net.to("cuda")

# 2. Load TripoSG pipeline
triposg_weights = 'pretrained_weights/TripoSG'
rmbg_weights_dir = 'pretrained_weights/RMBG-1.4'
snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=triposg_weights)
snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_weights_dir)

pipe = TripoSGPipeline.from_pretrained(triposg_weights).to("cuda", torch.float16)



UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# ---- SET NGROK AUTH TOKEN ----
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")
if NGROK_AUTH_TOKEN:
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
else:
    print("⚠️ No NGROK_AUTH_TOKEN found! Please set it in your environment.")

# ---- TEST HOME ROUTE ----
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "Backend server is running! Use POST /upload to generate STL."
    })

# ---- UPLOAD ROUTE ----
@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files or "description" not in request.form:
        return jsonify({"error": "Image and description required"}), 400

    text_input = request.form["description"]
    prompt = translate.translate_to_english(text_input)

    file = request.files["image"]
    filename = secure_filename(file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)

    output_file = os.path.join(OUTPUT_FOLDER, filename.split(".")[0] + ".stl")

    scribbler.generate_glb_from_scribble(
        image_path=image_path,
        prompt=prompt,
        output_path=output_file,
        device="cuda"
    )

    return send_file(output_file, as_attachment=True)

@app.route("/image_upload" , methods= ["POST"])
def image_route():
  if "image_upload" not in request.files:
    return jsonify({"error": "Image required " }), 400
  file = request.files["image_upload"]
  filename = secure_filename(file.filename)
  image_path = os.path.join(UPLOAD_FOLDER, filename)
  file.save(image_path)
  output_file = os.path.join(OUTPUT_FOLDER, filename.split(".")[0] + "_image.stl")
  mesh = run_triposg(
      pipe = pipe,
      image_input = image_path,
      rmbg_net = rmbg_net,
      seed = 123,
      num_inference_steps=50,
      guidance_scale=7.0,
      faces=20000     
  )

  mesh.export(output_file)
  return send_file(output_file, as_attachment=True)

if __name__ == "__main__":
    port = 5000
    public_url = ngrok.connect(port)
    print("✅ ngrok tunnel URL:", public_url)
    app.run(port=port)
