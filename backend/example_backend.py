#THIS IS THE BARE MINIMUM FOR THE BACKEND REQUIRED.
import os
from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
from pyngrok import ngrok
from flask_cors import CORS  # allow frontend (Render) to call backend

# Your existing imports
from backend import scribble_func as scribbler
from backend import translate

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

if __name__ == "__main__":
    port = 5000
    public_url = ngrok.connect(port)
    print("✅ ngrok tunnel URL:", public_url)
    app.run(port=port)
