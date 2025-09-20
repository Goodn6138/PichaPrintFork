from backend import scribble_func as scribbler
from backend import translate

x = translate.translate_to_english("ng'ombe andiliete")
print(x)


image_path = "/content/download.png"
prompt = x
output_path = "cow.obj"

# Call your function
scribbler.generate_glb_from_scribble(
    image_path=image_path,
    prompt=prompt,
    output_path=output_path,
    device="cuda"  # or "cpu" if no GPU
)
