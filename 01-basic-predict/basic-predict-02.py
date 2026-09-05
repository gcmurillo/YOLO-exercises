from PIL import Image
from ultralytics import YOLO
from pathlib import Path

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolo26n.pt")


# Carpeta 'images' relativa a este script
image_dir = Path(__file__).parent / "images"

# Recolectar rutas de imágenes con extensiones comunes
images = sorted(
    str(p)
    for p in image_dir.iterdir()
    if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
)

if not images:
    print(f"No se encontraron imágenes en {image_dir}")
else:
    for image_path in images:
        print(f"Procesando {image_path}...")
        img = Image.open(image_path)  # PIL image
        results = model.predict(source=image_path, save=True)  # save plotted images

