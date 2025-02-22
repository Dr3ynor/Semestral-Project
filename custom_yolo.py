# yolo task=detect mode=train model=yolov5s.pt data=data.yaml epochs=20 imgsz=640
# yolo task=detect mode=predict model= data=data.yaml imgsz=640

"""rf = Roboflow(api_key="i1wEL7gP5ZGA42vbNNEV")
project = rf.workspace("voynich").project("voynich")
version = project.version(1)
dataset = version.download("yolov11")
  """
# /home/jakub/school/SemestralProject/Voynich-2/runs/detect/train/weights


from roboflow import Roboflow
rf = Roboflow(api_key="i1wEL7gP5ZGA42vbNNEV")
project = rf.workspace("voynich").project("voynich")
version = project.version(2)
dataset = version.download("yolov11")
                
"""

from ultralytics import YOLO
import os

# Cesty k modelu, vstupním a výstupním složkám
model_path = "Voynich-1/runs/detect/train4/weights/best.pt"  # Cesta k modelu
input_folder = "voynich_pages"       # Složka s obrázky
output_folder = "custom_model_predictions"  # Složka, kam uložit výsledky

# Načtení modelu
model = YOLO(model_path)

# Ujisti se, že výstupní složka existuje
os.makedirs(output_folder, exist_ok=True)

# Predikce na všech obrázcích ve složce
for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".jpeg", ".png")):  # Filtruj obrázky
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Proveď predikci
        results = model(input_path)

        # Ulož výsledek
        results[0].save(filename=output_path) 

print("Predikce dokončena! Výsledky jsou ve složce:", output_folder)
"""
