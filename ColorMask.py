import cv2
import numpy as np
import glob
import os

# Mappa classe → colore BGR
COLOR_MAP = {
    0: (0, 0, 0),  # sfondo nero
    1: (0, 255,   0),    # bugne = verde
    2: (0,   0, 255),    # elementi arco = rosso
    3: (255,   0,   0),  # elementi cornice = blu
}


# Cartelle
INPUT_DIR  = "/Users/mattiacastiello/Desktop/tesi/code/PreparazioneDB/Filejson/masks"      # qui ci metti tutte le maschere
OUTPUT_DIR = "color_masks/"    # qui verranno salvate le immagini colorate
os.makedirs(OUTPUT_DIR, exist_ok=True)

for path in glob.glob(os.path.join(INPUT_DIR, "*.png")):
    # 1. Carico la maschera in scala di grigi (i valori 0,1,2,3)
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        continue

    # Se la maschera è a triplo canale, la converto in singolo
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    # 2. Coloro ogni classe con il colore fisso
    for cls, col in COLOR_MAP.items():
        colored[mask == cls] = col

    # (opzionale) 3. Se vuoi un outline bianco su ogni classe, smussa qui:
    # contours, _ = cv2.findContours((mask>0).astype(np.uint8), 
    #                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(colored, contours, -1, (255,255,255), thickness=2)

    # 4. Salvo
    out_name = os.path.basename(path).replace(".png", "_colored.png")
    cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), colored)
    print(f"Salvata: {out_name}")