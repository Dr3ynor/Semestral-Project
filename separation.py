import os
import numpy as np
import cv2
from pdf2image import convert_from_path
import pymupdf

pdf_path = "/home/jakub/school/semestralni_prace/voynich.pdf"
destination_folder = "voynich_pages"
os.makedirs(destination_folder,exist_ok=True)

doc = pymupdf.open(pdf_path)
for page_num in range(len(doc)):
    images = convert_from_path(pdf_path, first_page=page_num + 1, last_page=page_num + 1)
    for img in images:
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        output_path = os.path.join(destination_folder,f"page_{page_num+1}.png")
        cv2.imwrite(output_path,img_cv)
        