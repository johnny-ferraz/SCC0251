# Aluno: João Lucas Almeida Caldas Ferraz
# Número USP: 12609651
# Disciplina: Processamento de Imagens (SCC0251) - 2025/1
# Projeto Final - Filtragem de Ruído em Imagens de Mamografia Digital

import pydicom
from pydicom.dataset import FileDataset
import datetime
import numpy as np
import os

def saveROIs(img, filepath, coordinates):
    # Extrai uma ROI 144x144 centrada nas coordenadas (x, y)
    x, y = coordinates
    roi = img[x-72:x+72, y-72:y+72]

    # Converte a ROI para uint16 (formato esperado pelo DICOM)
    roi_uint16 = roi.astype(np.uint16)

    # Cria os diretórios se não existirem
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Cria estrutura básica de um arquivo DICOM
    file_meta = pydicom.dataset.FileMetaDataset()
    ds = FileDataset(filepath, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # Define data e hora de criação do arquivo
    dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime('%Y%m%d')
    ds.ContentTime = dt.strftime('%H%M%S')

    # Define atributos obrigatórios da imagem DICOM
    ds.Rows, ds.Columns = roi.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = roi_uint16.tobytes() # Dados da imagem em bytes

    # Salva o arquivo DICOM no caminho indicado
    ds.save_as(filepath)