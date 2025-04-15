import torch
import os
from datetime import datetime, timedelta
import numpy as np
from eraser.tinysam import sam_model_registry, SamPredictor
from eraser.EraserState import EraserState

sam = None
predictors = {}
eraser_state = EraserState()
FILES_PATH = r'internal/eraser/files'
SAM_CHECKPOINT = r'models/tinysam_42.3.pth'
MODEL_TYPE = "vit_t"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"# "xpu" if torch.xpu.is_available() else "cpu"

def initialize_sam():
    global sam
    if sam is None:
        sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
        sam.to(device=DEVICE)
        
def initialize_predictor(id: str, image: np.ndarray):
    global sam, predictors
    if sam is None:
        raise RuntimeError("SAM model not initialized")
    if id not in predictors:  
        predictor = SamPredictor(sam)
        predictors[id] = predictor
        predictor.set_image(image)

def get_sam_predictor(id: str) -> SamPredictor:
    global predictors
    if id not in predictors:
        raise RuntimeError("SAM predictor not initialized for id: {}".format(id))
    return predictors[id]

def delete_sam_predictor() -> list:
    global predictors
    ids_to_delete = []
    for id in predictors:
        file_path = os.path.join(FILES_PATH, id)
        used_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        if used_time < datetime.now() - timedelta(days=1):
            del predictors[id]
            ids_to_delete.append(id)
            
            for key in eraser_state.history.keys():
                if id in key:
                    del eraser_state.history[key]
                    break