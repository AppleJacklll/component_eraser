from werkzeug.datastructures import FileStorage
import pypdfium2 as pdfium
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import os
import cv2
import numpy as np
import eraser.sam_service as sam_service
from eraser.EraserState import EraserState

eraser_state = EraserState()

def prepare_image(filename: str, username: str) -> tuple:
    
    print("Prepare image to erase")
    name, _ = os.path.splitext(filename)
    key = name + "." + username
    file_path = os.path.join(sam_service.FILES_PATH, name)
    image = convert_to_rgb(file_path, filename)
    if not image:
        return None, None
    
    image_np = np.array(image)
    scaled_image, h, w = scale_image(image_np)
    combined_mask = np.zeros(scaled_image.shape[:2], dtype=np.uint8)
    
    sam_service.initialize_predictor(name, scaled_image)
    
    buffered = BytesIO()
    Image.fromarray(scaled_image).save(buffered, format="WebP", quality=85)
    buffered.seek(0)
    
    eraser_state.history[key] = []
    eraser_state.history[key].append((combined_mask, None, scaled_image))
    return buffered, h, w

def predict(filename: str, username: str, coordinates: tuple, label: int) -> BytesIO:
    
    print("Predict to erase")
    name, _ = os.path.splitext(filename)
    key = name + "." + username
    file_path = os.path.join(sam_service.FILES_PATH, name)
    scaled_image_path = os.path.join(file_path, f"{name}_scaled.webp")
    with Image.open(scaled_image_path) as scaled_image:
        image = np.array(scaled_image)
    
    mask, logit, _ = eraser_state.history[key][-1]
    predictor = sam_service.get_sam_predictor(name)
    
    if logit is not None:
        mask_input = logit[None, :, :]
    else:
        mask_input = None
    
    input_point = np.array([coordinates])
    input_label = np.array([1])
    
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        mask_input=mask_input,
    )
    
    best_logit = logits[np.argmax(scores), :, :]
    best_mask = masks[np.argmax(scores), :, :]
    
    best_mask = best_mask.astype(np.uint8) * 255
    if label == 1:
        best_mask = np.logical_or(best_mask, mask).astype(np.uint8) * 255
    else:
        best_mask = np.logical_and(np.logical_not(best_mask), mask).astype(np.uint8) * 255
    overlay_image = overlay_mask(image, best_mask)
    
    eraser_state.history[key].append((best_mask, best_logit, overlay_image))
    
    buffered = BytesIO()
    Image.fromarray(overlay_image).save(buffered, format="PNG")
    buffered.seek(0)
    return buffered


def submit_mask(filename: str, username: str, submit_mask: FileStorage) -> BytesIO:
    
    print("Submit mask to erase")
    name, _ = os.path.splitext(filename)
    key = name + "." + username
    file_path = os.path.join(sam_service.FILES_PATH, name)
    scaled_image_path = os.path.join(file_path, f"{name}_scaled.webp")
    with Image.open(scaled_image_path) as scaled_image:
        image = np.array(scaled_image)
    
    mask, _, _ = eraser_state.history[key][-1]
    
    with Image.open(submit_mask) as submit_mask_image:
        submit_mask = np.array(submit_mask_image)
    
    if submit_mask.shape[2] == 4:
        mask_binary = (submit_mask[:, :, 3] > 0).astype(np.uint8)
    else:
        mask_gray = cv2.cvtColor(submit_mask, cv2.COLOR_RGB2GRAY)
        mask_binary = (mask_gray > 0).astype(np.uint8)
        
    new_mask = np.logical_or(mask_binary, mask).astype(np.uint8) * 255
    overlay_image = overlay_mask(image, new_mask)
    
    eraser_state.history[key].append((new_mask, None, overlay_image))
    
    buffered = BytesIO()
    Image.fromarray(overlay_image).save(buffered, format="PNG")
    buffered.seek(0)
    return buffered
    

def undo(filename: str, username: str) -> BytesIO:
    
    print("Undo")
    name, _ = os.path.splitext(filename)
    key = name + "." + username
    if len(eraser_state.history[key]) <= 1:
        return BytesIO()
        
    eraser_state.history[key].pop()
    _, _, overlay_image = eraser_state.history[key][-1]
    
    buffered = BytesIO()
    Image.fromarray(overlay_image).save(buffered, format="PNG")
    buffered.seek(0)
    return buffered

def erase(filename: str, username: str) -> BytesIO:
    
    print("Erase")
    name, _ = os.path.splitext(filename)
    key = name + "." + username
    file_path = os.path.join(sam_service.FILES_PATH, name)
    image_np = np.array(convert_to_rgb(file_path, filename))
    
    mask, _, _ = eraser_state.history[key][-1]
    
    original_size = image_np.shape[:2][::-1]
    resized_mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
    
    image_np[resized_mask > 0] = [255, 255, 255]
    
    buffered = BytesIO()
    Image.fromarray(image_np).save(buffered, format="PNG")
    buffered.seek(0)
    return buffered

def reset(filename: str, username: str):
    
    print("Reset")
    name, _ = os.path.splitext(filename)
    key = name + "." + username
    eraser_state.history[key][:] = eraser_state.history[key][:1]

def convert_to_rgb(file_path: str, filename: str) -> Image:
    image_path = os.path.join(file_path, filename)
    
    if filename.endswith('.pdf'):
        try:
            pages = pdfium.PdfDocument(image_path)
            if not pages or len(pages) == 0:
                print("No pages found in the provided PDF file.")
                return None
            return pages[0].render(scale=4).to_pil().convert("RGB")
        except Exception as e:
            print("Error converting PDF to image", e)
            return None
    else:
        try:
            with Image.open(image_path) as image:
                return image.convert("RGB")
        except UnidentifiedImageError:
            return None

def scale_image(image: np.ndarray, max_size: int = 1080) -> cv2.typing.MatLike:
    h, w = image.shape[:2]
    scale_factor = min(max_size / max(h, w), 1.0)
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    new_size = (new_w, new_h)
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA), new_h, new_w

def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha_masked=0.5, alpha_unmasked=0.5) -> np.ndarray:
    rgba_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2RGBA)
    rgba_image = rgba_image.astype(np.float32)
    blue_color = np.array([0, 0, 255, 255], dtype=np.float32)
    grey_color = np.array([128, 128, 128, 255], dtype=np.float32)

    masked_area = mask > 0
    unmasked_area = ~masked_area

    overlay = rgba_image.copy()
    overlay[masked_area] = (1 - alpha_masked) * rgba_image[masked_area] + alpha_masked * blue_color
    overlay[unmasked_area] = (1 - alpha_unmasked) * rgba_image[unmasked_area] + alpha_unmasked * grey_color

    return np.clip(overlay, 0, 255).astype(np.uint8)