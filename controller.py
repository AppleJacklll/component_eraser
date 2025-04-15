from flask import Blueprint, request, jsonify, send_file
import os
import eraser.image_processing as image_processing
import eraser.sam_service as sam_service

NO_FILENAME = "No filename part in the request"
NO_USERNAME = "No username part in the request"
NO_VALID_FILENAME = "No filename provided"
NO_VALID_USERNAME = "No username provided"
MIME_TYPE = "image/octet-stream"

eraser_bp = Blueprint('eraser_bp', __name__)


@eraser_bp.route('/erase/prepare', methods=['POST'])
def prepare():
    
    print("Prepare image to erase endpoint")
    data = request.get_json()
    
    if 'filename' not in data:
        return jsonify({"error": NO_FILENAME}), 400
    
    if 'username' not in data:
        return jsonify({"error": NO_USERNAME}), 400
    
    filename = data.get('filename')
    if not filename or filename == "":
        return jsonify({"error": NO_VALID_FILENAME}), 400
    
    username = data.get('username')
    if not username or username == "":
        return jsonify({"error": NO_VALID_USERNAME}), 400
    
    image, h, w = image_processing.prepare_image(filename, username)
    name, _ = os.path.splitext(filename)
        
    response = send_file(
            image,
            as_attachment=True,
            download_name=f"{name}_scaled.webp",
            mimetype=MIME_TYPE
        )
    response.headers['height'] = h
    response.headers['width'] = w
    return response


@eraser_bp.route('/erase/predict', methods=['POST'])
def predict():
    
    print("Predict to erase endpoint")
    data = request.get_json()
    
    if 'filename' not in data:
        return jsonify({"error": NO_FILENAME}), 400
    
    if 'username' not in data:
        return jsonify({"error": NO_USERNAME}), 400
    
    if 'point' not in data:
        return jsonify({"error": "No point part in the request"}), 400
    
    filename = data.get('filename')
    if not filename or filename == "":
        return jsonify({"error": NO_VALID_FILENAME}), 400
    
    username = data.get('username')
    if not username or username == "":
        return jsonify({"error": NO_VALID_USERNAME}), 400
    
    point = data.get('point')
    if not point or point == "" or 'x' not in point or 'y' not in point or 'label' not in point:
        return jsonify({"error": "No point provided or invalid point format"}), 400
    
    coordinates = (point['x'], point['y'])
    label = int(point['label'])
    
    image = image_processing.predict(filename, username, coordinates, label)
        
    return send_file(
        image,
        as_attachment=True,
        download_name=filename,
        mimetype=MIME_TYPE
    )


@eraser_bp.route('/erase/submit-mask', methods=['POST'])
def submit_mask():
    
    print("Submit mask to erase endpoint")
    
    if 'filename' not in request.form:
        return jsonify({"error": NO_FILENAME}), 400
    
    if 'username' not in request.form:
        return jsonify({"error": NO_USERNAME}), 400
    
    if 'submitMask' not in request.files:
        return jsonify({"error": "No submitMask part in the request"}), 400
    
    filename = request.form.get('filename')
    if not filename or filename == "":
        return jsonify({"error": NO_VALID_FILENAME}), 400
    
    username = request.form.get('username')
    if not username or username == "":
        return jsonify({"error": NO_VALID_USERNAME}), 400
    
    submit_mask = request.files.get('submitMask')
    if not submit_mask or submit_mask == "":
        return jsonify({"error": "No submitMask provided"}), 400
    
    image = image_processing.submit_mask(filename, username, submit_mask)
        
    return send_file(
        image,
        as_attachment=True,
        download_name=filename,
        mimetype=MIME_TYPE
    )


@eraser_bp.route('/erase/undo', methods=['POST'])
def undo():
    
    print("Undo erase endpoint")
    data = request.get_json()
    
    if 'filename' not in data:
        return jsonify({"error": NO_FILENAME}), 400
    
    if 'username' not in data:
        return jsonify({"error": NO_USERNAME}), 400
    
    filename = data.get('filename')
    if not filename or filename == "":
        return jsonify({"error": NO_VALID_FILENAME}), 400
    
    username = data.get('username')
    if not username or username == "":
        return jsonify({"error": NO_VALID_USERNAME}), 400
    
    image = image_processing.undo(filename, username)
        
    result = send_file(
        image,
        as_attachment=True,
        download_name=filename,
        mimetype=MIME_TYPE
    )
    
    if image.getbuffer().nbytes == 0:
        result.headers['undo'] = 'false'
    else:
        result.headers['undo'] = 'true'
    return result


@eraser_bp.route('/erase', methods=['POST'])
def erase():
    
    print("Erase endpoint")
    data = request.get_json()
    
    if 'filename' not in data:
        return jsonify({"error": NO_FILENAME}), 400
    
    if 'username' not in data:
        return jsonify({"error": NO_USERNAME}), 400
    
    filename = data.get('filename')
    if not filename or filename == "":
        return jsonify({"error": NO_VALID_FILENAME}), 400
    
    username = data.get('username')
    if not username or username == "":
        return jsonify({"error": NO_VALID_USERNAME}), 400
    
    image = image_processing.erase(filename, username)
        
    return send_file(
        image,
        as_attachment=True,
        download_name=filename,
        mimetype=MIME_TYPE
    )
    
    
@eraser_bp.route('/erase/reset', methods=['POST'])
def reset():
    
    print("Reset erase endpoint")
    data = request.get_json()
    
    if 'filename' not in data:
        return jsonify({"error": NO_FILENAME}), 400
    
    if 'username' not in data:
        return jsonify({"error": NO_USERNAME}), 400
    
    filename = data.get('filename')
    if not filename or filename == "":
        return jsonify({"error": NO_VALID_FILENAME}), 400
    
    username = data.get('username')
    if not username or username == "":
        return jsonify({"error": NO_VALID_USERNAME}), 400
    
    image_processing.reset(filename, username)
        
    return jsonify({"status": "success"}), 200
    

@eraser_bp.route('/erase/delete', methods=['POST'])
def delete():
    
    print("Delete for erase endpoint")
    ids_to_delete = sam_service.delete_sam_predictor()
    
    return jsonify(ids_to_delete), 200
    