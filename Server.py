from flask import Flask, request, jsonify
from ImageProcess import preprocess_image

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello, World!'})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    data = request.json
    base64_image = data.get('image')

    if base64_image:
        # Process the base64 image and get the processed image path
        processed_image_binary, processed_image_base64 = preprocess_image(base64_image)

        # Respond with the processed image binary data
        return jsonify({'success': 'true', 'image': processed_image_base64})
    else:
        return jsonify({'error': 'No base64 image received'})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)