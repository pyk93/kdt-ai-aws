from flask import Flask, request, json
from model import MLModelHandler, DLModelHandler

app = Flask(__name__)

# assign model handler as global variable [2 LINES]
ml_handler = MLModelHandler()
dl_handler = DLModelHandler()


@app.route("/predict", methods=["POST"])
def predict():
    # handle request and body
    body = request.get_json()
    text = body.get('text', '')
    text = [text] if isinstance(text, str) else text
    model_type = body.get('model_type', 'ml')

    # model inference [2 LINES]
    if model_type == 'ml':
        predictions = ml_handler.handle(text)
    else:
        predictions = dl_handler.handle(text)

    # response
    result = json.dumps({str(i): {'text': t, 'label': l, 'confidence': c}
                         for i, (t, l, c) in enumerate(zip(text, predictions[0], predictions[1]))})
    return result

import io
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import jsonify
imagenet_class_index = json.load(open('data/imagenet_class_index.json'))
model_torchflasktutorial = models.densenet121(weights='IMAGENET1K_V1')
model_torchflasktutorial.eval()


def transform_image_torchflasktutorial(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction_torchflasktutorial(image_bytes):
    tensor = transform_image_torchflasktutorial(image_bytes=image_bytes)
    outputs = model_torchflasktutorial.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


@app.route('/predict_torch_tutorial', methods=['POST'])
def predict_torchtutorial():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction_torchflasktutorial(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})


if __name__ == '__main__':
    app.run()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port='5000', debug=True)
