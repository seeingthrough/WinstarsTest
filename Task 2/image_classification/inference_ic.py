import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os


def parse_args():
    """
    Parses CLI arguments for the inference script. Handles the dynamic input
    of the target image path and the trained model checkpoint.
    """
    parser = argparse.ArgumentParser(description="Inference for Animal Classification")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image you want to classify")
    parser.add_argument("--model_path", type=str, default="animal_model.pth", help="Path to saved .pth model")
    return parser.parse_args()


def predict(image_path, model_path):
    """
    Loads the trained ResNet18 model and applies the required image transformations.
    Performs a forward pass without gradient calculation to output the predicted
    class and its confidence score via Softmax.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint['class_names']

    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

    img_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, preds = torch.max(probabilities, 0)

    result_class = class_names[preds.item()]
    return result_class, confidence.item()


def main():
    """
    Entry point of the script. Validates the existence of the input image,
    executes the prediction, and formats the output for readability.
    """
    args = parse_args()

    if not os.path.exists(args.image_path):
        print(f"File not found: {args.image_path}")
        return

    result = predict(args.image_path, args.model_path)

    if result:
        label, score = result
        print("-" * 30)
        print(f"Prediction: {label.upper()}")
        print(f"Confidence: {score:.2%}")
        print("-" * 30)


if __name__ == "__main__":
    main()