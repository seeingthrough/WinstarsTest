import argparse
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from transformers import pipeline
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal Animal Checker Pipeline")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--ner_model", type=str, default="Task 2/ner_classification/model")
    parser.add_argument("--clf_model", type=str, default="Task 2/image_classification/model/im_cl_model.pth")
    return parser.parse_args()


def get_ner_animal(text, model_path):
    """Extracts animal from provided text with specified model"""
    ner_pipe = pipeline("ner", model=model_path, aggregation_strategy="simple")
    results = ner_pipe(text)
    for entity in results:
        if "ANIMAL" in entity['entity_group'].upper():
            return entity['word'].lower().strip()
    return None


def get_image_animal(image_path, model_path):
    """Classifies animal by its image with specified model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint['class_names']
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)

    return class_names[preds[0]].lower().strip()


def main():
    args = parse_args()
    print(f"--- Processing Text ---")
    animal_from_text = get_ner_animal(args.text, args.ner_model)
    print(f"--- Processing Image ---")
    animal_from_img = get_image_animal(args.image, args.clf_model)
    print("\n" + "=" * 30)
    print(f"NER found: {animal_from_text}")
    print(f"CV found:  {animal_from_img}")
    match = (animal_from_text == animal_from_img) and (animal_from_text is not None)
    print(f"RESULT: {match}")
    print("=" * 30)

    return match


if __name__ == "__main__":
    main()