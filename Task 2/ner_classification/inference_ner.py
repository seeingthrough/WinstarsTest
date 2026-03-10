import argparse
from transformers import pipeline
import os

def parse_args():
    """
    Parses command-line arguments. Handles dynamic inputs like
    the text string to analyze and the path to the trained NER model.
    """
    parser = argparse.ArgumentParser(description="Inference for Animal NER")
    parser.add_argument("--model_path", type=str, default="./ner_model", help="Path to trained model directory")
    parser.add_argument("--text", type=str, required=True, help="Text to analyze")
    return parser.parse_args()


def extract_animal(model_path, text):
    """
    Loads a fine-tuned Hugging Face transformer model for Named Entity Recognition.
    Processes the input text and extracts entities classified under the 'ANIMAL' group.
    """
    abs_model_path = os.path.abspath(model_path)

    if not os.path.exists(abs_model_path):
        raise OSError(f"Directory at {abs_model_path} not found!")

    ner_pipeline = pipeline(
        "ner",
        model=abs_model_path,
        tokenizer=abs_model_path,
        aggregation_strategy="simple"
    )

    results = ner_pipeline(text)
    animals = [res['word'] for res in results if res['entity_group'] == 'ANIMAL']

    return animals


def main():
    """
    Main execution block. Receives parsed arguments, executes the extraction pipeline,
    and prints the first detected animal, handling cases where no entities are found.
    """
    args = parse_args()

    print(f"Analyzing text: '{args.text}'")
    found_animals = extract_animal(args.model_path, args.text)
    if found_animals:
        print(f"Found animal: {found_animals[0]}")
        return found_animals[0]
    else:
        print("No animals found in text.")
        return None

if __name__ == "__main__":
    main()