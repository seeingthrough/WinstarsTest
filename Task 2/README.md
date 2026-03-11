# Task 2: CV & NLP Pipeline for animals classification

## This part combines two different fundamental approaches, computer vision, and natural language processing.
## The main goal is to identify animals in both images and text.

1. **Image Classification Model** - ResNet-18 network, trained on a image dataset (https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals) to classify categories of animals.

2. **NER Model** - a fine-tuned BERT trained on synthetic data to extract animals from text.

## Installation requirements:
1. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## How to open the EDA/Demo notebook:
1.
   ```bash
   jupyter notebook demo.ipynb
   ```

## How to install NER model:
1. Go to this website: https://huggingface.co/ukolchuga/WinstarsNER/tree/main, download .safetensors file and add it to Task 2/ner_classification/model folder

## Project Structure
```text
├── data/
│   ├── images/
│   │   ├── animals/                  # Folders with images per class
│   │   ├── animals.zip               # Archive with raw images
│   │   └── name of the animals.txt   # List of animal classes
│   └── data.json                     # Generated synthetic dataset for NER
├── image_classification/
│   ├── model/
│   │   └── im_cl_model.pth           # Saved PyTorch CV model weights
│   ├── inference_ic.py               # Script to predict animal from an image
│   └── train_ic.py                   # Script to train ResNet-18 on images
├── ner_classification/
│   ├── model/                        # Saved HuggingFace BERT model & tokenizer
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   ├── inference_ner.py              # Script to find animals in any text
│   └── train_ner.py                  # Script to fine-tune BERT for entity extraction
├── demo.ipynb                        # Jupyter Notebook with models testing
├── eda.ipynb                         # Jupyter Notebook with EDA
├── main_pipeline.py                  # Script comparing Image & Text model predictions
└── syntetic_data_generator.py        # Script to generate synthetic NER dataset
```

## Models Overview

### 1. ResNet-18 for Image Classification
A model with a custom-made fully connected layer for classification. The number of categories is 90.

The ResNet architecture utilizes "skip connections" to solve the vanishing gradient problem in deep networks. Models are labelled by their depth (e.g., ResNet-18, ResNet-34, ResNet-50).

We'll use the **ResNet-18** architecture with weights from the **ImageNet** dataset. ResNet-18 is trained on ImageNet and can classify images into 1000 object categories. The point is to re-use these weights via transfer learning.

We replace the final fully connected layer (`model.fc`) to match our specific number of classes, allowing the network to output predictions tailored to our animal dataset.

#### Training the model:
```bash
python image_classification/train_ic.py --data_dir data/images/animals/animals --epochs 5 --batch_size 32 --output_model image_classification/model/im_cl_model.pth
```

#### Inferencing the model:
```bash
python "Task 2/image_classification/inference_ic.py" --image_path "Task 2/data/test_images/bad_qual_pig.jpeg" --model_path "Task 2/image_classification/model/im_cl_model.pth"
```

**Output example:**
```text
------------------------------
Prediction: PIG
Confidence: 47.20%
------------------------------
```

### 2. BERT for Named Entity Recognition (NER)
The model identifies animals in text using BIO tokens ("O", "B-ANIMAL", "I-ANIMAL").
Synthetic data is generated using different templates via the `syntetic_data_generator.py` script.

We use `bert-base-uncased` from Hugging Face and add a token classification head. The tokenizer handles subwords, making the model robust to typos.

#### Generating Data & Training the model:
```bash
python syntetic_data_generator.py
python ner_classification/train_ner.py --data_path "Task 2/data/data.json" --epochs 3 --batch_size 16 --output_dir ner_classification/model
```

#### Inferencing the model:
```bash
python ner_classification/inference_ner.py --text "I was walking in the forest and suddenly a huge brown bear appeared." --model_path "Task 2/ner_classification/model"
```

**Output example:** ```text
Analyzing text: 'I was walking in the forest and suddenly a huge brown bear appeared.'
Found animal: bear


### 3. General Pipeline (Compare Image & Text Models)
*Note: Ensure both models are trained and saved in their respective `model/` directories before running the pipeline.*

**Input image:**
`test_image.jpg` (e.g., a picture of a bear)

**Input text:**
"Wow! That bear is so huge!!"

#### Running the pipeline:
```bash
python main_pipeline.py --image_path "Task 2/data/test_images/bad_qual_pig.jpg" --text_input "Wow! That pig is so huge!!"
```

**Output example:**
```text
==============================
NER found: pig
CV found:  pig
RESULT: True
==============================
```