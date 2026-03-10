# Task 1: MNIST Classification with TensorFlow and Scikit-learn

In this task, we will build three models to classify handwritten digits (0-9). We are using the classic MNIST dataset, which consists of 60,000 training images and 10,000 test images.

## Architecture

The code is designed with a focus on clean architecture:
* **Interface:** Each model is implemented as a separate class inheriting from the `MnistClassifierInterface`, which enforces two core methods: **`train`** and **`predict`**.
* **Design Patterns:** Model selection and initialization are encapsulated within the `MnistClassifier` wrapper class. Under the hood, it leverages a combination of **Abstract Factory** and **Strategy** design patterns, making the codebase flexible and easy to scale.

## Algorithms Implemented

We will build and compare three different approaches:
1. **Neural Network (Fully Connected)**
2. **Convolutional Neural Network (CNN)**
3. **Random Forest**

---

## How to Run the Code

**1. Create and activate a virtual environment** *(recommended to isolate dependencies)*:
```bash
python -m venv venv
```
# For Windows:
```bash
venv\Scripts\activate
```
# For macOS/Linux:
```bash
source venv/bin/activate
```
2. Install the required packages:

Bash
```bash
pip install -r requirements.txt
```
3. Open the demo notebook:
Launch Jupyter Notebook to interact with the demo.ipynb file:

Bash 
```bash
jupyter notebook demo.ipynb
```
4. Run all cells in jupyter notebook, models will automaticaly be trained and tested, relevant metrics will be displayed.