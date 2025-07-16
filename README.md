# Brain Tumor Detection and Classification using Transfer Learning

This project focuses on developing an automated system for classifying brain tumors from MRI images using deep learning techniques, specifically leveraging transfer learning with the EfficientNet B0 model.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Problem Statement](#problem-statement)
* [Objectives](#objectives)
* [Scope](#scope)
* [Methodology](#methodology)
* [Dataset](#dataset)
* [Model Architecture](#model-architecture)
* [Technologies Used](#technologies-used)
* [Results](#results)
* [How to Run](#how-to-run)
* [Future Enhancements](#future-enhancements)

---

## Project Overview

Brain tumors are life-threatening conditions requiring early and accurate diagnosis. Manual diagnosis from MRI scans is time-consuming and prone to human error. This project addresses this challenge by building a robust classification model to distinguish between four brain tumor classes: glioma tumor, meningioma tumor, pituitary tumor, and no tumor. The solution utilizes the EfficientNet B0 architecture, known for its balance between accuracy and computational efficiency.

---

## Problem Statement

The need for an automated system to classify brain tumors from MRI scans to overcome the limitations of manual diagnosis, which is time-consuming and susceptible to human error.

---

## Objectives

* To build a classification model capable of distinguishing between four brain tumor classes: glioma, meningioma, pituitary tumor, and no tumor.
* To apply deep learning using EfficientNet B0 for high accuracy and low computational cost.
* To evaluate model performance using standard classification metrics.

---

## Scope

This project focuses on image-based classification of brain tumors from MRI images using deep learning. It is limited to the four specified tumor classes and does not extend to tumor segmentation or 3D volume analysis. The dataset used is sourced from Kaggle.

---

## Methodology

The project follows a standard machine learning pipeline:

1.  **Data Collection and Preprocessing:** MRI images are loaded, resized to $224 \times 224$ pixels, and normalized.
2.  **Data Augmentation:** Applied using Keras' `ImageDataGenerator` to address class imbalance and improve model generalization.
3.  **Model Training with Transfer Learning:** EfficientNet B0, pre-trained on ImageNet, is used as the base model, with custom top layers fine-tuned for the specific classification task.
4.  **Model Evaluation and Visualization:** Performance is assessed using accuracy, loss plots, and a confusion matrix.

---

## Dataset

The project utilizes the **Kaggle Brain Tumor Classification MRI Dataset**, which includes pre-classified MRI images categorized into four folders:
* `glioma_tumor`
* `no_tumor`
* `meningioma_tumor`
* `pituitary_tumor`

---

## Model Architecture

The model is built on the **EfficientNet B0** architecture, employing transfer learning:

* **Base Model:** `EfficientNetB0` with `imagenet` weights, `include_top=False`.
* **Custom Layers:**
    * `GlobalAveragePooling2D`
    * `Dense` layer with 1024 units and `relu` activation.
    * `Dropout` layer (40% dropout rate).
    * Final `Dense` layer with 4 units (one per class) and `softmax` activation.

The model is compiled with the `Adam` optimizer and `categorical_crossentropy` loss. Callbacks like `TensorBoard`, `ModelCheckpoint`, and `ReduceLROnPlateau` are used during training.

---

## Technologies Used

* **Programming Language:** Python
* **Libraries:**
    * TensorFlow
    * Keras
    * NumPy
    * Pandas
    * OpenCV (`cv2`)
    * Matplotlib
    * Seaborn
    * Scikit-learn
    * Tqdm
* **Platform:** Google Colab

---

## Results

The model achieved high performance metrics:

* **Training Accuracy:** 99%
* **Validation Accuracy:** 94.6%
* **Test Accuracy:** 98%

EfficientNet B0 demonstrated superior accuracy and speed compared to older models, proving lightweight and suitable for deployment on low-resource systems.

---

## How to Run

1.  **Clone the Repository:** If this project is in a Git repository, clone it to your local machine.
2.  **Download Dataset:** Obtain the "Brain Tumor Classification (MRI)" dataset from Kaggle and place it in the appropriate directory (as referenced in the notebook, e.g., `../input/brain-tumor-classification-mri/`).
3.  **Install Dependencies:** Ensure you have all the required Python libraries installed. You can typically install them using pip:
    ```bash
    pip install tensorflow keras numpy pandas opencv-python matplotlib seaborn scikit-learn tqdm
    ```
4.  **Open Jupyter Notebook:** Launch Jupyter (or Google Colab) and open the `cnn-brain-tumor.ipynb` file.
5.  **Run Cells:** Execute the cells sequentially within the notebook to perform data loading, preprocessing, model training, and evaluation.

---

## Future Enhancements

* **Tumor Segmentation:** Implement segmentation models for precise tumor localization within MRI images.
* **3D MRI Volume Classification:** Extend the model to classify tumors from 3D MRI volumes for more comprehensive analysis.
* **Web/Mobile Application Deployment:** Deploy the model as a web or mobile application to enable real-time diagnosis.
