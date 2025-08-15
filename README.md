# Fake vs Real Voice Detection

This project uses machine learning to distinguish between genuine human speech and AI-generated or deepfake voice recordings. By analyzing various audio features, the model classifies voice samples as either **real** or **fake**, addressing a critical need in an era of rising synthetic media.

***

## 📜 Table of Contents
* [About the Project](#-about-the-project)
* [Key Features](#-key-features)
* [Dataset](#-dataset)
* [Technologies Used](#-technologies-used)
* [Getting Started](#-getting-started)
* [Usage](#-usage)
* [Model Training](#-model-training)
* [Contributing](#-contributing)
* [License](#-license)

***

## 📖 About the Project

As AI-generated voice technology becomes more accessible, the potential for misuse in scams, misinformation, and fraudulent activities grows. This project tackles this challenge by building a robust classification system. It extracts key audio features from voice recordings to train a machine learning model capable of identifying the subtle artifacts that differentiate synthetic speech from authentic human speech. The primary model used is a **Random Forest Classifier**, known for its high accuracy and robustness in classification tasks.

***

## ✨ Key Features

* **Audio Feature Extraction:** Utilizes libraries like Librosa to extract Mel-Frequency Cepstral Coefficients (MFCCs), which are fundamental in audio processing.
* **Machine Learning Model:** Implements a Random Forest Classifier to accurately categorize voice samples.
* **Data Visualization:** Includes exploratory data analysis to visualize the distributions of audio features.
* **Model Evaluation:** Assesses the model's performance using metrics such as accuracy and a confusion matrix.

***

## 📊 Dataset

The model is trained on the "Fake and real voice detection" dataset available on **Kaggle**. This dataset contains a balanced collection of both authentic and AI-generated voice recordings, making it ideal for training a binary classification model. The audio files are processed to extract features that serve as input for the machine learning algorithm.

You can find the dataset used for this project [here](https://www.kaggle.com/datasets/joshuadaug/fake-and-real-voice-detection).

***

## 💻 Technologies Used

* **Python**
* **Scikit-learn:** For building the classification model.
* **Pandas:** For data manipulation and management.
* **NumPy:** For numerical operations.
* **Librosa:** For audio processing and feature extraction.
* **Matplotlib & Seaborn:** For data visualization.
* **Jupyter Notebook:** For project development and documentation.

***

## 🚀 Getting Started

To get a local copy up and running, follow these steps.

1.  **Clone the Repository**
    ```sh
    git clone [https://github.com/Laabh-Gupta/Fake_Vs_Real_Voice_Detection.git](https://github.com/Laabh-Gupta/Fake_Vs_Real_Voice_Detection.git)
    ```
2.  **Navigate to Project Directory**
    ```sh
    cd Fake_Vs_Real_Voice_Detection
    ```
3.  **Install Dependencies**
    It is recommended to create a virtual environment. Install the required libraries using:
    ```sh
    pip install scikit-learn pandas numpy librosa matplotlib seaborn jupyterlab
    ```

***

## Usage

The entire workflow, from data loading and feature extraction to model training and evaluation, is contained within the `Fake_Vs_Real_Voice_Detection.ipynb` Jupyter Notebook.

***

## 📈 Model Training

The project uses a **Random Forest Classifier** trained on the extracted audio features. The dataset is split into training and testing sets to ensure the model generalizes well to new, unseen data. The model's performance is evaluated based on its accuracy in classifying voices and is visualized using a confusion matrix to show the rates of true positives, true negatives, false positives, and false negatives.

***

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improving this project, please fork the repository and create a pull request. You can also open an issue with the tag "enhancement".

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

***

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.