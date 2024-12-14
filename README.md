<div id="header" align="center"> <img src="https://media.giphy.com/media/M9gbBd9nbDrOTu1Mqx/giphy.gif" width="100"/> </div> <div id="badges" align="center"> <a href="your-linkedin-URL"> <img src="https://img.shields.io/badge/LinkedIn-blue?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn Badge"/> </a> <a href="your-youtube-URL"> <img src="https://img.shields.io/badge/YouTube-red?style=for-the-badge&logo=youtube&logoColor=white" alt="Youtube Badge"/> </a> <a href="your-twitter-URL"> <img src="https://img.shields.io/badge/Twitter-blue?style=for-the-badge&logo=twitter&logoColor=white" alt="Twitter Badge"/> </a> </div> <p align="center"> <img src="https://komarev.com/ghpvc/?username=SuyashUtekar&style=flat-square&color=blue" alt=""/> </p> <h1 align="center"> hey there <img src="https://media.giphy.com/media/hvRJCLFzcasrR4ia7z/giphy.gif" width="30px"/> </h1>

# PE_Malicious_File_Detection_Using_Deep_Learning
---
### :man_technologist: Overview :
This repository showcases a deep learning-based solution for detecting malicious Portable Executable (PE) files. The project leverages autoencoders for feature engineering and employs a neural network for accurate and reliable predictions to enhance cybersecurity.

### :telescope: Workflow Summary:

Feature Engineering: An autoencoder reduces the dimensionality of features, retaining the most important information in a bottleneck representation.
Modeling: A fully connected neural network (ANN) is trained on the extracted bottleneck features to classify files as malicious or legitimate.

### :seedling: Deep Learning Models and Techniques:

### :gear: Components:

1) **Autoencoder**:
- Encoder: Reduces the input dimensionality to a compact representation (bottleneck layer).
- Decoder: Reconstructs input features during training to minimize reconstruction loss.

2) **Artificial Neural Network (ANN)**:
- A dense neural network trained on bottleneck features for classification.


## :hammer_and_wrench: Tools and Frameworks:
<div> <img src="https://github.com/devicons/devicon/blob/master/icons/python/python-original-wordmark.svg" title="Python" alt="Python" width="40" height="40"/>&nbsp; <img src="https://github.com/devicons/devicon/blob/master/icons/tensorflow/tensorflow-original-wordmark.svg" title="TensorFlow" alt="TensorFlow" width="40" height="40"/>&nbsp; <img src="https://github.com/devicons/devicon/blob/master/icons/keras/keras-original.svg" title="Keras" alt="Keras" width="40" height="40"/>&nbsp; <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" title="Google Colab" alt="Google Colab" width="40" height="40"/>&nbsp; <img src="https://github.com/devicons/devicon/blob/master/icons/numpy/numpy-original.svg" title="NumPy" alt="NumPy" width="40" height="40"/>&nbsp; <img src="https://github.com/devicons/devicon/blob/master/icons/scikitlearn/scikitlearn-original.svg" title="Scikit-learn" alt="Scikit-learn" width="40" height="40"/>&nbsp; </div>

## 🚀 Implementation Steps:
1. **Preprocessing and Feature Extraction**
- Load and preprocess data.
- Extract features and prepare them for model input.
- Save preprocessed features for reuse.

2. **Train an Autoencoder**
- Build an autoencoder model with:
- Input Layer: Matches the dimension of input features.
- Encoder Layers: Compress the input features using fully connected layers.
- Bottleneck Layer: Capture the most significant features (latent representation).
- Decoder Layers: Reconstruct the original features to calculate reconstruction loss.
- Train the autoencoder to minimize reconstruction loss.
- Save the encoder portion of the model for feature extraction.
  
3. **Extract Bottleneck Features**
- Use the trained encoder to transform input data into bottleneck features.
  
4. **Train an Artificial Neural Network**
- Train an ANN model on bottleneck features with:
- Dense hidden layers.
- Dropout and batch normalization for regularization.
- Output layer with sigmoid activation for binary classification.
- Evaluate the model on validation and test datasets.

5. **Test the Final Model**
- Use the trained ANN to classify files as malicious or legitimate.
- Save the trained ANN and encoder for future predictions.

📊 Results:
1) **Autoencoder Bottleneck Features**
- The autoencoder successfully reduced the feature dimensions while preserving essential information.
- Bottleneck size: 16 dimensions.
  
2) **Classification Model**
- *Top-Performing Model*: The ANN model trained on bottleneck features achieved the highest accuracy.
- *Metrics*:
-- Accuracy: 0.9965
-- Precision: 0.9951
-- Recall: 0.9938
-- F1-Score: 0.9945

- *Cross-Validation*: Mean cross-validation score of 0.9872, showcasing generalization.
  
## 📈 Visualizations:
Model Loss
<img src="https://github.com/user-attachments/assets/autoencoder-loss-plot" width="600" height="400"/>
Accuracy Comparison
<img src="https://github.com/user-attachments/assets/accuracy-comparison" width="600" height="400"/>

## 📝 Conclusion
This project demonstrates a successful implementation of deep learning techniques for malicious file detection:

1. **Feature Engineering**:
The autoencoder effectively reduced the dimensionality, enabling better performance of the classification model.
2. **High Performance**:
The ANN trained on bottleneck features delivered state-of-the-art performance with high accuracy and robust generalization.
3. **Implementation**:
A streamlined process was implemented, ensuring reproducibility and real-world applicability for detecting malicious files.

## 🚀 Future Work
1. **Improved Architectures**:
Experiment with advanced architectures like convolutional autoencoders or transformers.
2. **Scalability**:
Extend the model for real-time file detection in production environments.
3. **Ensemble Learning**:
Combine results from multiple deep learning models for enhanced predictions.

## 🙏 Acknowledgments
Special thanks to open-source libraries like TensorFlow, Keras, and scikit-learn for facilitating this implementation.

