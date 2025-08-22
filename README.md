# Neural Network
<p align="left">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&duration=3000&pause=500&color=00FF80&center=false&vCenter=false&width=800&lines=banknotes.csv+-+Dataset+for+Fraud+Detection;edgedetection_output.png+-+Image+for+Edge+Detection;MNIST_cnn_model+-+Pretrained+CNN+Accuracy+98.87%" alt="Typing SVG" />
</p>



> This repository is a self-learning project on neural networks, covering both the foundations and practical implementations.  
   It combines theoretical understanding with applied experiments in fraud detection, computer vision, and sequence modeling.


 ## Overview

> The notebooks inside walk through key concepts in machine learning and deep learning, including:

  - Neural networks and their structure  
  - Activation functions and gradient descent  
  - Multilayer architectures and backpropagation  
  - Overfitting and model evaluation  
  - Convolutional Neural Networks (CNNs) for image tasks  
  - Recurrent Neural Networks (RNNs) for sequence tasks  

## Basics

 ###  Biological Inspiration
> - Human brain neurons fire when electrical signals cross a threshold.
> - Artificial Neural Networks (ANNs) mimic this behavior.
>- ANN = nodes (neurons) + weighted edges + biases + outputs.

---

 ###  Core Components
>- **Weights (W):** Control feature importance.
> - **Bias (b):** Adjusts activation threshold, allows flexibility.
>- **Hypothesis function:** `ŷ = f(Wx + b)`  
 > - Model's "guess" for outputs.
>- **Loss function:** Measures difference between prediction (`ŷ`) and true target (`y`).

---

 ###  Gradient Descent
>- Optimizes weights and bias by minimizing loss.
>- Update rule:  
  `W := W - α * ∂Loss/∂W`  
  `b := b - α * ∂Loss/∂b`
  
#### **Types:**
  >- **Batch GD:** Uses entire dataset.
  >- **Stochastic GD (SGD):** Uses 1 sample at a time.
  >- **Mini-batch GD:** Uses small random batches (most common).
>- **Backpropagation:** Algorithm to calculate gradients efficiently across layers.

---

 ### Activation Functions
>- **Step function:** Outputs 0 or 1 (rarely used today).
>- **Sigmoid:** Smooth curve in range (0, 1), used for probabilities.
>- **ReLU:** `0 if x < 0, x if x ≥ 0`, very common in modern networks.

---

 ### Deep Neural Networks (DNNs)
>- Multiple hidden layers stacked together.
>- Each layer transforms inputs into higher-level representations.
>- Final layer outputs probabilities for classification.

---

 ### Regularization: Dropout
>- Randomly disables some neurons during training.
>- Prevents overfitting and improves generalization.

---

 ## Applications

  >**Fraud Detection** using `banknotes.csv`  
   >- Data loading, train-test split, model building, training, and evaluation  
  
  >**Computer Vision** with edge detection  
  >- Input image: `edgedetection_input.input`  
   >- Processing, visualization, and saving output
### Edge Detection output image:
> `edgedetection_output.png`
  
  >- **Handwriting Recognition** with CNNs on MNIST  
   >- Pre-trained model: `MNIST_cnn_model` (accuracy: 0.9887)  
   >- Retrainable with custom naming (do not append `.keras` extension)
### Trained MNIST output Model
>`MNIST_cnn_model.keras`  
---
## Files

> required files are in this repository
>- `requirement.txt`
>- `banknotes.csv`
>- `edgedetection_input.input`
>- python file `neural_network.py`
>- `NERVMAP.txt`



