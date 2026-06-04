# Next Word Prediction using LSTM

A deep learning project that trains an LSTM-based language model to predict the next word in a sequence. The model is trained on a custom text corpus containing information about the CampusX Data Science Mentorship Program and learns word relationships to generate text one word at a time.

## Overview

This project demonstrates how Recurrent Neural Networks (RNNs), specifically Long Short-Term Memory (LSTM) networks, can be used for language modeling and next-word prediction.

Given an input sequence of words, the model predicts the most likely next word based on patterns learned during training.

### Example

Input:

```text
The course follows a monthly
```

Output:

```text
The course follows a monthly subscription
```

---

## Features

* Text preprocessing and tokenization using NLTK
* Vocabulary creation from a custom corpus
* Sequence generation for language modeling
* Padding for variable-length sequences
* LSTM-based neural network architecture
* GPU support using CUDA
* Next-word prediction and text generation

---

## Tech Stack

* Python
* PyTorch
* NLTK
* NumPy

---

## Project Workflow

### 1. Text Preprocessing

* Convert text to lowercase
* Tokenize text using NLTK
* Build vocabulary mappings
* Convert words into numerical indices

### 2. Training Data Creation

For each sentence:

```text
The course follows a monthly subscription
```

Training sequences become:

```text
The course
The course follows
The course follows a
The course follows a monthly
The course follows a monthly subscription
```

The last word becomes the target label.

### 3. Sequence Padding

All sequences are padded to a fixed length before training.

### 4. Model Training

The model learns to predict the next word using:

* Embedding Layer
* LSTM Layer
* Fully Connected Output Layer

---

## Model Architecture

```python
Embedding(vocab_size, 100)
        ↓
LSTM(100, 150)
        ↓
Linear(150, vocab_size)
```

### Architecture Details

| Layer     | Description                                   |
| --------- | --------------------------------------------- |
| Embedding | Converts token IDs into dense vectors         |
| LSTM      | Learns contextual relationships between words |
| Linear    | Produces probabilities over vocabulary        |

---

## Training Configuration

| Parameter     | Value            |
| ------------- | ---------------- |
| Epochs        | 50               |
| Batch Size    | 32               |
| Learning Rate | 0.001            |
| Optimizer     | Adam             |
| Loss Function | CrossEntropyLoss |

---

## Results

Training loss decreases consistently:

```text
Epoch 1  : 166.20
Epoch 10 : 59.46
Epoch 20 : 17.34
Epoch 30 : 7.76
Epoch 40 : 5.42
Epoch 50 : 4.31
```

---

## Text Generation Example

Input:

```python
input_text = "hi how are"
```

Generated:

```text
hi how are to
hi how are to make
hi how are to make our
hi how are to make our payments
hi how are to make our payments ?
hi how are to make our payments ? your
hi how are to make our payments ? your youtube
hi how are to make our payments ? your youtube channel
hi how are to make our payments ? your youtube channel or
hi how are to make our payments ? your youtube channel or website
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/next-word-prediction-lstm.git
cd next-word-prediction-lstm
```

Install dependencies:

```bash
pip install torch nltk numpy
```

Download NLTK tokenizer:

```python
import nltk
nltk.download('punkt')
```

---

## Run the Project

Train the model:

```bash
python train.py
```

Generate predictions:

```python
prediction(model, vocab, "The course follows a monthly")
```

---

## Future Improvements

* Use larger datasets
* Add GRU and Transformer models for comparison
* Implement Beam Search decoding
* Save and load trained models
* Build a Streamlit web application
* Train on domain-independent corpora

---

## Learning Outcomes

This project covers:

* Natural Language Processing fundamentals
* Tokenization and vocabulary creation
* Sequence modeling
* Word embeddings
* LSTM networks
* Language modeling
* Text generation with PyTorch

---

## Author

Adarsh Anand

If you found this project useful, feel free to ⭐ the repository.
