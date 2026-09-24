## Use Case: Next-Word Predictor & Smart Compose (Text Generation)

### 1. Problem Overview
Real-time writing assistance (such as email autocompletion or predictive keyboards) requires models that understand grammatical context and anticipate user intent. This project implements a **Next-Word Predictor** using a Long Short-Term Memory (LSTM) recurrent neural network to generate sequence-aware text suggestions given an input prompt.

### 2. Framing as a Supervised Learning Task
To train a sequence model on raw text without manual labeling, the problem is formulated as a self-supervised next-token classification task:

- **Input Sequences ($X$):** A rolling window of $n$ consecutive words or tokens representing prior context (e.g., `["machine", "learning", "is"]`).
- **Target Label ($y$):** The immediate subsequent word that follows the sequence (e.g., `"transformative"`), one-hot encoded over the corpus vocabulary.
- **Objective:** Minimize categorical cross-entropy loss by predicting the probability distribution over the vocabulary:
  $$\hat{y} = \text{softmax}(W \cdot h_t + b)$$

### 3. Architecture & Data Pipeline
1. **Tokenization & Sequencing:** Text corpus is tokenized, indexed, and split into overlapping n-gram sequences to generate feature-label pairs.
2. **Embedding Layer:** Maps integer token IDs into dense vector representations to capture semantic proximity between terms.
3. **LSTM Layer(s):** Employs memory cells with input, forget, and output gating mechanisms to retain long-range context and address the vanishing gradient problem inherent in standard RNNs.
4. **Dense Output Layer:** A fully connected layer with a softmax activation function outputting probabilities across the full vocabulary size ($V$).

### 4. Inference & Generation Strategies
At runtime, the model receives a seed string and iteratively predicts the next word using one of two decoding methods:
- **Greedy Search:** Selects the token with the highest argmax probability (deterministic, best for precise completions).
- **Temperature Sampling:** Adjusts logits before softmax to balance fluency and lexical diversity, preventing repetitive generation loops in long-form generation.
