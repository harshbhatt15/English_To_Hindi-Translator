# 🌐 English to Hindi Neural Machine Translation

An AI-powered **English → Hindi Neural Machine Translation (NMT)** system built using a **Seq2Seq LSTM Encoder-Decoder** architecture with TensorFlow/Keras. The project translates English sentences into Hindi through a trained deep learning model and provides an interactive web interface using Streamlit.

---

## 🚀 Live Demo

🔗 **Hugging Face:** https://huggingface.co/spaces/HARSH150308/english-to-hindi-translator

---

## 📌 Features

* 🌐 English → Hindi Neural Machine Translation
* 🧠 Seq2Seq LSTM Encoder-Decoder Architecture
* 📚 Trained on **104,084** English-Hindi sentence pairs
* 📊 Validated on **26,021** sentence pairs
* ⚡ Interactive Streamlit Web Application
* 💾 Pre-trained model for real-time inference
* 🔄 Encoder and Decoder models for inference
* 📝 Simple and user-friendly interface

---

## 📊 Model Performance

| Metric                   | Value                        |
| ------------------------ | ---------------------------- |
| Training Samples         | **104,084**                  |
| Validation Samples       | **26,021**                   |
| Best Validation Accuracy | **75.35%**                   |
| Best Validation Loss     | **1.3787**                   |
| Architecture             | Seq2Seq LSTM Encoder-Decoder |
| Framework                | TensorFlow / Keras           |

---

## 🏗️ Model Architecture

* Encoder-Decoder Neural Network
* LSTM-based Sequence-to-Sequence Model
* Keras Tokenizer
* Teacher Forcing during Training
* Separate Encoder & Decoder Models for Inference
* Early Stopping
* ReduceLROnPlateau Callback

---

## 🛠️ Tech Stack

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Streamlit
* Git
* GitHub

---

## 📂 Project Structure

```text
English_To_Hindi_Translator/
│
├── saved_models/
│   ├── encoder_model.h5
│   ├── decoder_model.h5
│   ├── input_tokenizer.pickle
│   ├── output_tokenizer.pickle
│   └── config.pickle
│
├── streamlit_app.py
├── requirements.txt
├── README.md
└── training_history.png
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/harshbhatt15/english_to_hindi-translator.git
cd english_to_hindi-translator
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run streamlit_app.py
```

---

## 🧪 Example

**Input**

```text
How are you?
```

**Output**

```text
आप कैसे हैं?
```

---

## 📈 Training Details

* Total Epochs: **13**
* Best Model Saved Automatically
* Learning Rate Reduction using ReduceLROnPlateau
* Early Stopping to Prevent Overfitting
* Model Checkpoint for Best Validation Loss

---

## 🔮 Future Improvements

* 🔹 Transformer-based Neural Machine Translation
* 🔹 Attention Mechanism
* 🔹 BLEU Score Evaluation
* 🔹 Support for Multiple Indian Languages
* 🔹 Speech-to-Text Translation
* 🔹 Text-to-Speech Output

---

## 👨‍💻 Author

**Harsh Bhatt**

* 🎓 BCA Student
* 🤖 AI & Machine Learning Enthusiast
* 💻 Python | Deep Learning | NLP
