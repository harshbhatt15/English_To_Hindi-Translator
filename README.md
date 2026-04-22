# 🌐 English to Hindi Translator

A Deep Learning based English → Hindi Translator built using Seq2Seq LSTM architecture and deployed with a Flask web application.

---

## 🚀 Features

* 🔤 Translate English text to Hindi
* 🧠 Seq2Seq LSTM Neural Network
* 🌐 Flask-based Web Application
* 📜 Translation History
* 💾 Export translations as CSV
* ⚡ Interactive UI with real-time translation

---

## 🛠️ Tech Stack

* **Python**
* **TensorFlow / Keras**
* **Flask**
* **NumPy, Pandas**
* **HTML, CSS, JavaScript**

---

## 📂 Project Structure

```
English_To_Hindi_Translator/
│
├── app.py
├── templates/
│   └── index.html
├── saved_models/
│   ├── encoder_model.h5
│   ├── decoder_model.h5
│   ├── input_tokenizer.pickle
│   ├── output_tokenizer.pickle
│   └── config.pickle
├── main.ipynb
├── training_history.png
└── README.md
```

---

## ▶️ How to Run

### 1️⃣ Clone the repository

```
git clone https://github.com/YOUR_USERNAME/English_To_Hindi-Translator.git
cd English_To_Hindi-Translator
```

---

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

### 3️⃣ Run the Flask app

```
python app.py
```

---

### 4️⃣ Open in browser

```
http://127.0.0.1:5000/
```

---

## 🧠 Model Details

* Encoder-Decoder Architecture
* LSTM layers for sequence learning
* Tokenization using Keras Tokenizer
* Trained on English-Hindi sentence pairs

---

## ⚠️ Note

* Model files (`.h5`, `.pickle`) may not be included due to size limits.
* You may need to train the model or download pre-trained weights.

---

## 📸 Demo

*Add screenshots here (UI / translation output)*

---

## 📈 Future Improvements

* 🌍 Support multiple languages
* 🎤 Speech-to-text input
* ☁️ Deployment on cloud (Render / AWS)

---

## 👨‍💻 Author

**Harsh Bhatt**
**BCA**
