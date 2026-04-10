# 😊 Facial Expression Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?style=for-the-badge&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-brightgreen?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**A real-time facial expression recognition system powered by deep learning.**

🔗 **[Live Demo](https://facial-expression-detector-omerkhan.streamlit.app)** — Try it in your browser, no installation needed!

</div>

---

## 🎯 What does this project do?

This app detects human faces in an image or live webcam feed and predicts the **emotional expression** of each face in real time. It can recognize **7 different emotions** with confidence scores.

---

## 😄 Supported Expressions

| Emoji | Expression |
|-------|-----------|
| 😡 | Angry |
| 🤢 | Disgust |
| 😨 | Fear |
| 😄 | Happy |
| 😐 | Neutral |
| 😢 | Sad |
| 😲 | Surprise |

---

## 🚀 Features

- 📸 **Image Upload** — Upload any photo and detect expressions on all faces
- 📷 **Live Webcam** — Real-time expression detection through your browser camera
- 📊 **Confidence Scores** — Shows probability for each expression class
- 👥 **Multiple Faces** — Detects and analyzes multiple faces in one image
- 🎨 **Beautiful UI** — Modern dark glassmorphism design

---

## 🧠 How it works

```
Input Image / Webcam Frame
        ↓
Face Detection (MTCNN - facenet-pytorch)
        ↓
Face Crop → Grayscale → Resize to 260×260
        ↓
Expression Prediction (EfficientNet-B2)
        ↓
Display Result with Label + Confidence
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning Model | EfficientNet-B2 (PyTorch) |
| Face Detection | MTCNN (facenet-pytorch) |
| Web App | Streamlit |
| Live Webcam | streamlit-webrtc |
| Model Hosting | Hugging Face |
| App Hosting | Streamlit Cloud |
| Language | Python 3.10 |

---

## 📁 Project Structure

```
expression-detector/
├── app.py              # Main Streamlit web app
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

> 📦 The trained model (`my_model.pth`) is hosted on [Hugging Face](https://huggingface.co/omer-khan/Facial-expression-detector-system) and downloaded automatically when the app starts.

---

## 💻 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR-USERNAME/expression-detector.git
cd expression-detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the model
Place `my_model.pth` in the same folder — or let the app download it automatically.

### 4. Run the app
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

> ⚡ **Tip:** Webcam works best when running locally on your own PC.

---

## 📊 Model Details

| Detail | Info |
|--------|------|
| Architecture | EfficientNet-B2 |
| Input Size | 260 × 260 px |
| Input Format | Grayscale (3-channel) |
| Output Classes | 7 expressions |
| Training Data | Custom dataset |
| Framework | PyTorch |

---

## ⚠️ Limitations

- Webcam detection is slow on cloud deployment due to network latency — best used locally
- Works best with **frontal, well-lit faces**
- Small or far-away faces (e.g. CCTV footage) may have lower accuracy

---

## 👨‍💻 Author

**Omer Khan**
- 🌐 Live App: [facial-expression-detector-omerkhan.streamlit.app](https://facial-expression-detector-omerkhan.streamlit.app)
- 🤗 Model: [Hugging Face](https://huggingface.co/omer-khan/Facial-expression-detector-system)
- 💻 GitHub: [@omer-khan](https://github.com/omer-khan)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">
Built with ❤️ using PyTorch + Streamlit
</div>
