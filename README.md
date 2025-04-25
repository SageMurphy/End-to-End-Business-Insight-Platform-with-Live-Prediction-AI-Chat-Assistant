# 📊 Data Science Internship - Assessment

## 🔥 Project Title:
**End-to-End Business Insight Platform with Live Prediction & AI Chat Assistant**

---

## 📌 Objective
Build a full-fledged analytics system that:
- Simulates dynamic, real-time business data.
- Predicts key metrics using an ML model.
- Offers an AI-powered RAG-based chatbot for answering queries from internal documents.
- Presents all insights via an intuitive Streamlit dashboard.

---

## ⚙️ Core Components

### 1. 🔄 Real-Time Data Generator
- **Domain**: E-commerce
- **Data Fields**:
  - Product Type
  - Price
  - Clicks/Views
- **Output**:
  - Continuously updating Pandas DataFrame with timestamps

### 2. 🧠 Machine Learning Module
- **Model Type**: Linear Regression
- **Problem Statement**: Sales Prediction
- **Library**: Scikit-learn
- **Persistence**: `.joblib` format

### 3. 📊 Streamlit Dashboard
- Real-time data preview and predictions
- Interactive graphs (line, bar charts)
- Filters (category selector)
- Simulation controls (speed adjustment)

### 4. 🤖📚 RAG-Based Document Chatbot
- **Input Documents**: 3–5 educational PDFs (topics like Python, ML, SQL)
- **Processing**:
  - Text extraction via PyMuPDF / PDFPlumber
  - Vector embeddings stored in Pinecone
  - LLM for answer generation (OpenAI, HuggingFace, etc.)
- **Interface**:
  - User text input
  - Relevant doc retrieval
  - Answer generation via LLM, displayed on the dashboard

### 5. 🧩 Unified System Integration
- Tabbed or sidebar-based Streamlit interface
- ML prediction and Chatbot access in one app

---

## 🛠️ How to Run

### Setup Instructions
```bash
# Clone repo
[https://github.com/yourusername/data-insight-platform.git](https://github.com/SageMurphy/End-to-End-Business-Insight-Platform-with-Live-Prediction-AI-Chat-Assistant)
cd data-insight-platform

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

### Project Structure
```
├── app.py                         # Unified Streamlit interface
├── simulate_data.py              # Script for live data generation
├── model_training.ipynb          # Notebook for training ML model
├── model.joblib                  # Saved model
├── chatbot.py                    # Backend for RAG chatbot
├── documents/                    # Folder containing PDF files
├── screenshots/                  # Dashboard and chatbot UI images
└── README.md                     # Project overview file
```
### Screenshots
![DASHBOARD2025-04-25T13_18_11 450Z](https://github.com/user-attachments/assets/128beb43-4b2a-45d7-a8a5-3096adbf37c0)

![CHATBOT025-04-25T13_17_35 675Z](https://github.com/user-attachments/assets/7697373a-3298-4042-abd9-4b98c0a1a1cc)

---

## 🧰 Tech Stack & Tools
- Python, Pandas, Numpy
- Scikit-learn
- Streamlit
- PyMuPDF / PDFPlumber
- Pinecone
- OpenAI / HuggingFace Transformers

---

## 📸 Showcase
- ✅ Real-Time Simulation Dashboard
- ✅ ML Prediction Interface
- ✅ Interactive Chatbot Section

---

## 📚 Resources
- [Streamlit Docs](https://docs.streamlit.io/)
- [Pinecone RAG Guide](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [Scikit-learn Docs](https://scikit-learn.org/stable/)
- [PyMuPDF Docs](https://pymupdf.readthedocs.io/)
- [HuggingFace Models](https://huggingface.co/models)
- [Python Random Docs](https://docs.python.org/3/library/random.html)

---

## 👥 Contributors
- Abhishek Shrimali
- Data Science Internship Assessment 2025 Cohort

---

 

