# 🩺 **Symptom Checker Chatbot**

A lightweight **AI-powered healthcare chatbot** that predicts possible medical conditions based on user-entered symptoms.
Uses a **hybrid AI approach**:

* 🤖 Machine Learning (TF-IDF + Naive Bayes)
* 🧠 Rule-based medical logic
* ⚡ Streamlit UI

Perfect for healthcare AI, ML, NLP, and real-world chatbot systems.

---

## 🚀 **Features**

* Predicts medical conditions from natural language symptoms
* Hybrid rule-based + ML model for better accuracy
* Fast and fully offline
* Clean Streamlit interface
* Easy to extend (add more symptoms or conditions)

---

## 📁 **Project Structure**

```
symptom-checker-chatbot/
│── model.pkl
│── requirements.txt
│── README.md
│── data/
│   └── clinical_notes_dataset.csv
│── src/
│   ├── preprocess.py
│   ├── train_model.py
│   └── chatbot.py
│── app/
│   └── app.py
```

---

## 🧠 **How It Works**

### 1️⃣ Rule-Based System

Maps common symptoms to conditions:

```
headache → Migraine  
cold → Common Cold  
chest pain → Possible Heart Attack  
```

### 2️⃣ Machine Learning

Trained on your clinical dataset to predict diagnoses using TF-IDF + Naive Bayes.

---

## 🧪 **Train the Model**

```
python src/train_model.py
```

This generates:

```
model.pkl
```

---

## ▶️ **Run the Chatbot**

Install dependencies:

```
pip install -r requirements.txt
```

Run Streamlit app:

```
streamlit run app/app.py
```

---

## 🩹 **Example**

**Input:**

```
I have headache and nausea
```

**Output:**

```
Possible Condition: Migraine
```

---

## ⚠️ **Disclaimer**

This chatbot is for **educational purposes only.**
It is **not** a medical diagnostic tool.

---

## 👩‍💻 **Author**

**Sri Harshitha Anugu**
AI / ML Engineer

---

