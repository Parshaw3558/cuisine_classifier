🍽️ Cuisine Classification ML App  
A machine learning project that predicts the **cuisine type** of a restaurant using details like restaurant name, city, and average cost for two.  
This project includes a **model training script**, a **Streamlit web app**, and all required ML assets.

---

📌 Project Structure

├── app.py                 
├── model.py               
├── Dataset.csv            
├── cuisine_model.pkl      
├── vectorizer.pkl        
├── label_encoder.pkl      
└── README.md            



📊 Project Overview

This project uses:
- **TF-IDF Vectorization**
- **LinearSVC (Support Vector Machine)**
- **Label Encoding for cuisine categories**

The model learns cuisine types from:
- Restaurant Name  
- City  
- Average Cost for Two  

The Streamlit app allows users to input restaurant details and get an instant cuisine prediction.

---

## 🧠 Model Details

### **model.py (Training Script)**  
The training script performs:
1. Load and clean dataset  
2. Prepare text data  
3. Encode labels  
4. Vectorize text using **TF-IDF**  
5. Train SVM classifier (**LinearSVC**)  
6. Evaluate accuracy  
7. Save the trained model and encoders

### ✔ Output Files:
- `cuisine_model.pkl`  
- `vectorizer.pkl`  
- `label_encoder.pkl`

These files are loaded by the Streamlit app.

---

## 🚀 How to Run the Streamlit App

### **1️⃣ Install Dependencies**
```bash
pip install streamlit scikit-learn joblib pandas
```

### **2️⃣ Place All Files in the Same Folder**
Make sure:
- `app.py`
- `cuisine_model.pkl`
- `vectorizer.pkl`
- `label_encoder.pkl`

are in the same directory.

### **3️⃣ Run the App**
```bash
streamlit run app.py
```

### ✔ App Features:
- Takes **Restaurant Name**, **City**, **Average Cost for Two**
- Predicts cuisine using trained ML model
- Simple and fast UI

---

## 🧪 Model Training (Optional)

To retrain the model, just run:

```bash
python model.py
```

This will:
- Load **Dataset.csv**
- Train a new model
- Generate new `pkl` files

---

## 🗂 Dataset Requirements

Your CSV must contain the following columns:

| Column Name | Description |
|------------|-------------|
| Restaurant Name | Restaurant title |
| City | Restaurant location |
| Average Cost for two | Cost estimate |
| Cuisines | Target label |

If these columns are missing, the script will raise an error.

---

## 🧩 Tech Stack

| Component | Technology |
|----------|------------|
| Frontend | Streamlit |
| Machine Learning | Scikit-learn |
| Vectorization | TF-IDF |
| Model | Linear SVM (LinearSVC) |
| Encoding | LabelEncoder |

---

## 📦 Deployment Options

You can deploy on:
- **Streamlit Cloud**
- **Hugging Face Spaces**
- **Render**
- **Railway.app**
- **Localhost**

For Streamlit Cloud:
Just upload:
- app.py  
- all `.pkl` files  
- Dataset.csv (optional)  
into one GitHub repo.

---

## ✨ Credits
Developed by **Parshaw** using Python and Streamlit.

---

## 📧 Support
If you need help with deployment, packaging, or project improvement, feel free to ask!
