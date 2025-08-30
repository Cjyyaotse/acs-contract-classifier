# ACS-contract-classifier
## 📑 Contract Classifier API

🚀 A FastAPI-based service that classifies contract documents into their respective categories using multiple strategies:

Few-Shot Classification (example-based, lightweight)

TF-IDF + Logistic Regression (classical ML pipeline)

LLM-based Classification (powered by OpenAI GPT models)

The API supports both raw text input and PDF uploads.
If a PDF is scanned (image-based), the system automatically uses OCRmyPDF (built on Tesseract OCR) to extract text before classification.

## ✨ Features

📄 Classify text or PDF contracts into categories such as Employment, NDA, Partnership, Service, Vendor.

🤖 Three independent services:

Few-Shot Router → Uses handcrafted examples.

TF-IDF + Logistic Regression → Machine learning trained model.

LLM Classifier → Uses OpenAI LLMs for zero-shot/few-shot inference.

🔍 OCR support via OCRmyPDF for scanned/image-based PDFs.

⚡ FastAPI backend with async endpoints.

🐳 Docker support for deployment.

🏗️ System Dependencies

To run OCR, install the following system packages:

sudo apt-get update && sudo apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    build-essential

## 🚀 Installation

Clone the repository

git clone https://github.com/Cjyyaotse/acs-contract-classifier.
cd contract-classifier


Create and activate virtual environment (optional)

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows


Install dependencies

pip install -r requirements.txt


Run the app

python src/app.py


The API will be available at:
## 👉 http://127.0.0.1:8000/

## 📡 API Endpoints
Root

GET / → Health check (verifies API is running)

Few-Shot Classifier

POST /few-shot/classify → Classify raw text

POST /few-shot/classify-pdf → Upload a PDF and classify

TF-IDF + Logistic Regression Classifier

POST /tf_logistic_regression/classify → Classify raw text

POST /tf_logistic_regression/classify-pdf → Upload a PDF and classify

LLM Classifier

POST /llm/classify → Classify raw text

POST /llm/classify-pdf → Upload a PDF and classify

## 🧠 Categories

Currently supported contract categories:

Employment

NDA

Partnership

Service

Vendor

## 🏋️ Model Training (TF-IDF + Logistic Regression)

If you want to retrain the TF-IDF + Logistic Regression model:

Prepare your dataset (e.g., a CSV with columns: text, label). Example:

text	label
"This agreement is between employer..."	Employment
"Confidential information shall not..."	NDA
"This partnership agreement is made..."	Partnership
"This services is ..."                  Service
"This vendor agreement is made..."      Vendor

Run the training notebook in "notebooks/TF_IDF_logistic_regression.ipynb:

Place the trained model in the models/ directory (already expected by the app).

🐳 Running with Docker

You can build and run the app inside Docker:

docker build -t contract-classifier .
docker run -d -p 8000:8000 contract-classifier

## 📖 Example Usage
Classify Raw Text
curl -X POST "http://127.0.0.1:8000/tf_logistic_regression/classify" \
     -H "Content-Type: application/json" \
     -d '{"text": "This agreement is made between employer and employee..."}'

Classify PDF
curl -X POST "http://127.0.0.1:8000/tf_logistic_regression/classify-pdf" \
     -F "file=@contract.pdf"

🛠️ Tech Stack

FastAPI
 – Web framework

Scikit-learn
 – ML pipeline

OpenAI API
 – LLM inference

OCRmyPDF
 + Tesseract
 – OCR

## 📜 License

MIT License – feel free to use and modify.