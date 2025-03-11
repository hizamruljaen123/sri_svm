import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
from urllib.parse import urlparse, parse_qs
from sklearn.feature_extraction.text import CountVectorizer
from google_play_scraper import reviews, Sort
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import pickle
from flask_cors import CORS
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import subprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
CORS(app)
stop_words = set(stopwords.words('indonesian'))

def preprocess_comment(comment: str) -> str:
    tokens = word_tokenize(comment.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(tokens)

def train_svm_manual(X, y, learning_rate=0.001, epochs=1000, lambda_param=0.01):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    for _ in range(epochs):
        for idx, x_i in enumerate(X):
            condition = y[idx] * (np.dot(x_i, weights) - bias) >= 1
            if condition:
                weights -= learning_rate * (2 * lambda_param * weights)
            else:
                weights -= learning_rate * (2 * lambda_param * weights - np.dot(x_i, y[idx]))
                bias -= learning_rate * y[idx]
    return weights, bias

def predict_svm_manual(X, weights, bias):
    linear_output = np.dot(X, weights) - bias
    return np.sign(linear_output)

def evaluate_model_manual(X_train, y_train, X_test, y_test, model):
    """
    Evaluasi model secara manual untuk klasifikasi biner (positif dan negatif).
    Menghitung akurasi, presisi, recall, dan F1-score.
    """
    # Prediksi label untuk data latih dan data uji
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Filter hanya kelas positif (1) dan negatif (-1)
    valid_indices_train = np.isin(y_train, [-1, 1])
    valid_indices_test = np.isin(y_test, [-1, 1])
    
    y_train_filtered = y_train[valid_indices_train]
    y_train_pred_filtered = y_train_pred[valid_indices_train]
    
    y_test_filtered = y_test[valid_indices_test]
    y_test_pred_filtered = y_test_pred[valid_indices_test]
    
    # Fungsi untuk menghitung confusion matrix secara manual
    def calculate_confusion_matrix(y_true, y_pred):
        TP = sum((y_true == 1) & (y_pred == 1))  # True Positives
        FP = sum((y_true == -1) & (y_pred == 1))  # False Positives
        TN = sum((y_true == -1) & (y_pred == -1))  # True Negatives
        FN = sum((y_true == 1) & (y_pred == -1))  # False Negatives
        return TP, FP, TN, FN
    
    # Hitung confusion matrix untuk data latih dan data uji
    TP_train, FP_train, TN_train, FN_train = calculate_confusion_matrix(y_train_filtered, y_train_pred_filtered)
    TP_test, FP_test, TN_test, FN_test = calculate_confusion_matrix(y_test_filtered, y_test_pred_filtered)
    
    # Hitung metrik evaluasi untuk data latih
    accuracy_train = (TP_train + TN_train) / (TP_train + FP_train + TN_train + FN_train)
    precision_train = TP_train / (TP_train + FP_train) if (TP_train + FP_train) > 0 else 0
    recall_train = TP_train / (TP_train + FN_train) if (TP_train + FN_train) > 0 else 0
    f1_train = 2 * (precision_train * recall_train) / (precision_train + recall_train) if (precision_train + recall_train) > 0 else 0
    
    # Hitung metrik evaluasi untuk data uji
    accuracy_test = (TP_test + TN_test) / (TP_test + FP_test + TN_test + FN_test)
    precision_test = TP_test / (TP_test + FP_test) if (TP_test + FP_test) > 0 else 0
    recall_test = TP_test / (TP_test + FN_test) if (TP_test + FN_test) > 0 else 0
    f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test) if (precision_test + recall_test) > 0 else 0
    
    # Plot confusion matrix untuk data latih
    cm_train = [[TN_train, FP_train], [FN_train, TP_train]]
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', xticklabels=["Negatif", "Positif"], yticklabels=["Negatif", "Positif"])
    plt.title('Confusion Matrix - Training Data')
    plt.xlabel('Predictions')
    plt.ylabel('Actual')
    cm_train_img_path = 'static/graph/confusion_matrix_train.png'
    plt.savefig(cm_train_img_path)
    plt.close()
    
    # Plot confusion matrix untuk data uji
    cm_test = [[TN_test, FP_test], [FN_test, TP_test]]
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=["Negatif", "Positif"], yticklabels=["Negatif", "Positif"])
    plt.title('Confusion Matrix - Testing Data')
    plt.xlabel('Predictions')
    plt.ylabel('Actual')
    cm_test_img_path = 'static/graph/confusion_matrix_test.png'
    plt.savefig(cm_test_img_path)
    plt.close()
    
    # Return evaluation metrics and confusion matrix image paths
    return {
        "accuracy_train": accuracy_train,
        "precision_train": precision_train,
        "recall_train": recall_train,
        "f1_train": f1_train,
        "accuracy_test": accuracy_test,
        "precision_test": precision_test,
        "recall_test": recall_test,
        "f1_test": f1_test,
        "confusion_matrix_train_path": cm_train_img_path,
        "confusion_matrix_test_path": cm_test_img_path
    }
def evaluate_model(X_train, y_train, X_test, y_test, model):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    accuracy_train = accuracy_score(y_train, y_train_pred)
    precision_train = precision_score(y_train, y_train_pred, average='binary')
    recall_train = recall_score(y_train, y_train_pred, average='binary')
    f1_train = f1_score(y_train, y_train_pred, average='binary')
    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision_test = precision_score(y_test, y_test_pred, average='binary')
    recall_test = recall_score(y_test, y_test_pred, average='binary')
    f1_test = f1_score(y_test, y_test_pred, average='binary')
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'])
    plt.title('Confusion Matrix - Training Data')
    plt.xlabel('Predictions')
    plt.ylabel('Actual')
    cm_train_img_path = 'static/graph/confusion_matrix_train.png'
    plt.savefig(cm_train_img_path)
    plt.close()
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'])
    plt.title('Confusion Matrix - Testing Data')
    plt.xlabel('Predictions')
    plt.ylabel('Actual')
    cm_test_img_path = 'static/graph/confusion_matrix_test.png'
    plt.savefig(cm_test_img_path)
    plt.close()
    return {
        "accuracy_train": accuracy_train,
        "precision_train": precision_train,
        "recall_train": recall_train,
        "f1_train": f1_train,
        "accuracy_test": accuracy_test,
        "precision_test": precision_test,
        "recall_test": recall_test,
        "f1_test": f1_test,
        "confusion_matrix_train_path": cm_train_img_path,
        "confusion_matrix_test_path": cm_test_img_path
    }

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    return encoded

@app.route('/show_data_latih', methods=['GET'])
def show_data_latih():
    gotube_train_file = "static/data/gotube/data_latih_labeled.xlsx"
    youtube_train_file = "static/data/youtube/data_latih_labeled.xlsx"
    # Pastikan kedua file ada
    if not os.path.exists(gotube_train_file) or not os.path.exists(youtube_train_file):
        missing_files = []
        if not os.path.exists(gotube_train_file):
            missing_files.append(gotube_train_file)
        if not os.path.exists(youtube_train_file):
            missing_files.append(youtube_train_file)
        return jsonify({
            "status": "error",
            "message": f"File berikut tidak ditemukan: {', '.join(missing_files)}"
        }), 404
    
    # Baca data dari kedua file
    gotube_df = pd.read_excel(gotube_train_file)
    youtube_df = pd.read_excel(youtube_train_file)
    
    # Tambahkan kolom "sumber" untuk identifikasi
    gotube_df["sumber"] = "GoTube"
    youtube_df["sumber"] = "YouTube"
    
    # Gabungkan data dari GoTube dan YouTube
    combined_df = pd.concat([gotube_df, youtube_df], ignore_index=True)
    
    # Preprocessing kolom "komentar" jika ada
    if "komentar" in combined_df.columns:
        combined_df["processed_komentar"] = combined_df["komentar"].apply(preprocess_comment)
    
    # Kembalikan data dalam format JSON
    return jsonify(combined_df.to_dict(orient='records'))
def evaluate_model_manual(X_train, y_train, X_test, y_test, model):
    """
    Evaluasi model secara manual untuk klasifikasi biner (positif dan negatif).
    Menghitung akurasi, presisi, recall, dan F1-score.
    """
    # Prediksi label untuk data latih dan data uji
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Filter hanya kelas positif (1) dan negatif (-1)
    valid_indices_train = np.isin(y_train, [-1, 1])
    valid_indices_test = np.isin(y_test, [-1, 1])
    
    y_train_filtered = y_train[valid_indices_train]
    y_train_pred_filtered = y_train_pred[valid_indices_train]
    
    y_test_filtered = y_test[valid_indices_test]
    y_test_pred_filtered = y_test_pred[valid_indices_test]
    
    # Fungsi untuk menghitung confusion matrix secara manual
    def calculate_confusion_matrix(y_true, y_pred):
        TP = sum((y_true == 1) & (y_pred == 1))  # True Positives
        FP = sum((y_true == -1) & (y_pred == 1))  # False Positives
        TN = sum((y_true == -1) & (y_pred == -1))  # True Negatives
        FN = sum((y_true == 1) & (y_pred == -1))  # False Negatives
        return TP, FP, TN, FN
    
    # Hitung confusion matrix untuk data latih dan data uji
    TP_train, FP_train, TN_train, FN_train = calculate_confusion_matrix(y_train_filtered, y_train_pred_filtered)
    TP_test, FP_test, TN_test, FN_test = calculate_confusion_matrix(y_test_filtered, y_test_pred_filtered)
    
    # Hitung metrik evaluasi untuk data latih
    accuracy_train = (TP_train + TN_train) / (TP_train + FP_train + TN_train + FN_train)
    precision_train = TP_train / (TP_train + FP_train) if (TP_train + FP_train) > 0 else 0
    recall_train = TP_train / (TP_train + FN_train) if (TP_train + FN_train) > 0 else 0
    f1_train = 2 * (precision_train * recall_train) / (precision_train + recall_train) if (precision_train + recall_train) > 0 else 0
    
    # Hitung metrik evaluasi untuk data uji
    accuracy_test = (TP_test + TN_test) / (TP_test + FP_test + TN_test + FN_test)
    precision_test = TP_test / (TP_test + FP_test) if (TP_test + FP_test) > 0 else 0
    recall_test = TP_test / (TP_test + FN_test) if (TP_test + FN_test) > 0 else 0
    f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test) if (precision_test + recall_test) > 0 else 0
    
    # Plot confusion matrix untuk data latih
    cm_train = [[TN_train, FP_train], [FN_train, TP_train]]
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', xticklabels=["Negatif", "Positif"], yticklabels=["Negatif", "Positif"])
    plt.title('Confusion Matrix - Training Data')
    plt.xlabel('Predictions')
    plt.ylabel('Actual')
    cm_train_img_path = 'static/graph/confusion_matrix_train.png'
    plt.savefig(cm_train_img_path)
    plt.close()
    
    # Plot confusion matrix untuk data uji
    cm_test = [[TN_test, FP_test], [FN_test, TP_test]]
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=["Negatif", "Positif"], yticklabels=["Negatif", "Positif"])
    plt.title('Confusion Matrix - Testing Data')
    plt.xlabel('Predictions')
    plt.ylabel('Actual')
    cm_test_img_path = 'static/graph/confusion_matrix_test.png'
    plt.savefig(cm_test_img_path)
    plt.close()
    
    # Return evaluation metrics and confusion matrix image paths
    return {
        "accuracy_train": accuracy_train,
        "precision_train": precision_train,
        "recall_train": recall_train,
        "f1_train": f1_train,
        "accuracy_test": accuracy_test,
        "precision_test": precision_test,
        "recall_test": recall_test,
        "f1_test": f1_test,
        "confusion_matrix_train_path": cm_train_img_path,
        "confusion_matrix_test_path": cm_test_img_path
    }

@app.route('/evaluate_model', methods=['GET'])
def evaluate_model_route():
    """
    Endpoint untuk mengevaluasi model SVM secara terpisah untuk data YouTube dan GoTube,
    serta evaluasi pada data gabungan.
    """
    # Path ke file Excel
    gotube_train_file = "static/data/gotube/data_latih_labeled.xlsx"
    gotube_test_file = "static/data/gotube/data_uji_test.xlsx"
    youtube_train_file = "static/data/youtube/data_latih_labeled.xlsx"
    youtube_test_file = "static/data/youtube/data_uji_test.xlsx"
    
    # Pastikan semua file ada
    if not all(os.path.exists(file) for file in [gotube_train_file, gotube_test_file, youtube_train_file, youtube_test_file]):
        return jsonify({
            "status": "error",
            "message": "Beberapa file tidak ditemukan di folder gotube atau youtube."
        }), 404

    def evaluate_source_data(train_data, test_data, source_name):
        """Helper function untuk evaluasi data dari satu sumber"""
        # Preprocessing data training
        train_data["processed_komentar"] = train_data["komentar"].apply(preprocess_comment)
        y_train = train_data["label"].map({"positif": 1, "negatif": -1}).fillna(-1)
        
        # Preprocessing data testing
        test_data["processed_komentar"] = test_data["komentar"].apply(preprocess_comment)
        
        # Mapping rating ke sentimen untuk data testing
        def map_rating_to_sentiment(rating):
            if rating in [1, 2]:
                return -1  # negatif
            elif rating in [3, 4, 5]:
                return 1   # positif
            return 0  # netral
        
        y_test = test_data["rating"].apply(map_rating_to_sentiment)
        y_test = y_test.fillna(-1)
        
        # Load vectorizer dan model
        with open("vectorizer.pkl", "rb") as f_vec:
            vectorizer = pickle.load(f_vec)
        with open("svm_model.pkl", "rb") as f_model:
            svm_model = pickle.load(f_model)
        
        # Vectorize data
        X_train_vec = vectorizer.transform(train_data["processed_komentar"]).toarray()
        X_test_vec = vectorizer.transform(test_data["processed_komentar"]).toarray()
        
        # Evaluasi
        evaluation_results = evaluate_model_manual(X_train_vec, y_train, X_test_vec, y_test, svm_model)
        
        # Save confusion matrices with source-specific names
        plt.figure(figsize=(6, 6))
        sns.heatmap(confusion_matrix(y_train, svm_model.predict(X_train_vec)), 
                    annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Negatif", "Positif"],
                    yticklabels=["Negatif", "Positif"])
        plt.title(f'Confusion Matrix - {source_name} Training Data')
        plt.xlabel('Predictions')
        plt.ylabel('Actual')
        cm_train_path = f'static/graph/confusion_matrix_{source_name.lower()}_train.png'
        plt.savefig(cm_train_path)
        plt.close()
        
        plt.figure(figsize=(6, 6))
        sns.heatmap(confusion_matrix(y_test, svm_model.predict(X_test_vec)),
                    annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Negatif", "Positif"],
                    yticklabels=["Negatif", "Positif"])
        plt.title(f'Confusion Matrix - {source_name} Testing Data')
        plt.xlabel('Predictions')
        plt.ylabel('Actual')
        cm_test_path = f'static/graph/confusion_matrix_{source_name.lower()}_test.png'
        plt.savefig(cm_test_path)
        plt.close()
        
        return {
            "accuracy_train": evaluation_results["accuracy_train"],
            "precision_train": evaluation_results["precision_train"],
            "recall_train": evaluation_results["recall_train"],
            "f1_train": evaluation_results["f1_train"],
            "accuracy_test": evaluation_results["accuracy_test"],
            "precision_test": evaluation_results["precision_test"],
            "recall_test": evaluation_results["recall_test"],
            "f1_test": evaluation_results["f1_test"],
            "confusion_matrix_train_path": cm_train_path,
            "confusion_matrix_test_path": cm_test_path
        }
    
    # Load dan evaluasi data GoTube
    gotube_train = pd.read_excel(gotube_train_file)
    gotube_test = pd.read_excel(gotube_test_file)
    gotube_results = evaluate_source_data(gotube_train, gotube_test, "GoTube")
    
    # Load dan evaluasi data YouTube
    youtube_train = pd.read_excel(youtube_train_file)
    youtube_test = pd.read_excel(youtube_test_file)
    youtube_results = evaluate_source_data(youtube_train, youtube_test, "YouTube")
    
    # Gabungkan data latih dan data uji dari GoTube dan YouTube
    combined_train = pd.concat([gotube_train, youtube_train], ignore_index=True)
    combined_test = pd.concat([gotube_test, youtube_test], ignore_index=True)
    
    # Evaluasi pada data gabungan
    combined_results = evaluate_source_data(combined_train, combined_test, "Combined")
    
    # Return hasil evaluasi terpisah untuk kedua sumber dan data gabungan
    return jsonify({
        "status": "success",
        "gotube_evaluation": gotube_results,
        "youtube_evaluation": youtube_results,
        "combined_evaluation": combined_results
    })


def evaluate_model_manual_all(X_train, y_train, X_test, y_test, model):
    """
    Evaluasi model secara manual untuk klasifikasi biner (positif dan negatif).
    Menghitung akurasi, presisi, recall, dan F1-score.
    """
    # Prediksi label untuk data latih dan data uji
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Filter hanya kelas positif (1) dan negatif (-1)
    valid_indices_train = np.isin(y_train, [-1, 1])
    valid_indices_test = np.isin(y_test, [-1, 1])
    
    y_train_filtered = y_train[valid_indices_train]
    y_train_pred_filtered = y_train_pred[valid_indices_train]
    
    y_test_filtered = y_test[valid_indices_test]
    y_test_pred_filtered = y_test_pred[valid_indices_test]
    
    # Fungsi untuk menghitung confusion matrix secara manual
    def calculate_confusion_matrix(y_true, y_pred):
        TP = sum((y_true == 1) & (y_pred == 1))  # True Positives
        FP = sum((y_true == -1) & (y_pred == 1))  # False Positives
        TN = sum((y_true == -1) & (y_pred == -1))  # True Negatives
        FN = sum((y_true == 1) & (y_pred == -1))  # False Negatives
        return TP, FP, TN, FN
    
    # Hitung confusion matrix untuk data latih dan data uji
    TP_train, FP_train, TN_train, FN_train = calculate_confusion_matrix(y_train_filtered, y_train_pred_filtered)
    TP_test, FP_test, TN_test, FN_test = calculate_confusion_matrix(y_test_filtered, y_test_pred_filtered)
    
    # Hitung metrik evaluasi untuk data latih
    accuracy_train = (TP_train + TN_train) / (TP_train + FP_train + TN_train + FN_train)
    precision_train = TP_train / (TP_train + FP_train) if (TP_train + FP_train) > 0 else 0
    recall_train = TP_train / (TP_train + FN_train) if (TP_train + FN_train) > 0 else 0
    f1_train = 2 * (precision_train * recall_train) / (precision_train + recall_train) if (precision_train + recall_train) > 0 else 0
    
    # Hitung metrik evaluasi untuk data uji
    accuracy_test = (TP_test + TN_test) / (TP_test + FP_test + TN_test + FN_test)
    precision_test = TP_test / (TP_test + FP_test) if (TP_test + FP_test) > 0 else 0
    recall_test = TP_test / (TP_test + FN_test) if (TP_test + FN_test) > 0 else 0
    f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test) if (precision_test + recall_test) > 0 else 0
    
    # Return evaluation metrics
    return {
        "accuracy_train": accuracy_train,
        "precision_train": precision_train,
        "recall_train": recall_train,
        "f1_train": f1_train,
        "accuracy_test": accuracy_test,
        "precision_test": precision_test,
        "recall_test": recall_test,
        "f1_test": f1_test,
        "confusion_matrix_train": [[TN_train, FP_train], [FN_train, TP_train]],
        "confusion_matrix_test": [[TN_test, FP_test], [FN_test, TP_test]]
    }

@app.route('/evaluate_model_all', methods=['GET'])
def evaluate_model_route_1():
    """
    Endpoint untuk mengevaluasi model SVM secara terpisah untuk data YouTube dan GoTube,
    serta evaluasi pada data gabungan.
    """
    # Path ke file Excel
    gotube_train_file = "static/data/gotube/data_latih_labeled.xlsx"
    gotube_test_file = "static/data/gotube/data_uji_test.xlsx"
    youtube_train_file = "static/data/youtube/data_latih_labeled.xlsx"
    youtube_test_file = "static/data/youtube/data_uji_test.xlsx"
    
    # Pastikan semua file ada
    if not all(os.path.exists(file) for file in [gotube_train_file, gotube_test_file, youtube_train_file, youtube_test_file]):
        return jsonify({
            "status": "error",
            "message": "Beberapa file tidak ditemukan di folder gotube atau youtube."
        }), 404

    def evaluate_source_data(train_data, test_data, source_name):
        """Helper function untuk evaluasi data dari satu sumber"""
        # Preprocessing data training
        train_data["processed_komentar"] = train_data["komentar"].apply(preprocess_comment)
        y_train = train_data["label"].map({"positif": 1, "negatif": -1}).fillna(-1)
        
        # Preprocessing data testing
        test_data["processed_komentar"] = test_data["komentar"].apply(preprocess_comment)
        
        # Mapping rating ke sentimen untuk data testing
        def map_rating_to_sentiment(rating):
            if rating in [1, 2]:
                return -1  # negatif
            elif rating in [3, 4, 5]:
                return 1   # positif
            return 0  # netral
        
        y_test = test_data["rating"].apply(map_rating_to_sentiment)
        y_test = y_test.fillna(-1)
        
        # Load vectorizer dan model berdasarkan sumber
        model_path = f"static/models/svm_model_{source_name.lower()}.pkl"
        vectorizer_path = f"static/models/vectorizer_{source_name.lower()}.pkl"
        
        with open(vectorizer_path, "rb") as f_vec:
            vectorizer = pickle.load(f_vec)
        with open(model_path, "rb") as f_model:
            svm_model = pickle.load(f_model)
        
        # Vectorize data
        X_train_vec = vectorizer.transform(train_data["processed_komentar"]).toarray()
        X_test_vec = vectorizer.transform(test_data["processed_komentar"]).toarray()
        
        # Evaluasi
        evaluation_results = evaluate_model_manual_all(X_train_vec, y_train, X_test_vec, y_test, svm_model)
        
        return evaluation_results
    
    # Load dan evaluasi data GoTube
    gotube_train = pd.read_excel(gotube_train_file)
    gotube_test = pd.read_excel(gotube_test_file)
    gotube_results = evaluate_source_data(gotube_train, gotube_test, "GoTube")
    
    # Load dan evaluasi data YouTube
    youtube_train = pd.read_excel(youtube_train_file)
    youtube_test = pd.read_excel(youtube_test_file)
    youtube_results = evaluate_source_data(youtube_train, youtube_test, "YouTube")
    
    # Gabungkan data latih dan data uji dari GoTube dan YouTube
    combined_train = pd.concat([gotube_train, youtube_train], ignore_index=True)
    combined_test = pd.concat([gotube_test, youtube_test], ignore_index=True)
    
    # Evaluasi pada data gabungan
    combined_results = evaluate_source_data(combined_train, combined_test, "Combined")
    
    # Gabungkan semua confusion matrix dalam satu heatmap besar
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot confusion matrix untuk GoTube
    sns.heatmap(gotube_results["confusion_matrix_test"], annot=True, fmt='d', cmap='Greys',
                xticklabels=["Negatif", "Positif"], yticklabels=["Negatif", "Positif"],
                ax=axes[0], cbar=False)
    axes[0].set_title('Confusion Matrix - GoTube')
    axes[0].set_xlabel('Predictions')
    axes[0].set_ylabel('Actual')
    
    # Plot confusion matrix untuk YouTube
    sns.heatmap(youtube_results["confusion_matrix_test"], annot=True, fmt='d', cmap='Greys',
                xticklabels=["Negatif", "Positif"], yticklabels=["Negatif", "Positif"],
                ax=axes[1], cbar=False)
    axes[1].set_title('Confusion Matrix - YouTube')
    axes[1].set_xlabel('Predictions')
    axes[1].set_ylabel('Actual')
    
    # Plot confusion matrix untuk Combined
    sns.heatmap(combined_results["confusion_matrix_test"], annot=True, fmt='d', cmap='Greys',
                xticklabels=["Negatif", "Positif"], yticklabels=["Negatif", "Positif"],
                ax=axes[2], cbar=False)
    axes[2].set_title('Confusion Matrix - Combined')
    axes[2].set_xlabel('Predictions')
    axes[2].set_ylabel('Actual')
    
    # Simpan heatmap gabungan
    combined_heatmap_path = 'static/graph/combined_confusion_matrix.png'
    plt.tight_layout()
    plt.savefig(combined_heatmap_path)
    plt.close()
    
    # Return hasil evaluasi terpisah untuk kedua sumber dan data gabungan
    return jsonify({
        "status": "success",
        "gotube_evaluation": gotube_results,
        "youtube_evaluation": youtube_results,
        "combined_evaluation": combined_results,
        "combined_heatmap_path": combined_heatmap_path
    })
@app.route('/show_data_uji', methods=['GET'])
def show_data_uji():
    # Definisikan path ke file data uji dari GoTube dan YouTube
    gotube_test_file = "static/data/gotube/data_uji_test.xlsx"
    youtube_test_file = "static/data/youtube/data_uji_test.xlsx"
    
    # Pastikan kedua file ada
    if not os.path.exists(gotube_test_file) or not os.path.exists(youtube_test_file):
        missing_files = []
        if not os.path.exists(gotube_test_file):
            missing_files.append(gotube_test_file)
        if not os.path.exists(youtube_test_file):
            missing_files.append(youtube_test_file)
        return jsonify({
            "status": "error",
            "message": f"File berikut tidak ditemukan: {', '.join(missing_files)}"
        }), 404
    
    # Baca data dari kedua file
    gotube_df = pd.read_excel(gotube_test_file)
    youtube_df = pd.read_excel(youtube_test_file)
    
    # Tambahkan kolom "sumber" untuk identifikasi
    gotube_df["sumber"] = "GoTube"
    youtube_df["sumber"] = "YouTube"
    
    # Gabungkan data dari GoTube dan YouTube
    combined_df = pd.concat([gotube_df, youtube_df], ignore_index=True)
    
    # Preprocessing kolom "komentar" jika ada
    if "komentar" in combined_df.columns:
        combined_df["processed_komentar"] = combined_df["komentar"].apply(preprocess_comment)
    
    # Kembalikan data dalam format JSON
    return jsonify(combined_df.to_dict(orient='records'))

@app.route('/show_classification_results', methods=['GET'])
def show_classification_results():
    # File sumber dari GoTube dan YouTube
    gotube_train_file = "static/data/gotube/classification_results.xlsx"
    youtube_train_file = "static/data/youtube/classification_results.xlsx"

    # Periksa apakah kedua file ada
    if not os.path.exists(gotube_train_file) or not os.path.exists(youtube_train_file):
        return jsonify({
            "status": 404,
            "message": "Salah satu atau kedua file classification_results.xlsx tidak ditemukan."
        }), 404

    try:
        # Membaca file Excel
        gotube_df = pd.read_excel(gotube_train_file)
        youtube_df = pd.read_excel(youtube_train_file)

        # Menambahkan kolom sumber untuk identifikasi
        gotube_df["sumber"] = "GoTube"
        youtube_df["sumber"] = "YouTube"

        # Menggabungkan kedua data
        combined_df = pd.concat([gotube_df, youtube_df], ignore_index=True)

        # Menyiapkan hasil untuk dikembalikan
        results = []
        for _, row in combined_df.iterrows():
            prediksi_sentimen = row["prediksi_sentimen"] if pd.notna(row["prediksi_sentimen"]) else '-'
            results.append({
                "komentar": row["komentar"],
                "prediksi_sentimen": prediksi_sentimen,
                "sumber": row["sumber"]
            })

        return jsonify({
            "status": 200,
            "data_result": results
        })

    except Exception as e:
        return jsonify({
            "status": 500,
            "message": f"Terjadi kesalahan saat membaca file: {str(e)}"
        }), 500

@app.route('/show_tfidf', methods=['GET'])
def show_tfidf():
    
    # Path ke file Excel di kedua folder
    gotube_train_file = "static/data/gotube/data_latih_labeled.xlsx"
    gotube_test_file = "static/data/gotube/data_uji_test.xlsx"
    youtube_train_file = "static/data/youtube/data_latih_labeled.xlsx"
    youtube_test_file = "static/data/youtube/data_uji_test.xlsx"
    
    # Pastikan semua file ada
    if not all(os.path.exists(file) for file in [gotube_train_file, gotube_test_file, youtube_train_file, youtube_test_file]):
        return jsonify({
            "status": "error",
            "message": "Beberapa file tidak ditemukan di folder gotube atau youtube."
        }), 404
    
    # Load data latih dan data uji dari kedua folder
    gotube_train = pd.read_excel(gotube_train_file)
    gotube_test = pd.read_excel(gotube_test_file)
    youtube_train = pd.read_excel(youtube_train_file)
    youtube_test = pd.read_excel(youtube_test_file)
    
    # Gabungkan data latih dan data uji dari kedua sumber
    combined_train = pd.concat([gotube_train, youtube_train], ignore_index=True)
    combined_test = pd.concat([gotube_test, youtube_test], ignore_index=True)
    
    # Preprocess the comments
    combined_train["processed_komentar"] = combined_train["komentar"].apply(preprocess_comment)
    combined_test["processed_komentar"] = combined_test["komentar"].apply(preprocess_comment)
    
    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit the vectorizer on the combined training data and transform both datasets
    X_train_tfidf = vectorizer.fit_transform(combined_train["processed_komentar"])
    X_test_tfidf = vectorizer.transform(combined_test["processed_komentar"])
    
    # Get feature names (words) from the vectorizer
    feature_names = vectorizer.get_feature_names_out()
    
    # Convert sparse matrices to dense arrays
    X_train_tfidf_dense = X_train_tfidf.toarray()
    X_test_tfidf_dense = X_test_tfidf.toarray()
    
    # Save TF-IDF results for combined training data to a text file
    with open("static/tfidf_train.txt", "w", encoding="utf-8") as f_train:
        for i, row in enumerate(X_train_tfidf_dense):
            non_zero_indices = np.where(row > 0.0)[0]
            f_train.write(f"Dokumen {i}:\n")
            for idx in non_zero_indices:
                f_train.write(f"- {feature_names[idx]}: {row[idx]:.4f}\n")
            f_train.write("\n")
    
    # Save TF-IDF results for combined test data to a text file
    with open("static/tfidf_test.txt", "w", encoding="utf-8") as f_test:
        for i, row in enumerate(X_test_tfidf_dense):
            non_zero_indices = np.where(row > 0.0)[0]
            f_test.write(f"Dokumen {i}:\n")
            for idx in non_zero_indices:
                f_test.write(f"- {feature_names[idx]}: {row[idx]:.4f}\n")
            f_test.write("\n")
    
    # Prepare the response
    response = {
        "status": "success",
        "message": "TF-IDF berhasil dihitung dan disimpan dalam file .txt.",
        "tfidf_train_path": "static/tfidf_train.txt",
        "tfidf_test_path": "static/tfidf_test.txt"
    }
    
    return jsonify(response)

@app.route('/train_svm_manual', methods=['GET'])
def route_train_svm_manual():
    # Path ke file Excel di kedua folder
    gotube_file = "static/data/gotube/data_latih_labeled.xlsx"
    youtube_file = "static/data/youtube/data_latih_labeled.xlsx"
    
    # Pastikan kedua file ada
    if not os.path.exists(gotube_file) or not os.path.exists(youtube_file):
        return jsonify({
            "status": "error",
            "message": "File data_latih_labeled.xlsx tidak ditemukan di salah satu folder."
        }), 404
    
    # Load data dari kedua file
    try:
        gotube_data = pd.read_excel(gotube_file)
        youtube_data = pd.read_excel(youtube_file)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error membaca file Excel: {str(e)}"
        }), 500
    
    # Fungsi untuk melatih model SVM
    def train_and_save_model(data, source_name):
        # Handle NaN values dengan memberikan nilai default
        data["komentar"] = data["komentar"].fillna("tidak ada komentar")
        data["sentimen"] = data["sentimen"].fillna("netral")
        
        # Preprocessing komentar
        data["processed_komentar"] = data["komentar"].apply(preprocess_comment)
        
        # Map sentiment labels to numeric values (positif = 1, negatif = -1)
        sentimen_map = {"positif": 1, "negatif": -1, "netral": 0}
        data["label"] = data["sentimen"].map(sentimen_map)
        
        # Filter out neutral sentiments for binary classification
        filtered_data = data[data["label"] != 0].copy()
        X = filtered_data["processed_komentar"]
        y = filtered_data["label"]
        
        # Vectorization (convert text to numerical vectors)
        vectorizer_path = f"static/models/vectorizer_{source_name.lower()}.pkl"
        model_path = f"static/models/svm_model_{source_name.lower()}.pkl"
        
        # Pastikan file model dan vectorizer ada
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            return {
                "status": "error",
                "message": f"Model atau vectorizer untuk aplikasi '{source_name}' tidak ditemukan."
            }
        
        # Load vectorizer dan model yang sudah ada
        with open(vectorizer_path, "rb") as f_vec:
            vectorizer = pickle.load(f_vec)
        with open(model_path, "rb") as f_model:
            svm_model = pickle.load(f_model)
        
        # Vectorize data
        X_vec = vectorizer.transform(X).toarray()
        
        # Normalize the feature vectors
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_vec)
        
        # Apply t-SNE to reduce to 2D
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X_scaled)
        
        # Normalisasi hasil t-SNE
        tsne_scaler = StandardScaler()
        X_tsne_normalized = tsne_scaler.fit_transform(X_tsne)
        
        # Visualize the results using normalized t-SNE with hyperplane and margins
        plt.figure(figsize=(10, 6))
        colors = ['red', 'green']
        labels_map = ["negatif", "positif"]
        for lbl, c in zip([-1, 1], colors):
            plt.scatter(X_tsne_normalized[y == lbl, 0],
                        X_tsne_normalized[y == lbl, 1],
                        c=c,
                        label=labels_map[int((lbl + 1)/2)],
                        alpha=0.6)
        
        # Create grid points in normalized t-SNE space
        x_min, x_max = X_tsne_normalized[:, 0].min() - 0.5, X_tsne_normalized[:, 0].max() + 0.5
        y_min, y_max = X_tsne_normalized[:, 1].min() - 0.5, X_tsne_normalized[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        grid_points_tsne_normalized = np.c_[xx.ravel(), yy.ravel()]
        
        # Transform grid points back to original t-SNE space
        grid_points_tsne = tsne_scaler.inverse_transform(grid_points_tsne_normalized)
        

        grid_points_full = np.zeros((grid_points_tsne.shape[0], X_scaled.shape[1]))
        for i, point in enumerate(grid_points_tsne):
            # Find the nearest point in X_tsne to the grid point
            distances = np.linalg.norm(X_tsne - point, axis=1)
            nearest_idx = np.argmin(distances)
            # Use the corresponding point in the original space
            grid_points_full[i] = X_scaled[nearest_idx]
        
        # Hitung decision function menggunakan model SVM yang sudah ada
        Z = svm_model.decision_function(grid_points_full)
        
        # Reshape Z ke bentuk 2D yang sesuai dengan grid
        Z = Z.reshape(xx.shape)
        
        # Normalisasi nilai Z
        z_scaler = StandardScaler()
        Z_normalized = z_scaler.fit_transform(Z.reshape(-1, 1)).reshape(xx.shape)
        
        # Plot the decision boundary and margins
        plt.contour(xx, yy, Z_normalized, levels=[0], linewidths=2, colors='black')  # Hyperplane
        plt.contour(xx, yy, Z_normalized, levels=[-1], linewidths=1, colors='blue', linestyles='dashed')  # Margin negatif
        plt.contour(xx, yy, Z_normalized, levels=[1], linewidths=1, colors='red', linestyles='dashed')   # Margin positif
        
        plt.title(f"Visualisasi t-SNE + SVM Hyperplane ({source_name})")
        plt.xlabel("t-SNE 1 (Normalized)")
        plt.ylabel("t-SNE 2 (Normalized)")
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Adjust the layout to prevent overlap
        plt.tight_layout()
        
        # Save the plot to a file
        save_path = f"static/graph/train_svm_manual_{source_name.lower()}.png"
        plt.savefig(save_path)
        plt.close()
        
        return {
            "model_path": model_path,
            "vectorizer_path": vectorizer_path,
            "graph_path": save_path,
            "weights": svm_model.coef_.tolist(),
            "bias": svm_model.intercept_.tolist()
        }
    
    # Latih model untuk GoTube
    gotube_results = train_and_save_model(gotube_data, "GoTube")
    if "status" in gotube_results and gotube_results["status"] == "error":
        return jsonify(gotube_results), 404
    
    # Latih model untuk YouTube
    youtube_results = train_and_save_model(youtube_data, "YouTube")
    if "status" in youtube_results and youtube_results["status"] == "error":
        return jsonify(youtube_results), 404
    
    # Return response
    return jsonify({
        "status": "success",
        "message": "Visualisasi SVM dengan t-SNE selesai untuk GoTube dan YouTube.",
        "gotube_results": gotube_results,
        "youtube_results": youtube_results
    })
@app.route('/predict_svm/<app_name>', methods=['GET'])
def route_predict_svm(app_name):
    # Tentukan folder data berdasarkan app_name
    data_folder = f"static/data/{app_name}/"
    
    # Pastikan folder data ada
    if not os.path.exists(data_folder):
        return jsonify({
            "status": "error",
            "message": f"Folder data untuk aplikasi '{app_name}' tidak ditemukan."
        }), 404
    
    # Tentukan path file data uji
    data_test_file = os.path.join(data_folder, "data_uji_test.xlsx")
    
    # Pastikan file data uji ada
    if not os.path.exists(data_test_file):
        return jsonify({
            "status": "error",
            "message": f"File 'data_uji_test.xlsx' untuk aplikasi '{app_name}' tidak ditemukan."
        }), 404
    
    # Load data uji
    data_test = pd.read_excel(data_test_file)

    # Pastikan kolom "komentar" ada
    if "komentar" not in data_test.columns:
        return jsonify({
            "status": "error",
            "message": "Kolom 'komentar' tidak ditemukan dalam data uji."
        }), 400

    # Preprocessing komentar (menghapus NaN dan memproses teks)
    data_test["komentar"] = data_test["komentar"].fillna("tidak ada komentar")
    data_test["processed_komentar"] = data_test["komentar"].apply(preprocess_comment)

    # Tentukan path model dan vectorizer berdasarkan app_name
    model_path = f"static/models/svm_model_{app_name.lower()}.pkl"
    vectorizer_path = f"static/models/vectorizer_{app_name.lower()}.pkl"
    
    # Pastikan file model dan vectorizer ada
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return jsonify({
            "status": "error",
            "message": f"Model atau vectorizer untuk aplikasi '{app_name}' tidak ditemukan."
        }), 404
    
    # Load model dan vectorizer
    with open(model_path, "rb") as f_model:
        svm_model = pickle.load(f_model)
    with open(vectorizer_path, "rb") as f_vec:
        vectorizer = pickle.load(f_vec)
    
    # Vectorize data uji
    X_test_unlabeled = vectorizer.transform(data_test["processed_komentar"]).toarray()

    # Normalisasi data dengan StandardScaler untuk menghindari bias model
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test_unlabeled)

    # Gunakan decision_function untuk melihat distribusi skor prediksi
    decision_scores = svm_model.decision_function(X_test_scaled)

    # Atur ambang batas custom jika model cenderung bias
    threshold = 0  # Default SVM threshold (bisa diatur berdasarkan distribusi data latih)
    
    # Prediksi label dengan ambang batas
    data_test["prediksi_label"] = np.where(decision_scores > threshold, 1, -1)

    # Map label numerik ke string
    label_map_reverse = {1: "positif", -1: "negatif"}
    data_test["prediksi_sentimen"] = data_test["prediksi_label"].map(label_map_reverse)
    
    # Simpan hasil prediksi ke file Excel
    output_file = os.path.join(data_folder, "classification_results.xlsx")
    data_test[['komentar', 'prediksi_sentimen']].to_excel(output_file, index=False)
    
    # Hitung jumlah data berdasarkan label (positif dan negatif)
    positive_count_test = int((data_test["prediksi_label"] == 1).sum())  # Konversi ke int
    negative_count_test = int((data_test["prediksi_label"] == -1).sum())  # Konversi ke int
    
    # Return respons JSON
    return jsonify({
        "status": "success",
        "message": f"Prediksi dengan SVM selesai untuk aplikasi '{app_name}'. Hasil disimpan di '{output_file}'.",
        "data_counts": {
            "positive_test": positive_count_test,
            "negative_test": negative_count_test
        }
    })



def extract_app_id(playstore_url):
    parsed_url = urlparse(playstore_url)
    query_params = parse_qs(parsed_url.query)
    if "id" in query_params:
        return query_params["id"][0]
    raise ValueError("Invalid Play Store URL. Could not find 'id' parameter.")

def scrape_reviews_from_url(playstore_url, lang='id', country='id', num_reviews=100):
    app_id = extract_app_id(playstore_url)
    print(f"Scraping reviews for App ID: {app_id}")
    all_reviews = []
    count = 0
    while count < num_reviews:
        result, _ = reviews(app_id, lang=lang, country=country, sort=Sort.NEWEST, count=min(100, num_reviews - count))
        all_reviews.extend(result)
        count += len(result)
        if len(result) < 100:
            break
    df = pd.DataFrame(all_reviews)
    df = df[['userName', 'content', 'score']]
    df.rename(columns={'userName': 'user', 'content': 'komentar', 'score': 'rating'}, inplace=True)
    return df

def add_labels_to_reviews(df):
    def label_rating(rating):
        if rating in [1, 2]:
            return 'negatif'
        elif rating in [3, 4, 5]:
            return 'positif'
    df['label'] = df['rating'].apply(label_rating)
    df['sentimen'] = df['label']
    return df

@app.route('/get_data_latih_info', methods=['GET'])
def get_data_latih_info():
    # Path to data_latih_labeled.xlsx for both folders
    gotube_file = "static/data/gotube/data_latih_labeled.xlsx"
    youtube_file = "static/data/youtube/data_latih_labeled.xlsx"

    # Check if files exist
    if not os.path.exists(gotube_file) or not os.path.exists(youtube_file):
        return jsonify({
            "status": "error",
            "message": "File data_latih_labeled.xlsx tidak ditemukan di salah satu folder."
        }), 404

    # Load data from both files
    gotube_data = pd.read_excel(gotube_file)
    youtube_data = pd.read_excel(youtube_file)

    # Calculate sentiment counts for each dataset
    gotube_counts = gotube_data['sentimen'].value_counts().to_dict()
    youtube_counts = youtube_data['sentimen'].value_counts().to_dict()

    # Prepare response
    response = {
        "status": "success",
        "gotube_counts": {
            "positive": gotube_counts.get('positif', 0),
            "negative": gotube_counts.get('negatif', 0)
        },
        "youtube_counts": {
            "positive": youtube_counts.get('positif', 0),
            "negative": youtube_counts.get('negatif', 0)
        }
    }

    return jsonify(response)
@app.route('/scrape_reviews', methods=['POST'])
def route_scrape_reviews():
    data = request.get_json()
    if not data or 'playstore_url' not in data:
        return jsonify({
            "status": "error",
            "message": "Harap sertakan 'playstore_url' dalam body request JSON."
        }), 400
    playstore_url = data['playstore_url']
    lang = data.get('lang', 'id')
    country = data.get('country', 'id')
    num_reviews = data.get('num_reviews', 1000)
    data_type = data.get('data_type', 'latih')
    data_latih_file = "data_latih.xlsx"
    data_latih_labeled_file = "data_latih_labeled.xlsx"
    data_uji_file = "data_uji_test.xlsx"
    if data_type == 'latih':
        if os.path.exists(data_latih_file):
            df = pd.read_excel(data_latih_file)
            df = add_labels_to_reviews(df)
            df.to_excel(data_latih_labeled_file, index=False)
            msg = (f"File '{data_latih_file}' ditemukan. Menambahkan label pada data yang ada dan menyimpannya di '{data_latih_labeled_file}'.")
        else:
            df = scrape_reviews_from_url(playstore_url, lang=lang, country=country, num_reviews=num_reviews)
            df = add_labels_to_reviews(df)
            df.to_excel(data_latih_file, index=False)
            df.to_excel(data_latih_labeled_file, index=False)
            msg = (f"File '{data_latih_file}' belum ada. Melakukan scraping. Hasil disimpan di '{data_latih_file}' dan '{data_latih_labeled_file}'.")
    else:
        if os.path.exists(data_uji_file):
            df = pd.read_excel(data_uji_file)
        else:
            df = scrape_reviews_from_url(playstore_url, lang=lang, country=country, num_reviews=num_reviews)
            df['rating'] = df['rating']
            df.to_excel(data_uji_file, index=False)
            msg = (f"File '{data_uji_file}' belum ada. Melakukan scraping. Hasil disimpan di '{data_uji_file}'.")
    return jsonify({
        "status": "success",
        "message": msg,
        "data_shape": df.shape,
        "head_data": df.to_dict(orient='records')
    })

@app.route('/admin_scraping')
def admin_scraping():
    return render_template('admin_scraping.html')

@app.route('/data_komentar')
def komentar():
    return render_template('admin_data_komentar.html')

@app.route('/data_latih')
def data_latih():
    return render_template('admin_data_latih.html')

@app.route('/delete_data_latih', methods=['GET'])
def delete_data_latih():
    file_path = 'data_latih_labeled.xlsx'
    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({
            "status": "success",
            "message": f"File '{file_path}' has been deleted."
        }), 200
    else:
        return jsonify({
            "status": "error",
            "message": f"File '{file_path}' not found."
        }), 404

@app.route('/delete_data_uji', methods=['GET'])
def delete_data_uji():
    file_path = 'data_uji_test.xlsx'
    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({
            "status": "success",
            "message": f"File '{file_path}' has been deleted."
        }), 200
    else:
        return jsonify({
            "status": "error",
            "message": f"File '{file_path}' not found."
        }), 404

@app.route('/open_data_latih', methods=['GET'])
def open_data_latih():
    file_path = 'data_latih_labeled.xlsx'
    if os.path.exists(file_path):
        try:
            subprocess.Popen(['start', file_path], shell=True)
            return jsonify({
                "status": "success",
                "message": f"Opening file '{file_path}'."
            }), 200
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Failed to open '{file_path}': {str(e)}"
            }), 500
    else:
        return jsonify({
            "status": "error",
            "message": f"File '{file_path}' not found."
        }), 404

@app.route('/open_data_uji', methods=['GET'])
def open_data_uji():
    file_path = 'data_uji_test.xlsx'
    if os.path.exists(file_path):
        try:
            subprocess.Popen(['start', file_path], shell=True)
            return jsonify({
                "status": "success",
                "message": f"Opening file '{file_path}'."
            }), 200
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Failed to open '{file_path}': {str(e)}"
            }), 500
    else:
        return jsonify({
            "status": "error",
            "message": f"File '{file_path}' not found."
        }), 404
@app.route('/tfidf_results', methods=['GET'])
def tfidf_results():
    return render_template('admin_tf_idf.html')

@app.route('/', methods=['GET'])
def index():
    return render_template('admin_index.html')

@app.route('/result')
def result():
    return render_template('admin_result.html')

if __name__ == '__main__':
    os.makedirs('static/graph', exist_ok=True)
    app.run(debug=True, port=5000)