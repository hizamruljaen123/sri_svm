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
    filename = "data_latih_labeled.xlsx"
    if not os.path.exists(filename):
        return jsonify({
            "status": "error",
            "message": f"File {filename} tidak ditemukan."
        }), 404
    df = pd.read_excel(filename)
    if "komentar" in df.columns:
        df["processed_komentar"] = df["komentar"].apply(preprocess_comment)
    return jsonify(df.to_dict(orient='records'))

@app.route('/evaluate_model', methods=['GET'])
def evaluate_model_route():
    # Load training and test data
    data_train = pd.read_excel("data_latih_labeled.xlsx")
    data_test = pd.read_excel("data_uji_test.xlsx")
    
    # Preprocessing the training and test data
    X_train = data_train["komentar"].apply(preprocess_comment)
    X_test = data_test["komentar"].apply(preprocess_comment)
    
    # Load the vectorizer used during training
    with open("vectorizer.pkl", "rb") as f_vec:
        vectorizer = pickle.load(f_vec)
    
    # Vectorize the training and test data using the same vectorizer
    X_train_vec = vectorizer.transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()
    
    # Labels for training data
    y_train = data_train["label"].map({"positif": 1, "negatif": -1})
    y_train = y_train[y_train != 0]  # Filter out neutral labels
    
    # For test data, map ratings to sentiment labels (negatif, positif only)
    def map_rating_to_sentiment(rating):
        if rating in [1, 2]:
            return "negatif"
        elif rating in [3, 4, 5]:
            return "positif"
    
    data_test["sentimen"] = data_test["rating"].apply(map_rating_to_sentiment)
    y_test = data_test["sentimen"].map({"positif": 1, "negatif": -1})
    y_test = y_test[y_test != 0]  # Filter out neutral labels
    
    # Train SVM manually
    weights, bias = train_svm_manual(X_train_vec, y_train.to_numpy())
    
    # Predict using manual SVM
    y_train_pred = predict_svm_manual(X_train_vec, weights, bias)
    y_test_pred = predict_svm_manual(X_test_vec, weights, bias)
    
    # Calculate evaluation metrics manually
    accuracy_train = np.mean(y_train == y_train_pred)
    precision_train = np.sum((y_train == 1) & (y_train_pred == 1)) / np.sum(y_train_pred == 1)
    recall_train = np.sum((y_train == 1) & (y_train_pred == 1)) / np.sum(y_train == 1)
    f1_train = 2 * (precision_train * recall_train) / (precision_train + recall_train)
    
    accuracy_test = np.mean(y_test == y_test_pred)
    precision_test = np.sum((y_test == 1) & (y_test_pred == 1)) / np.sum(y_test_pred == 1)
    recall_test = np.sum((y_test == 1) & (y_test_pred == 1)) / np.sum(y_test == 1)
    f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test)
    
    # Confusion matrix
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    
    # Plotting confusion matrix for training data
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'])
    plt.title('Confusion Matrix - Training Data')
    plt.xlabel('Predictions')
    plt.ylabel('Actual')
    cm_train_img_path = 'static/graph/confusion_matrix_train.png'
    plt.savefig(cm_train_img_path)
    plt.close()
    
    # Plotting confusion matrix for testing data
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'])
    plt.title('Confusion Matrix - Testing Data')
    plt.xlabel('Predictions')
    plt.ylabel('Actual')
    cm_test_img_path = 'static/graph/confusion_matrix_test.png'
    plt.savefig(cm_test_img_path)
    plt.close()
    
    # Return evaluation results and confusion matrix image paths
    return jsonify({
        "status": "success",
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
    })
@app.route('/show_data_uji', methods=['GET'])
def show_data_uji():
    filename = "data_uji_test.xlsx"
    if not os.path.exists(filename):
        return jsonify({
            "status": "error",
            "message": f"File {filename} tidak ditemukan."
        }), 404
    df = pd.read_excel(filename)
    if "komentar" in df.columns:
        df["processed_komentar"] = df["komentar"].apply(preprocess_comment)
    return jsonify(df.to_dict(orient='records'))

@app.route('/show_classification_results', methods=['GET'])
def show_classification_results():
    filename = "classification_results.xlsx"
    if not os.path.exists(filename):
        return jsonify({
            "status": 404,
            "message": f"File {filename} tidak ditemukan."
        }), 404
    df = pd.read_excel(filename)
    results = []
    for index, row in df.iterrows():
        prediksi_sentimen = row["prediksi_sentimen"] if pd.notna(row["prediksi_sentimen"]) else '-'
        results.append({
            "komentar": row["komentar"],
            "prediksi_sentimen": prediksi_sentimen
        })
    return jsonify({
        "status": 200,
        "data_result": results
    })
@app.route('/show_tfidf', methods=['GET'])
def show_tfidf():
    """
    Endpoint untuk menampilkan representasi TF-IDF dari data latih dan data uji.
    Menghasilkan file .txt untuk data latih dan data uji dengan format:
    Dokumen [index]:
    - Kata1: Nilai_TFIDF
    - Kata2: Nilai_TFIDF
    ...
    """
    # Load training and test data
    data_train_file = "data_latih_labeled.xlsx"
    data_test_file = "data_uji_test.xlsx"
    
    if not os.path.exists(data_train_file) or not os.path.exists(data_test_file):
        return jsonify({
            "status": "error",
            "message": "File data_latih_labeled.xlsx atau data_uji_test.xlsx tidak ditemukan."
        }), 404
    
    # Read the Excel files into DataFrames
    data_train = pd.read_excel(data_train_file)
    data_test = pd.read_excel(data_test_file)
    
    # Preprocess the comments
    data_train["processed_komentar"] = data_train["komentar"].apply(preprocess_comment)
    data_test["processed_komentar"] = data_test["komentar"].apply(preprocess_comment)
    
    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit the vectorizer on the training data and transform both datasets
    X_train_tfidf = vectorizer.fit_transform(data_train["processed_komentar"])
    X_test_tfidf = vectorizer.transform(data_test["processed_komentar"])
    
    # Get feature names (words) from the vectorizer
    feature_names = vectorizer.get_feature_names_out()
    
    # Convert sparse matrices to dense arrays
    X_train_tfidf_dense = X_train_tfidf.toarray()
    X_test_tfidf_dense = X_test_tfidf.toarray()
    
    # Save TF-IDF results for training data to a text file
    with open("static/tfidf_train.txt", "w", encoding="utf-8") as f_train:
        for i, row in enumerate(X_train_tfidf_dense):
            non_zero_indices = np.where(row > 0.0)[0]
            f_train.write(f"Dokumen {i}:\n")
            for idx in non_zero_indices:
                f_train.write(f"- {feature_names[idx]}: {row[idx]:.4f}\n")
            f_train.write("\n")
    
    # Save TF-IDF results for test data to a text file
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
    data_train = pd.read_excel("data_latih_labeled.xlsx")
    data_train["processed_komentar"] = data_train["komentar"].apply(preprocess_comment)
    sentimen_map = {"positif": 1, "negatif": -1}
    data_train["label"] = data_train["sentimen"].map(sentimen_map)
    filtered_data = data_train[data_train["label"] != 0].copy()
    X = filtered_data["processed_komentar"]
    y = filtered_data["label"]
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X).toarray()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_vec)
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_scaled, y)
    plt.figure(figsize=(10, 6))
    colors = ['red', 'green']
    labels_map = ["negatif", "positif"]
    for lbl, c in zip([-1, 1], colors):
        plt.scatter(X_tsne[y == lbl, 0], X_tsne[y == lbl, 1], c=c, label=labels_map[int((lbl + 1)/2)], alpha=0.6)
    xx, yy = np.meshgrid(np.linspace(X_tsne[:, 0].min(), X_tsne[:, 0].max(), 200), np.linspace(X_tsne[:, 1].min(), X_tsne[:, 1].max(), 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = svm_model.decision_function(scaler.transform(vectorizer.transform([f"{x} {y}" for x, y in grid_points]).toarray()))
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
    plt.contour(xx, yy, Z, levels=[-1], linewidths=1, colors='blue', linestyles='dashed')
    plt.contour(xx, yy, Z, levels=[1], linewidths=1, colors='red', linestyles='dashed')
    plt.title("Visualisasi t-SNE + SVM Decision Boundary")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.grid(alpha=0.3)
    save_path = "static/graph/train_svm_manual.png"
    plt.savefig(save_path)
    plt.close()
    fig = plt.figure(figsize=(10, 6))
    for lbl, c in zip([-1, 1], colors):
        plt.scatter(X_tsne[y == lbl, 0], X_tsne[y == lbl, 1], c=c, label=labels_map[int((lbl + 1)/2)], alpha=0.6)
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
    plt.contour(xx, yy, Z, levels=[-1], linewidths=1, colors='blue', linestyles='dashed')
    plt.contour(xx, yy, Z, levels=[1], linewidths=1, colors='red', linestyles='dashed')
    plt.title("Visualisasi t-SNE + SVM Decision Boundary (Base64)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.grid(alpha=0.3)
    fig_base64 = fig_to_base64(fig)
    plt.close(fig)
    response = {
        "status": "success",
        "message": "Training SVM Manual selesai.",
        "graph_path": save_path,
        "graph_base64": f"data:image/png;base64,{fig_base64}",
        "weights": svm_model.coef_.tolist(),
        "bias": svm_model.intercept_.tolist()
    }
    return jsonify(response)

@app.route('/predict_svm', methods=['GET'])
def route_predict_svm():
    data_test = pd.read_excel("data_uji_test.xlsx")
    data_test["processed_komentar"] = data_test["komentar"].apply(preprocess_comment)
    def label_sentiment(rating):
        if rating in [1, 2]:
            return "negatif"
        elif rating in [3, 4, 5]:
            return "positif"
        else:
            return "neutral"
    data_test["prediksi_sentimen"] = data_test["rating"].apply(label_sentiment)
    with open("svm_model.pkl", "rb") as f_model:
        svm_model = pickle.load(f_model)
    with open("vectorizer.pkl", "rb") as f_vec:
        vectorizer = pickle.load(f_vec)
    X_test_unlabeled = vectorizer.transform(data_test["processed_komentar"]).toarray()
    data_test["prediksi_label"] = svm_model.predict(X_test_unlabeled)
    data_test["prediksi_label"] = data_test["prediksi_label"].replace(0, -1)
    label_map_reverse = {1: "positif", -1: "negatif"}
    data_test["prediksi_sentimen"] = data_test["prediksi_label"].map(label_map_reverse)
    data_test = data_test[data_test["prediksi_sentimen"].isin(["positif", "negatif"])]
    data_test[['komentar', 'prediksi_sentimen']].to_excel("classification_results.xlsx", index=False)
    return jsonify({
        "status": "success",
        "message": "Prediksi dengan SVM (pickle) selesai dan hasil disimpan di 'classification_results.xlsx'."
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

@app.route('/')
def result():
    return render_template('admin_result.html')

if __name__ == '__main__':
    os.makedirs('static/graph', exist_ok=True)
    app.run(debug=True, port=5000)