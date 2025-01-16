import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Agar tidak muncul GUI backend error di server
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

app = Flask(__name__)
CORS(app)
stop_words = set(stopwords.words('indonesian'))

def preprocess_comment(comment: str) -> str:
    """
    Melakukan tokenisasi, lower-casing, menghilangkan kata non-alfanumerik 
    dan stopwords pada komentar.
    """
    tokens = word_tokenize(comment.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(tokens)

def train_svm_manual(X, y, learning_rate=0.001, epochs=1000, lambda_param=0.01):
    """
    Pelatihan SVM secara manual (tanpa library SVM).
    X : numpy array (fitur)
    y : numpy array (label), misal -1 atau +1.
    """
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for _ in range(epochs):
        for idx, x_i in enumerate(X):
            condition = y[idx] * (np.dot(x_i, weights) - bias) >= 1
            if condition:
                # Update weight hanya regularisasi
                weights -= learning_rate * (2 * lambda_param * weights)
            else:
                # Update weight + regularisasi + penalty
                weights -= learning_rate * (2 * lambda_param * weights - np.dot(x_i, y[idx]))
                bias -= learning_rate * y[idx]
    return weights, bias

def predict_svm_manual(X, weights, bias):
    """
    Prediksi label dengan model SVM manual.
    """
    linear_output = np.dot(X, weights) - bias
    return np.sign(linear_output)

def fig_to_base64(fig):
    """
    Mengubah figure matplotlib menjadi string base64
    sehingga bisa dikirim langsung dalam JSON.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    return encoded


# ---------------------
# ROUTE 1: Menampilkan Data Latih (JSON)
# ---------------------
@app.route('/show_data_latih', methods=['GET'])
def show_data_latih():
    """
    Mengembalikan data latih (data_latih_labeled.xlsx) dalam bentuk JSON.
    """
    filename = "data_latih_labeled.xlsx"  # Sesuaikan nama file
    if not os.path.exists(filename):
        return jsonify({
            "status": "error",
            "message": f"File {filename} tidak ditemukan."
        }), 404

    df = pd.read_excel(filename)
    # Contoh, jika ingin menambahkan kolom processed:
    if "komentar" in df.columns:
        df["processed_komentar"] = df["komentar"].apply(preprocess_comment)
    return jsonify(df.to_dict(orient='records'))


# ---------------------
# ROUTE 2: Menampilkan Data Uji (JSON)
# ---------------------
@app.route('/show_data_uji', methods=['GET'])
def show_data_uji():
    """
    Mengembalikan data uji (data_uji_test.xlsx) dalam bentuk JSON.
    """
    filename = "data_uji_test.xlsx"  # Sesuaikan nama file
    if not os.path.exists(filename):
        return jsonify({
            "status": "error",
            "message": f"File {filename} tidak ditemukan."
        }), 404

    df = pd.read_excel(filename)
    # Contoh, jika ingin menambahkan kolom processed:
    if "komentar" in df.columns:
        df["processed_komentar"] = df["komentar"].apply(preprocess_comment)
    return jsonify(df.to_dict(orient='records'))


# ---------------------
# ROUTE 3: Menampilkan Hasil Prediksi (JSON)
# ---------------------
@app.route('/show_classification_results', methods=['GET'])
def show_classification_results():
    filename = "classification_results.xlsx"  # Sesuaikan nama file
    if not os.path.exists(filename):
        return jsonify({
            "status": 404,
            "message": f"File {filename} tidak ditemukan."
        }), 404

    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(filename)

    # Manually convert the DataFrame to a list of dictionaries (manual JSON conversion)
    results = []
    for index, row in df.iterrows():
        # Check if 'prediksi_sentimen' is NaN and replace with '-'
        prediksi_sentimen = row["prediksi_sentimen"] if pd.notna(row["prediksi_sentimen"]) else '-'
        
        results.append({
            "komentar": row["komentar"],  # Assuming the column name is 'komentar'
            "prediksi_sentimen": prediksi_sentimen  # Replace NaN with '-'
        })

    # Return the response with status 200 and the data under the 'data_result' key
    return jsonify({
        "status": 200,
        "data_result": results
    })

# ---------------------
# ROUTE 4: Training SVM Manual
# ---------------------
@app.route('/train_svm_manual', methods=['GET'])
def route_train_svm_manual():
    """
    Endpoint untuk melakukan training dengan SVM manual.
    Membaca data dari data_latih_labeled.xlsx, melakukan preprocess, 
    training SVM manual, dan menampilkan visualisasi t-SNE + hyperplane.
    Hasil plot disimpan ke static/graph/train_svm_manual.png
    Hasil juga dapat dikembalikan dalam bentuk base64 via JSON.
    """
    # Load the labeled training data
    data_train = pd.read_excel("data_latih_labeled.xlsx")

    # Preprocessing
    data_train["processed_komentar"] = data_train["komentar"].apply(preprocess_comment)

    # Map sentiment labels to numeric values (positif = 1, negatif = -1)
    sentimen_map = {"positif": 1, "negatif": -1, "netral": 0}
    data_train["label"] = data_train["sentimen"].map(sentimen_map)

    # Filter out neutral sentiments for training (as SVM is typically binary)
    filtered_data = data_train[data_train["label"] != 0].copy()
    X = filtered_data["processed_komentar"]
    y = filtered_data["label"]

    # Vectorization (convert text to numerical vectors)
    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X).toarray()

    # Normalize the feature vectors before applying t-SNE
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_vec)

    # Apply t-SNE to reduce to 2D (only for visualization)
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    # Train SVM model
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_scaled, y)

    # Visualize the results using t-SNE
    plt.figure(figsize=(10, 6))
    colors = ['red', 'green']
    labels_map = ["negatif", "positif"]
    for lbl, c in zip([-1, 1], colors):
        plt.scatter(X_tsne[y == lbl, 0],
                    X_tsne[y == lbl, 1],
                    c=c,
                    label=labels_map[int((lbl + 1)/2)],
                    alpha=0.6)

    # Plot the SVM decision boundary and margins in the 2D t-SNE space
    xx, yy = np.meshgrid(np.linspace(X_tsne[:, 0].min(), X_tsne[:, 0].max(), 200),
                         np.linspace(X_tsne[:, 1].min(), X_tsne[:, 1].max(), 200))

    # Compute decision function in original feature space
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = svm_model.decision_function(scaler.transform(vectorizer.transform([f"{x} {y}" for x, y in grid_points]).toarray()))
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

    # Plot the margins
    plt.contour(xx, yy, Z, levels=[-1], linewidths=1, colors='blue', linestyles='dashed')
    plt.contour(xx, yy, Z, levels=[1], linewidths=1, colors='red', linestyles='dashed')

    plt.title("Visualisasi t-SNE + SVM Decision Boundary")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.grid(alpha=0.3)

    # Save the plot to a file
    save_path = "static/graph/train_svm_manual.png"
    plt.savefig(save_path)
    plt.close()

    # Create figure for base64 encoding
    fig = plt.figure(figsize=(10, 6))
    for lbl, c in zip([-1, 1], colors):
        plt.scatter(X_tsne[y == lbl, 0],
                    X_tsne[y == lbl, 1],
                    c=c,
                    label=labels_map[int((lbl + 1)/2)],
                    alpha=0.6)
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


# ---------------------
# ROUTE 5: Prediksi dengan Model (Pickle) + Visualisasi t-SNE
# ---------------------
@app.route('/predict_svm', methods=['GET'])
def route_predict_svm():
    """
    Endpoint untuk melakukan prediksi pada data uji (data_uji_test.xlsx) 
    menggunakan model SVM yang sudah disimpan (svm_model.pkl & vectorizer.pkl).
    Hanya menampilkan hasil analisis sentimen (positif atau negatif).
    """
    # Load the data from the uploaded file
    data_test = pd.read_excel("data_uji_test.xlsx")  # Data uji tanpa label
    data_test["processed_komentar"] = data_test["komentar"].apply(preprocess_comment)

    # Mapping rating to sentiments
    def label_sentiment(rating):
        if rating in [1, 2]:
            return "negatif"
        elif rating == 3:
            return "netral"
        elif rating in [4, 5]:
            return "positif"

    data_test["prediksi_sentimen"] = data_test["rating"].apply(label_sentiment)

    # Load model & vectorizer
    with open("svm_model.pkl", "rb") as f_model:
        svm_model = pickle.load(f_model)
    with open("vectorizer.pkl", "rb") as f_vec:
        vectorizer = pickle.load(f_vec)

    # Transform data uji & prediksi
    X_test_unlabeled = vectorizer.transform(data_test["processed_komentar"])
    data_test["prediksi_label"] = svm_model.predict(X_test_unlabeled)
    data_test["prediksi_label"] = data_test["prediksi_label"].replace(0, -1)

    # Map label numerik ke string
    label_map_reverse = {1: "positif", -1: "negatif"}
    data_test["prediksi_sentimen"] = data_test["prediksi_label"].map(label_map_reverse)

    # Create summary
    sentiment_counts = data_test["prediksi_sentimen"].value_counts().to_dict()
    predictions_data = data_test[["komentar", "prediksi_sentimen"]].to_dict(orient='records')

    response = {
        "status": "success",
        "message": "Prediksi dengan SVM (pickle) selesai.",
        "sentiment_distribution": sentiment_counts,
        "predictions": predictions_data
    }
    return jsonify(response)

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
        result, _ = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=Sort.NEWEST,  # Correctly use Sort enum for sorting
            count=min(100, num_reviews - count)
        )
        all_reviews.extend(result)
        count += len(result)
        if len(result) < 100:
            break

    # Convert to DataFrame
    df = pd.DataFrame(all_reviews)
    df = df[['userName', 'content', 'score']]
    df.rename(columns={'userName': 'user', 'content': 'komentar', 'score': 'rating'}, inplace=True)

    return df

def add_labels_to_reviews(df):
    # Add labeling based on rating
    def label_rating(rating):
        if rating in [1, 2]:
            return 'negatif'
        elif rating == 3:
            return 'netral'
        elif rating in [4, 5]:
            return 'positif'

    df['label'] = df['rating'].apply(label_rating)
    df['sentimen'] = df['label']  # Add 'sentimen' column identical to 'label'
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
    data_type = data.get('data_type', 'latih')  # Determine if it's 'latih' or 'uji'

    # Define the file names
    data_latih_file = "data_latih.xlsx"
    data_latih_labeled_file = "data_latih_labeled.xlsx"
    data_uji_file = "data_uji_test.xlsx"

    if data_type == 'latih':  # If it's training data, add labels
        if os.path.exists(data_latih_file):
            df = pd.read_excel(data_latih_file)
            df = add_labels_to_reviews(df)  # Function that labels the reviews
            df.to_excel(data_latih_labeled_file, index=False)
            msg = (f"File '{data_latih_file}' ditemukan. "
                   f"Menambahkan label pada data yang ada dan menyimpannya di '{data_latih_labeled_file}'.")
        else:
            df = scrape_reviews_from_url(playstore_url, lang=lang, country=country, num_reviews=num_reviews)
            df = add_labels_to_reviews(df)  # Add sentiment labels to reviews
            df.to_excel(data_latih_file, index=False)
            df.to_excel(data_latih_labeled_file, index=False)
            msg = (f"File '{data_latih_file}' belum ada. Melakukan scraping. "
                   f"Hasil disimpan di '{data_latih_file}' dan '{data_latih_labeled_file}'.")
    else:  # If it's test data, only save ratings
        if os.path.exists(data_uji_file):
            df = pd.read_excel(data_uji_file)
        else:
            df = scrape_reviews_from_url(playstore_url, lang=lang, country=country, num_reviews=num_reviews)
        df['rating'] = df['rating']  # Ensure that only ratings are stored
        df.to_excel(data_uji_file, index=False)
        msg = (f"File '{data_uji_file}' belum ada. Melakukan scraping. "
               f"Hasil disimpan di '{data_uji_file}'.")
    
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
    # Path to the 'data_latih_labeled.xlsx' file
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
    # Path to the 'data_uji_test.xlsx' file
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
# Function to open the data_latih_labeled.xlsx file
@app.route('/open_data_latih', methods=['GET'])
def open_data_latih():
    file_path = 'data_latih_labeled.xlsx'
    
    if os.path.exists(file_path):
        try:
            # Use subprocess to open the Excel file with the default application
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

# Function to open the data_uji_test.xlsx file
@app.route('/open_data_uji', methods=['GET'])
def open_data_uji():
    file_path = 'data_uji_test.xlsx'
    
    if os.path.exists(file_path):
        try:
            # Use subprocess to open the Excel file with the default application
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
@app.route('/')
def result():
    return render_template('admin_result.html')
# Jalankan Flask
if __name__ == '__main__':
    # Pastikan folder static/graph ada
    os.makedirs('static/graph', exist_ok=True)
    app.run(debug=True, port=5000)
