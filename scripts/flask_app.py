"""
Flask Application for Milestone 5.

This script provides a REST API endpoint for MNIST digit prediction:
- Accepts POST requests with base64-encoded images
- Decodes images to numpy arrays
- Runs prediction using the trained neural network
- Saves image and prediction to the database
- Returns the prediction to the client

Additionally, this version includes a front-end upload page at /
with a form to upload images for prediction.
"""
import time
import base64
import io
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for
from PIL import Image
from werkzeug.utils import secure_filename
import requests

from src.model_io import load_model
from src.db_helper import (
    create_database,
    create_tables,
    get_connection,
    insert_input_data,
    insert_prediction
)

app = Flask(__name__)

# Global variables for model and database connection
model = None
db_conn = None

# Allowed file types for upload
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def wait_for_db(max_retries: int = 30, retry_delay: int = 2):
    """Wait for the database to be ready."""
    import psycopg2
    conn_params = {
        'host': 'db',
        'user': 'postgres',
        'password': 'postgres'
    }

    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(
                host=conn_params['host'],
                user=conn_params['user'],
                password=conn_params['password'],
                dbname='postgres'
            )
            conn.close()
            print("Database is ready!")
            return True
        except Exception:
            print(f"Waiting for database... (attempt {attempt + 1}/{max_retries})")
            time.sleep(retry_delay)

    raise Exception("Database did not become ready in time")


def initialize_app():
    """Initialize the Flask application with model and database."""
    global model, db_conn

    print("=" * 60)
    print("MILESTONE 5 - Flask REST API Application")
    print("=" * 60)

    # Wait for database to be ready
    print("\n[Step 1/4] Waiting for database to be ready...")
    wait_for_db()

    # Initialize database
    print("\n[Step 2/4] Initializing database...")
    conn_params = {
        'host': 'db',
        'user': 'postgres',
        'password': 'postgres'
    }
    create_database(conn_params, 'milestone_5')

    # Create tables
    print("\n[Step 3/4] Creating tables...")
    db_conn = get_connection(
        host='db',
        database='milestone_5',
        user='postgres',
        password='postgres'
    )
    create_tables(db_conn)

    # Load trained model
    print("\n[Step 4/4] Loading trained neural network model...")
    try:
        model = load_model('/app/models/mnist_model.keras')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    print("\n" + "=" * 60)
    print("Flask REST API ready!")
    print("Endpoint: POST /predict")
    print("=" * 60)


def decode_base64_image(base64_string: str) -> np.ndarray:
    """
    Decode a base64-encoded image to a numpy array.
    Numpy array of shape (28, 28, 1) normalized to [0, 1]
    """
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]

    image_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_bytes))

    if image.mode != 'L':
        image = image.convert('L')

    if image.size != (28, 28):
        image = image.resize((28, 28), Image.Resampling.LANCZOS)

    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0
    image_array = image_array.reshape(28, 28, 1)

    return image_array


@app.route('/predict', methods=['POST'])
def predict():
    """REST endpoint for MNIST digit prediction."""
    global model, db_conn

    try:
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({'error': 'Missing "image" field in request body'}), 400

        try:
            image_array = decode_base64_image(data['image'])
        except Exception as e:
            return jsonify({'error': f'Failed to decode image: {str(e)}'}), 400

        true_label = data.get('true_label', -1)
        input_data_id = insert_input_data(db_conn, image_array, int(true_label))
        prediction_input = image_array.reshape(1, 28, 28, 1)
        prediction_probs = model.predict(prediction_input, verbose=0)
        predicted_label = int(prediction_probs[0].argmax())
        confidence = float(prediction_probs[0].max())
        prediction_id = insert_prediction(
            db_conn,
            input_data_id=input_data_id,
            predicted_label=predicted_label,
            confidence=confidence,
            prediction_probabilities=prediction_probs[0]
        )

        response = {
            'prediction': predicted_label,
            'confidence': confidence,
            'probabilities': prediction_probs[0].tolist(),
            'input_data_id': input_data_id,
            'prediction_id': prediction_id
        }

        print(f"Prediction: {predicted_label} (confidence: {confidence:.4f})")
        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'}), 200


# --- FRONT-END ROUTES ---

@app.route("/", methods=["GET"])
def index():
    """Render the upload page."""
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle uploaded images from the front-end."""
    if "image" not in request.files:
        return redirect(request.url)

    file = request.files["image"]

    if file.filename == "" or not allowed_file(file.filename):
        return redirect(request.url)

    # Preprocess image
    image = Image.open(file).convert("L").resize((28, 28))
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = image_array.reshape(28, 28, 1)

    # Convert image to base64 for /predict
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Call /predict internally
    response = requests.post("http://localhost:5000/predict", json={"image": img_str})
    if response.status_code == 200:
        data = response.json()
        prediction = data["prediction"]
        confidence = data.get("confidence")
    else:
        prediction = "Error"
        confidence = None

    image_data = img_str

    return render_template(
        "upload.html",
        prediction=prediction,
        confidence=confidence,
        image_data=image_data
    )


if __name__ == '__main__':
    initialize_app()
    app.run(host='0.0.0.0', port=5000, debug=False)
