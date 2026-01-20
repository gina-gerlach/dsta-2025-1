"""
Database Application for Milestone 3.

This script demonstrates the complete workflow of:
1. Initializing the PostgreSQL database
2. Loading a trained neural network model
3. Loading a sample from the MNIST dataset
4. Saving the sample to the database (serialization)
5. Loading the sample back from the database (deserialization)
6. Making a prediction using the neural network
7. Saving the prediction with a foreign key reference to the input data
"""
import time
import numpy as np
from src.load_data import load_mnist_data
from src.model_io import load_model
from src.db_helper import (
    create_database,
    create_tables,
    get_connection,
    insert_input_data,
    get_input_data,
    insert_prediction
)
from PIL import Image


def wait_for_db(max_retries: int = 30, retry_delay: int = 2):
    """
    Wait for the database to be ready.

    Args:
        max_retries: Maximum number of connection attempts
        retry_delay: Seconds to wait between retries
    """
    conn_params = {
        'host': 'db',
        'user': 'postgres',
        'password': 'postgres'
    }

    for attempt in range(max_retries):
        try:
            # Try to connect to postgres database
            import psycopg2
            conn = psycopg2.connect(
                host=conn_params['host'],
                user=conn_params['user'],
                password=conn_params['password'],
                dbname='postgres'
            )
            conn.close()
            print("Database is ready!")
            return True
        except Exception as e:
            print(f"Waiting for database... (attempt {attempt + 1}/{max_retries})")
            time.sleep(retry_delay)

    raise Exception("Database did not become ready in time")


def display_image(image_array: np.ndarray, label: int):
    """
    Display image information and optionally save it for verification.

    Args:
        image_array: Numpy array of the image
        label: The label of the image
    """
    print(f"\nImage Information:")
    print(f"  Shape: {image_array.shape}")
    print(f"  Data type: {image_array.dtype}")
    print(f"  Min value: {image_array.min():.4f}")
    print(f"  Max value: {image_array.max():.4f}")
    print(f"  True label: {label}")

    # Convert to PIL Image for verification
    # MNIST images are normalized [0, 1], convert to [0, 255]
    img_rescaled = (image_array.squeeze() * 255).astype(np.uint8)
    img = Image.fromarray(img_rescaled, mode='L')

    # Save the image for visual verification
    img.save('/app/models/sample_image.png')
    print(f"  Saved image to /app/models/sample_image.png for verification")


def main():
    """Execute the complete database workflow."""
    print("=" * 60)
    print("MILESTONE 3 - Database Application")
    print("=" * 60)

    #  Wait for database to be ready
    print("\n[Step 1/8] Waiting for database to be ready...")
    wait_for_db()

    # Initialize database
    print("\n[Step 2/8] Initializing database...")
    conn_params = {
        'host': 'db',
        'user': 'postgres',
        'password': 'postgres'
    }
    create_database(conn_params, 'milestone_3')

    # Create tables
    print("\n[Step 3/8] Creating tables...")
    conn = get_connection(
        host='db',
        database='milestone_3',
        user='postgres',
        password='postgres'
    )
    create_tables(conn)

    # Load trained model
    print("\n[Step 4/8] Loading trained neural network model...")
    try:
        model = load_model('/app/models/mnist_model.keras')
        print("Model loaded successfully from /app/models/mnist_model.keras")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model file exists in the models volume")
        conn.close()
        return

    # Load a sample from MNIST dataset
    print("\n[Step 5/8] Loading sample from MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    # Use the first test sample
    sample_image = x_test[0]
    sample_label = y_test[0].argmax()  # Convert one-hot to class index

    print(f"Sample selected: digit '{sample_label}'")
    display_image(sample_image, sample_label)

    # Save sample to database (serialization)
    print("\n[Step 6/8] Saving sample to database (serialization)...")
    input_data_id = insert_input_data(conn, sample_image, int(sample_label))

    # Load sample from database (deserialization)
    print("\n[Step 7/8] Loading sample from database (deserialization)...")
    retrieved_image, retrieved_label = get_input_data(conn, input_data_id)

    # Verify the deserialization worked correctly
    print("\nVerifying deserialization:")
    if np.array_equal(sample_image, retrieved_image):
        print("  Image data matches perfectly!")
    else:
        print("  Warning: Image data does not match")

    if sample_label == retrieved_label:
        print("  Label matches perfectly!")
    else:
        print("  Warning: Label does not match")

    # Display the retrieved image
    display_image(retrieved_image, retrieved_label)

    # Make prediction
    print("\n[Step 8/8] Making prediction with neural network...")

    # Reshape for prediction
    prediction_input = retrieved_image.reshape(1, 28, 28, 1)
    prediction_probs = model.predict(prediction_input, verbose=0)

    predicted_label = int(prediction_probs[0].argmax())
    confidence = float(prediction_probs[0].max())

    print(f"\nPrediction Results:")
    print(f"  Predicted digit: {predicted_label}")
    print(f"  Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    print(f"  True label: {retrieved_label}")

    # Step 9: Save prediction to database with foreign key
    print("\nSaving prediction to database...")
    prediction_id = insert_prediction(
        conn,
        input_data_id=input_data_id,
        predicted_label=predicted_label,
        confidence=confidence,
        prediction_probabilities=prediction_probs[0]
    )

    print(f"\n" + "=" * 60)
    print("DATABASE SUMMARY")
    print("=" * 60)
    print(f"Database 'milestone_3' initialized")
    print(f"Tables 'input_data' and 'predictions' created")
    print(f"Sample image saved with ID: {input_data_id}")
    print(f"Prediction saved with ID: {prediction_id}")
    print(f"Foreign key link: predictions.input_data_id = {input_data_id}")
    print("=" * 60)

    # Close connection
    conn.close()
    print("\nDatabase connection closed. Application completed successfully")


if __name__ == "__main__":
    main()
