"""
Database helper module for storing and retrieving images in PostgreSQL.
"""
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import io
from typing import Tuple, Optional, Dict, Any


def serialize_image(image_array: np.ndarray) -> bytes:
    """
    Serialize a numpy array (image) to bytes for database storage.

    This solves the "impedance mismatch" problem by converting
    the numpy array representation to a format PostgreSQL can store.

    Args:
        image_array: Numpy array representing the image

    Returns:
        bytes: Serialized image data
    """
    # Use numpy's built-in serialization to bytes
    buffer = io.BytesIO()
    np.save(buffer, image_array)
    return buffer.getvalue()


def deserialize_image(image_bytes: bytes) -> np.ndarray:
    """
    Deserialize bytes from database back to numpy array.

    This is the "reverse transformation" that converts database
    representation back to Python/numpy representation.

    Args:
        image_bytes: Serialized image data from database

    Returns:
        np.ndarray: Reconstructed image array
    """
    buffer = io.BytesIO(image_bytes)
    return np.load(buffer, allow_pickle=False)


def create_database(conn_params: Dict[str, str], db_name: str = "milestone_3") -> None:
    """
    Create the milestone_3 database if it doesn't exist.

    Args:
        conn_params: Connection parameters (host, user, password)
        db_name: Name of the database to create
    """
    # Connect to default postgres database
    conn = psycopg2.connect(
        host=conn_params['host'],
        user=conn_params['user'],
        password=conn_params['password'],
        dbname='postgres'
    )
    conn.autocommit = True
    cursor = conn.cursor()

    # Check if database exists
    cursor.execute(
        "SELECT 1 FROM pg_database WHERE datname = %s",
        (db_name,)
    )
    exists = cursor.fetchone()

    if not exists:
        cursor.execute(f'CREATE DATABASE {db_name}')
        print(f"Database '{db_name}' created successfully")
    else:
        print(f"Database '{db_name}' already exists")

    cursor.close()
    conn.close()


def create_tables(conn) -> None:
    """
    Create the input_data and predictions tables if they don't exist.

    Database Schema:
    ----------------
    input_data table:
        - id (SERIAL PRIMARY KEY): Auto-incrementing unique identifier
        - image_data (BYTEA): Serialized numpy array of the image
        - true_label (INTEGER): The actual digit (0-9) in the image
        - image_shape (VARCHAR): Shape of the image for validation
        - created_at (TIMESTAMP): When the data was inserted

    predictions table:
        - id (SERIAL PRIMARY KEY): Auto-incrementing unique identifier
        - input_data_id (INTEGER FOREIGN KEY): References input_data(id)
        - predicted_label (INTEGER): The predicted digit (0-9)
        - confidence (REAL): Confidence score of the prediction
        - prediction_probabilities (BYTEA): Full probability distribution (serialized)
        - created_at (TIMESTAMP): When the prediction was made

    Args:
        conn: Database connection object
    """
    cursor = conn.cursor()

    # Create input_data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS input_data (
            id SERIAL PRIMARY KEY,
            image_data BYTEA NOT NULL,
            true_label INTEGER NOT NULL,
            image_shape VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create predictions table with foreign key to input_data
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            input_data_id INTEGER NOT NULL REFERENCES input_data(id) ON DELETE CASCADE,
            predicted_label INTEGER NOT NULL,
            confidence REAL NOT NULL,
            prediction_probabilities BYTEA NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create indexes for faster querying
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_input_data_label
        ON input_data(true_label)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_predictions_label
        ON predictions(predicted_label)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_predictions_input
        ON predictions(input_data_id)
    """)

    conn.commit()
    cursor.close()
    print("Tables created successfully")


def insert_input_data(
    conn,
    image_array: np.ndarray,
    true_label: int
) -> int:
    """
    Insert image data into the input_data table.

    Args:
        conn: Database connection
        image_array: Numpy array of the image
        true_label: The true label of the image (0-9 for MNIST)

    Returns:
        int: The ID of the inserted row
    """
    cursor = conn.cursor()

    # Serialize the image
    image_bytes = serialize_image(image_array)
    image_shape = str(image_array.shape)

    # Insert into database
    cursor.execute("""
        INSERT INTO input_data (image_data, true_label, image_shape)
        VALUES (%s, %s, %s)
        RETURNING id
    """, (image_bytes, true_label, image_shape))

    row_id = cursor.fetchone()[0]
    conn.commit()
    cursor.close()

    print(f"Inserted image data with ID: {row_id}")
    return row_id


def get_input_data(conn, data_id: int) -> Tuple[np.ndarray, int]:
    """
    Retrieve and deserialize image data from the database.

    Args:
        conn: Database connection
        data_id: ID of the data to retrieve

    Returns:
        tuple: (image_array, true_label)
    """
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    cursor.execute("""
        SELECT image_data, true_label, image_shape
        FROM input_data
        WHERE id = %s
    """, (data_id,))

    row = cursor.fetchone()
    cursor.close()

    if not row:
        raise ValueError(f"No data found with ID: {data_id}")

    # Deserialize the image
    image_array = deserialize_image(row['image_data'])

    print(f"Retrieved image with shape: {row['image_shape']}")
    return image_array, row['true_label']


def insert_prediction(
    conn,
    input_data_id: int,
    predicted_label: int,
    confidence: float,
    prediction_probabilities: np.ndarray
) -> int:
    """
    Insert a prediction into the predictions table.

    Args:
        conn: Database connection
        input_data_id: Foreign key reference to input_data table
        predicted_label: The predicted class (0-9 for MNIST)
        confidence: Confidence score of the prediction
        prediction_probabilities: Full probability distribution

    Returns:
        int: The ID of the inserted prediction
    """
    cursor = conn.cursor()

    # Serialize the probability array
    prob_bytes = serialize_image(prediction_probabilities)

    # Insert into database
    cursor.execute("""
        INSERT INTO predictions
        (input_data_id, predicted_label, confidence, prediction_probabilities)
        VALUES (%s, %s, %s, %s)
        RETURNING id
    """, (input_data_id, predicted_label, confidence, prob_bytes))

    prediction_id = cursor.fetchone()[0]
    conn.commit()
    cursor.close()

    print(f"Inserted prediction with ID: {prediction_id}")
    return prediction_id


def get_connection(
    host: str = "db",
    database: str = "milestone_3",
    user: str = "postgres",
    password: str = "postgres"
) -> psycopg2.extensions.connection:
    """
    Get a database connection.

    Args:
        host: Database host
        database: Database name
        user: Database user
        password: Database password

    Returns:
        Database connection object
    """
    return psycopg2.connect(
        host=host,
        database=database,
        user=user,
        password=password
    )
