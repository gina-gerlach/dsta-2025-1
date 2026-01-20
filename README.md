# dsta-2025-1

**Course:** Data Science Toolkits and Architectures
**Authors:** Gina Gerlach & Sven Regli

This project demonstrates a complete machine learning workflow for training a **Convolutional Neural Network (CNN)** to recognize handwritten digits (0–9) from the MNIST dataset, achieving around 99% test accuracy. The project showcases **Docker containerization, PostgreSQL database integration, Flask REST API**, and a **front-end webpage** for image upload and prediction.

## Project Structure
```
dsta-2025-1/
├── docker/                      # All Docker-related files
│   ├── Dockerfile.db
│   ├── Dockerfile.flask
│   ├── Dockerfile.train
│   ├── Dockerfile.wandb
│   └── docker_entrypoint.sh
├── scripts/                     # Executable scripts
│   ├── db_app.py
│   ├── flask_app.py
│   ├── main.py
│   ├── postgres_jokes.py
│   ├── train_and_save.py
│   ├── train_and_save_wandb.py
│   └── templates/
│   	└── upload.html               # Front-end upload page
├── src/                         # Reusable library modules
    ├── __init__.py
    ├── create_model.py             # CNN architecture definition
    ├── load_data.py                # MNIST data loading and preprocessing
    ├── model_io.py                 # Model saving/loading utilities
    ├── predict.py                  # Inference functions
    ├── train_model.py              # Training orchestration
    ├── db_helper.py                # PostgreSQL database utilities
    └── postgres_jokes.py           # PostgreSQL testing script
├── tests/                       # Test files
│   └── test_flask_api.py
├── notebooks/
├── report/
│   └── report.md                # Comprehensive milestone documentation
├── docker-compose.yml           # Multi-container orchestration
├── requirements.txt
└── README.md

```
## Requirements

- **Docker** and **Docker Compose** (recommended for full application)
- **Python 3.10+** (for local development)
- **Git** (for version control)

---

## Quick Start: Multi-Container Application (Milestone 3)

The complete application uses Docker Compose to orchestrate three services:
1. **PostgreSQL Database** - Persistent data storage
2. **Model Training** - Trains and saves the CNN model
3. **Flask REST API** - Exposes the prediction API
4. **Front-End Webpage** - Upload images and view predictions

### 1. Clone the Repository

```bash
git clone https://github.com/gina-gerlach/dsta-2025-1.git
cd dsta-2025-1
```

### 2. Run with Docker Compose

```bash
docker-compose up --build
```

**What happens:**
- PostgreSQL database starts with health checks
- Model training service runs once to generate the MNIST model
- Flask container:
  - Waits for the database to be ready and the model to be trained
  - Starts the REST API (port 5001)
  - Serves the front-end at http://localhost:5001/
- Front-end webpage:
- Users can upload images (PNG, JPG)
- Images are converted to grayscale, resized to 28×28, normalized
- Prediction results, confidence score, and database IDs are displayed

### 3. Clean Up

```bash
# Stop containers
docker-compose down

# Remove volumes (fresh start)
docker-compose down -v
```


## Documentation

For detailed documentation, see [report/report.md](report/report.md), which includes:
- Complete setup instructions for each milestone
- Architectural decisions and rationale
- Database schema and ER diagrams
- Docker Compose configuration explanations
- Flask API and front-end implementation details
- Conceptual questions and answers

---

## Technologies Used

- **Python 3.12** - Core programming language
- **TensorFlow/Keras** - Deep learning framework
- **NumPy** - Numerical computing
- **PostgreSQL** - Relational database
- **psycopg2** - PostgreSQL adapter for Python
- **Docker & Docker Compose** - Containerization and orchestration
- **Pillow** - Image processing

---

