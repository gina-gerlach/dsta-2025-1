# dsta-2025-1

**Course:** Data Science Toolkits and Architectures
**Authors:** Gina Gerlach & Sven Regli

This project demonstrates a complete machine learning workflow for training a **Convolutional Neural Network (CNN)** to recognize handwritten digits (0–9) from the MNIST dataset, achieving around 99% test accuracy. The project showcases Docker containerization, PostgreSQL database integration, and multi-container orchestration using Docker Compose.

## Project Structure
```
dsta-2025-1/
├── .dockerignore
├── .gitignore
├── docker-compose.yml          # Multi-container orchestration
├── Dockerfile                  # Original single-container setup
├── Dockerfile.train            # Model training container
├── Dockerfile.db               # Database application container
├── main.py                     # Standalone training script
├── db_app.py                   # Database integration application
├── README.md
├── requirements.txt
├── report/
│   └── report.md              # Comprehensive milestone documentation
└── src/
    ├── __init__.py
    ├── create_model.py         # CNN architecture definition
    ├── load_data.py            # MNIST data loading and preprocessing
    ├── model_io.py             # Model saving/loading utilities
    ├── predict.py              # Inference functions
    ├── train_model.py          # Training orchestration
    ├── db_helper.py            # PostgreSQL database utilities
    └── postgres_jokes.py       # PostgreSQL testing script
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
3. **Python Application** - Loads data, makes predictions, and stores results in the database

### 1. Clone the Repository

```bash
git clone https://github.com/gina-gerlach/dsta-2025-1.git
cd dsta-2025-1
```

### 2. Run with Docker Compose

```bash
docker-compose up
```

**What happens:**
- PostgreSQL database starts with health checks
- Model training service trains the CNN and saves it to a Docker volume
- Application service:
  - Waits for database to be healthy and model to be trained
  - Creates `milestone_3` database with `input_data` and `predictions` tables
  - Loads a sample MNIST image
  - Serializes and stores the image in the database (as BYTEA)
  - Retrieves and deserializes the image
  - Makes a prediction using the trained model
  - Stores the prediction with a foreign key reference to the input data

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

