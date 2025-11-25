# dsta-2025-1

**Course:** Data Science Toolkits and Architectures  
**Authors:** Gina Gerlach & Sven Regli  

This project trains a **Convolutional Neural Network (CNN)** to recognize handwritten digits (0–9) from the MNIST dataset.  
The model achieves around 99% test accuracy after training.

## Project structure
```
dsta-2025-1/
├── .dockerignore
├── .gitignore
├── Dockerfile
├── main.py
├── README.md
├── requirements.txt
├── report/
│   └── report.md
└── src/
    ├── __init__.py
    ├── create_model.py
    ├── load_data.py
    ├── model_io.py
    ├── predict.py
    └── train_model.py
```
## Requirements

Before running the project, ensure the following are installed:

- **Python 3.10+**
- **pip** (Python package manager)
- **venv** (for virtual environments)
- **Git** (for version control)
- **Docker** (for reproducability across machines)

## Update 25.11.2025 - Dockerized Workflow

### 1. Clone the repository at the release tag

```bash
git clone https://github.com/gina-gerlach/dsta-2025-1.git
cd dsta-2025-1
git checkout main 
```

### 2. Build the Docker image

```bash
docker build -t dsta-mnist:latest .
```

### 3. Run the container

```bash
docker run --rm -it dsta-mnist:latest
```

### Results:
- MNIST dataset downloads automatically
- Neural network is trained and evaluated
- Model is saved and loaded inside the container
- Predictions are printed for sample test images

Note: No virtual environment or additional setup is needed; Docker handles all dependencies.
