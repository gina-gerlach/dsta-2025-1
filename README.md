# dsta-2025-1

**Course:** Data Science Toolkits and Architectures  
**Authors:** Gina Gerlach & Sven Regli  

This project trains a **Convolutional Neural Network (CNN)** to recognize handwritten digits (0–9) from the MNIST dataset.  
The model achieves around 99% test accuracy after training.

## Project structure
```
dsta-2025-1/
├── .gitignore
├── src/
│   └── models/
│       ├── __init__.py
│       └── mnist_convnet.py
├── report/
│   └── report.md
├── requirements.txt
└── README.md

```
## Requirements

Before running the project, ensure the following are installed:

- **Python 3.10+**
- **pip** (Python package manager)
- **venv** (for virtual environments)
- **Git** (for version control)

### Install dependencies

Create and activate a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate   # macOS / Linux
# or
.venv\Scripts\activate     # Windows PowerShell
```

then install the requirements packages:

 ``` bash
 pip install -r requirements.txt
 ```

## Run the code

To run follow the process below:

1. Clone the repository

``` bash
git clone git@github.com:gina-gerlach/dsta-2025-1
```

As the file is in the src folder change the directory to **src/models/**
``` bash
cd dsta-2025-1/src/models/
```

2. Run the script

``` bash
python mnist_convnet.py
```
