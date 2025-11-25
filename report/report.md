# Milestone Report
**Course:** Data Science Toolkits and Architectures  
**Authors:** Gina Gerlach & Sven Regli  
**Milestone 1** focuses on setting up the development environment, retrieving and running a deep learning model using the MNIST dataset, ensuring reproducibility, and establishing proper Git-based collaboration workflows.
**Milestone 2** focuses on improving project structure and dependency management, enforcing clean and reproducible development workflows through proper Git practices, virtual environments, and Docker, while expanding the codebase to support modular design, model training, saving/loading, and predictable cross-machine execution.

## Table of Contents
* **Milestone 1:**
- [1. Base Setup and Environment](#1-base-setup-and-environment)
- [2. Dataset Description](#2-dataset-description)
- [3. Commit Python file to Git Repository](#3-commit-python-file-to-git-repository)
- [4. Run Code](#4-run-code)
- [5. Explain the Code](#5-explain-the-code)
- [6. Add Documentation File](#6-add-documentation-file)
- [7. .gitignore Configuration](#7-gitignore-configuration)
- [8. Create Report and src Folder](#8-create-report-and-src-folder)
- [9. Issues and how they were solved](#9-issues-and-how-they-were-solved)
- [11. Tag and Release](#11-tag-and-release)
...
* **Milestone 2:**
- [12. .gitignore Dev Branch and Update](#12-gitignore-dev-branc-and-update)
- [13. Conceptual Questions](#13-conceptual-questions)
- [14. Code Modularization and Refactoring](#14-code-modularization-and-refactoring)
- [15. pip Requirement File and Virtual Environment](#15-pip-requirement-file-and-virtual-environment)
- [16. Dockerization](#16-dockerization)
- [17. Testing "Dockerized" Code](#17-testing-dockerized-code)
- [18. Tag and Release](#18-tag-and-release)


---
# Milestone 1

## 1. Base Setup and Environment

This section details the initial environment configuration for each team member.

### Gina's Setup:

*   Created an Ubuntu 22.04 (ARM64) virtual machine using **UTM (Universal Turing Machine)** on macOS.
*   Allocated **4 GB RAM** and set up a shared folder for file transfer.
*   Installed Python, Git, and other necessary tools.

### Sven's Setup:

*   Installed **WSL (Windows Subsystem for Linux)** with Ubuntu 24.04.3.
*   Installed Python 3.12.3.
*   Git was preinstalled with the Ubuntu 24.04 distribution.

The following command was used to install base packages on Ubuntu:

```bash
sudo apt install -y python3 python3-pip git python3-venv wget unzip build-essential curl
```


## 2. Dataset Description
| Feature | Value | Notes |
| :--- | :--- | :--- |
| **Name** | MNIST Handwritten Digit Database | Image collection of handwritten digits (0-9). |
| **Size** | 18.2 MB | |
| **Total Rows/Samples** | 70,000 | |
| **Image Type/Format** | 28x28px black and white images | |
| **Source** | Census Bureau employees and high school students | Evenly distributed between training and testing sets. |
| **Training Set Size** | 60,000 | Used for creating pattern recognition methods. |
| **Testing Set Size** | 10,000 | Used for evaluating model performance. |
| **Problem Type** | Classification | |
| **Number of Classes** | 10 | One class per digit (0-9). |
| **Images Per Class** | 7,000 total | 6,000 training, 1,000 testing. |

## 3. Commit Python File to Git Repository

The source code for the MNIST CNN example was obtained from the official Keras GitHub repository 
### 3.1 Configure Git and SSH

- Connect Git credentials

``` bash
git config --global user.name "MY USERNAME"
git config --global user.email "MY EMAIL"
```


- Create and add SSH key

``` bash
# Generate a new SSH key using the ed25519 algorithm
ssh-keygen -t ed25519 -C "MY EMAIL"

# start the SSH agent
eval "$(ssh-agent -s)"

# add the SSH key to the SSH agent
ssh-add ~/.ssh/id_ed25519

# Display the public key and copy it to GitHub
cat ~/.ssh/id_ed25519.pub
```

- Added key to GitHub and verified connection

``` bash
ssh -T git@github.com
```

### 3.2 Clone and Commit Code

``` bash
cd ~/projects

git clone git@github.com:gina-gerlach/dsta-2025-1

cd dsta-2025-1

git checkout -b gina_milestone1 
```

Download only the required script (using curl). Curl was chosen because only the single script was needed; otherwise, `git clone` would have been used

``` bash
curl -o mnist_convnet.py https://github.com/keras-team/keras-io/blob/master/examples/vision/mnist_convnet.py
```

Commit and push changes:

``` bash
git add mnist_convnet.py

git commit -m "Add initial MNIST code base and updated report" -m "Action corresponds to milestone 1 task #3. Updates to report.md reflect steps taken to complete task #3."

git push origin gina_milestone1
```


## 4. Run Code

### Enviroment setup

**1. Update and upgrade packages:**

``` bash
sudo apt update && sudo apt upgrade -y
```

**2. Install Python and dependencies:**

``` bash
sudo apt install -y python3 python3-pip git python3-venv
```

**3. Create a project directory:**

```bash
mkdir ~/projects
cd ~/projects
```
**4. Create and activate a virtual environment**

```bash
python3 -m venv venv
```

``` bash
source venv/bin/activate
```

**5. Install required packages**

```bash
pip install --upgrade pip
pip install tensorflow keras numpy
```

**6. Run the MNIST model**

```bash
python mnist_convnet.py
```

### Current versions

- Python 3.10.12 | 3.12.3

- NumPy 2.2.6

- Keras 3.11.3

- TensorFlow 2.20.0

These versions were pinned  in `requirements.txt` :

```bash
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Add requirements.txt for reproducibility" -m "File contains versions of packages needed to run code"
git push origin gina_milestone1
```
### Are the versions dependent on the system the code is being run on?

The code was tested on **Ubuntu**, **macOS** and  **Windows**, confirming that it runs independently of the operating system

Example for macOS: 
```bash
#create test folder and clone repository
mkdir ~/test_env && cd ~/test_env
git clone git@github.com:gina-gerlach/dsta-2025-1
cd dsta-2025-1 #before moving to the `src` folder
```
```bash
#create and activate the venv
python3 -m venv venv
source venv/bin/activate

#install requirements
pip install -r requirements.txt

#run the code
python mnist_convnet.py
```

Same can be said for running it on a windows system: 

```bash
#clone repository
git clone https://github.com/gina-gerlach/dsta-2025-1.git
```
```python
#create and activate the venv
python -m venv .venv 
.venv/Scripts/Activate.ps1

#install requirements
pip install -r requirements.txt

#run the code
python mnist_convnet.py
```
## 5. Explain the Code

### What is the input to and the output from the neural network

**Input:** `input_shape = (28, 28, 1)`
→
28 pixel by 28 pixel grayscale (1 color channel) images

**Output:** `num_classes = 10` 
→
10 classes that the image can be sorted into (0-9)

### What is Keras? And how does it relate to Tensorflow?

Keras is an API or Application Programming Interface used for building neural networks. 
It is high level, meaning it automates otherwise complex problems associated with building a neural network. 
For example, Keras has functions and classes to create layers, build models, preprocess data, and train and evaluate models.


TensorFlow, on the otherhand, is a low-level engine that actually performs the operations called via Keras. 
In relation to Keras, TensorFlow is backend engine that performs the computations while Keras is the user friendly interface.

### How is the data loaded

``` python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
```

This code snippet automatically downloads the MNIST dataset from Keras and stores it locally. For subsequent runs, the cached version is used


### Which dependencies are imported, what do they do?

**numpy:** (numerical python) for array manipulation and preprocessing

**keras** for model building, training and evaluation

**keras.layers** for Convolutional Neural Network (CNN) layers

### What kind of neural network architecture are you dealing with?

A Convolutional Neural Network (CNN) or "convnet" as seen in the code when it uses the function:

> ```layers.Conv2D```

### Why do you need a neural network for this task in the first place?

The problem is a pattern recognition problem so you need a model that can learn spatial features from images.
A simpler machine learning model wouldn't work because you need someone to manually engineer the features which is too complex. A neural network learns the features independently.

## 6. Add Documentation File

A `README.md` was added to explain setup and execution, and this `report.md` (inside the **report/** folder) documents each milestone task and result.



## 7. .gitignore Configuration

To prevent unnecessary or large files from being committed, a `.gitignore` file was created.  
It excludes Python caches, virtual environments, data files, system files, and IDE-specific folders.

Below is a shortened example of the key entries; the actual file in the repository contains additional entries for broader coverage.

```bash
# Python cache and virtual environments
__pycache__/
*.pyc
.venv/
venv/

# Data or model files
*.csv
*.zip
*.h5
*.npz

# System/IDE
.DS_Store
.ipynb_checkpoints/
```

## 8. Create Report and src Folder
To improve organization and follow best practices, the project was restructured as follows:

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


Created folder and move files:

```bash
# create folders
mkdir report
mkdir src
# move files
mv report.md report/
mv mnist_convnet.py src/models/
```
Then the changes were committed. Since the files were moved, you'll need to change the directory to run the script: 
``` bash
cd src/models/
python mnist_convnet.py
```

## 9. Issues and How They Were Solved

**Gina:**
The commit messages were not as clean or detailed as I would have liked. I modified my commits with the following commands:

``` bash
git log --oneline

git commit --amend -m "Title" -m "Subtitle"

git rebase -i <commit-hash from log>^

git rebase --continue

git push force
```
**Sven:** 
Encountered merge conflicts after Gina’s branch was merged first and mine was therefore not up to date.
Resolved this by rebasing and manually fixing merge conflicts:
``` bash 
git checkout main
git fetch main
git checkout feature/Milestone1_sven
git rebase main
#resolve the merge conflicts
git add . 
git rebase --continue
```


## 10. Pull Request and Code Review
After finalizing all files in the branch `gina_milestone1`, a Pull Request was opened to merge it into `main`.  
The PR was **reviewed and approved by Sven** before merging.  
This ensures that only reviewed, functional code enters the main branch.

## 11. Tag and Release
After merging into `main`, a release tag was created for grading:

```bash
git tag milestone_1
git push origin milestone_1
```
>  **Milestone 1 Summary:**  
> Both team members successfully set up Linux-based environments, ran a reproducible MNIST Convolutional Neural Network (CNN) model, and implemented a clean Git-based collaboration workflow.  
> This milestone establishes a solid technical foundation for future milestones and ongoing project development.


---
# Milestone 2

Milestone 2 focused on the development of a Dockerization of the project and to split the single mnist_convnet.py file into reusable modules. We split the 6 Tasks among us and documented each Task using issues on Github.

## 12. .gitignore Dev Branch and Update

The initial .gitignore was already created and added to the root folder in Step 7.
To coordinate changes to the .gitignore, we created a dedicated dev branch for editing the .gitignore. This keeps the .gitignore updates centralized and version-controlled. It also allows us to work on seperate features without opening pull requests to main for small .gitignore changes and reduces merge conflicts.

```bash
git checkout -b dev
git push origin dev
```

From that branch the .gitignore is then edited and commited

```bash
git add .gitignore
git commit -m "Update..."
```

Then each of us could pull the latest dev branch to update the .gitignore with their own branches

```bash
git fetch origin
git checkout dev
git pull
```

To keep the repository clean, we use rebase to integrate changes to .gitignore without merging into main unnecessarily

```bash
git checkout feature_branch
git rebase dev
```

An issue arose when updating a feature branch after many commits had been made. Instead of using rebase, which requires you resolve conflicts on each commit, we used merge at that time. Then conflicts only had to be resolved once

```bash
git checkout feature_branch
git merge dev
```

At this time, the following entries were added:

| Group | File Types | Notes |
| :--- | :--- | :--- |
| **Large datasets/ML files** | .csv, .zip, .h5, .npz, .pt, .pth, checkpoints/, data/, datasets/ | These files are large binary blobs that change often and are therefore not suitable for version control. Excluding them prevents bloating the repository and increasing pull time. |
| **Mesia/assets** | media/, assets/, .png, .jpg, .jpeg, .gif, .mp4, .mp3  | Image, audio, and video files are large binaries and should not be versioned. |
| **OS-specific system files** | Thumbs.db, desktop.ini, ~ , .nfs, .Trash- | These files are automatically generated by Windows, MacOS or Linux/Ubuntu. They aren't relevant to the project and ignoring them prevents clutter. |


## 13. Conceptual Questions

**What is a Hash function? What are some use cases?**

A hash function converts data into a unique alphanumeric output string. This output, called a hash, is completely unique to the input and if even one digit of the input changes, so does the hash.

Hash functions are used in:
- Code versioning and Reproducibility: On Git, file version and commits have a hash which is used to ensure that they are what they say they are and can be reproduced accurately.
- Dependency Verification: For requirements files, the SHA256 hash of a package can be used to verify that the package the user downloads is the same as the author. 
- Data Integrity: When downloading large datasets, hashes may be provided to ensure that the data was downloaded correctly to the user's computer and no files were corrupted during download.
- Security: SSH protocol relies on hashing as part of it's encryption process.

**What is a Python module, pckage and script? How do they differ from one another?**

A Python module is a single file (module.py) that is imported into other Python code. It contains definitions for functions, classes, and variables. 

A Python package is folder with a collection of modules and typically includes an _ init _.py file. You import a package with syntax like:
```bash
from package.module1 import function
```

A Python script is a Python file that is meant to be executed directly from the command line. A modul can be run as a script, but a file is considered a script when its main purpose is execution, not to be imported into another file. 

Overall, Python modules and packages are components of a project, that can make them run, and a script is code that completes the desired action of the project.

**How would you explain a Docker container and volume to a child?**

A Docker container is like a backpack. It holds everything you need for doing your school work - notebooks, pens, pencils, textbooks, etc. You can take this backpack to school, to the library, to a friend's house and still be able to do your schoolwork the same way every time without worrying about if you have what you need in it.

A Docker Volume is like a magic notebook outside of that backpack that saves all the work you do with the tools in your backpack automatically. If you lose your backpack you still have all of the work you did saved in the magic notebook.

**What is your preference concerning the use of Python virtualenv and Docker? When would you use one or the other?**

Our preference is to use Python virtualenvs for local development and package isolation and Docker for sharing testing and deployment. You can also use a virtual environment within a Docker container for optimal reproducibility and deployment.
Python virtualenvs is easy to use and lightweight and manages Python package versions well. It's good when only working on small scripts.
However, Docker is more powerful and addresses system-level environment conflicts with all OS dependencies. This is best for deploying complex workflows that are run across multiple services - for example ML pipelines that are linked with a web application and PostgreSQL database. 

**What is the Docker build context?**

The Docker build context is the set of local files at a specific path that the Docker daemon can access during the image build process. In this path are the local files (source code, environment files, configuration) and the Dockerfile (the file that provides the "recipe" for building an image out of the application).

**How can you assess the quality of a Python package on PyPI**

PyPI, or the Python Package Index, is the central repository for Python software. You can assess the quality of a package by checking its:
- Package metadata: Look for a recent release date, up to date versioning and a clear description
- Project Information: Should have a link to a GitHub repo with commits, contributors and issue tracker activity
- Project Trust: Check the number of downloads, stars on GitHub and if it has been used by reputable projects
- Documentation: Should have a README, Application Programming Interface (API) docs, and usage examples/tutoritals.


## 14. Code Modularization and Refactoring

### Objectives
Tasks 3 and 4 required restructuring the original monolithic `mnist_convnet.py` script into a modular architecture with the following functionality:

- Can load data
- Can train (fit) a neural network on the data
- Can save a fitted model to a ".h5" file (or saved model type for newer TensorFlow 2.0 versions)
- Can load a ".h5" file, using Keras (or saved model type for newer TensorFlow 2.0 versions)
- Can perform predictions using a "fitted" model, using Keras

### Modular Architecture

All objectives were met by splitting the original monolithic script into five specialized modules within the `src/` package:

#### 1. [load_data.py](src/load_data.py)
**Purpose:** Data loading and preprocessing
**Functions:** `load_mnist_data()`
**Responsibilities:**
- Loads the MNIST dataset from Keras datasets
- Normalizes pixel values to the [0, 1] range
- Reshapes images to (28, 28, 1) format
- Converts labels to one-hot encoded categorical format
- Defines data constants (`NUM_CLASSES=10`, `INPUT_SHAPE=(28, 28, 1)`)

**Rationale:** Separates data handling logic from model and training logic. This module can be reused by any component that needs MNIST data and ensures consistent preprocessing across the entire pipeline.

#### 2. [create_model.py](src/create_model.py)
**Purpose:** Neural network architecture definition
**Functions:** `create_model()`
**Responsibilities:**
- Defines the CNN architecture (2 Conv2D layers, 2 MaxPooling layers, Dropout, Dense output layer)
- Compiles the model with categorical crossentropy loss and Adam optimizer
- Imports configuration constants from `load_data.py` (`INPUT_SHAPE`, `NUM_CLASSES`)

**Rationale:** Isolates the model architecture in a single location following the Single Responsibility Principle. This makes it easy to experiment with different architectures by simply modifying this module without affecting training or prediction code. The function-based approach allows for easy testing and potential parameterization in the future.

#### 3. [train_model.py](src/train_model.py)
**Purpose:** Model training orchestration
**Functions:** `train_model(epochs=5, batch_size=128)`
**Responsibilities:**
- Orchestrates the training process by importing and calling `load_mnist_data()` and `create_model()`
- Fits the model on training data with specified hyperparameters
- Evaluates the model on test data
- Returns the trained model

**Rationale:** This module acts as a high-level training coordinator that brings together data loading and model creation. By accepting `epochs` and `batch_size` as parameters, it provides flexibility for hyperparameter tuning. This separation allows training logic to be reused independently of data loading or model architecture changes.

#### 4. [model_io.py](src/model_io.py)
**Purpose:** Model persistence (saving and loading)
**Functions:** `save_model(model, filepath='mnist_model.h5')`, `load_model(filepath='mnist_model.h5')`
**Responsibilities:**
- Saves trained models to disk in HDF5 format
- Loads previously saved models from disk
- Provides clear console feedback about save/load operations

**Rationale:** Separates I/O operations from training and prediction logic. This module provides a clean interface for model persistence, making it easy to save checkpoints during training or deploy models in production. The default filepath parameter reduces boilerplate while maintaining flexibility.

#### 5. [predict.py](src/predict.py)
**Purpose:** Model inference
**Functions:** `predict(model, x)`, `predict_classes(model, x)`
**Responsibilities:**
- `predict()`: Returns raw probability distributions for each class
- `predict_classes()`: Returns the predicted class labels (argmax of probabilities)

**Rationale:** Encapsulates inference logic in dedicated functions. This separation allows prediction functionality to be reused across different contexts (batch predictions, single predictions, evaluation scripts) without duplicating code. The two-function approach provides flexibility for users who need either raw probabilities or class labels.

### Main Entry Point: [main.py](main.py)

The `main.py` script serves as the orchestration layer that demonstrates the complete machine learning pipeline:

```python
from src.train_model import train_model
from src.load_data import load_mnist_data
from src.model_io import save_model, load_model
from src.predict import predict_classes
```

**Execution Flow:**
1. Loads and preprocesses data using `load_mnist_data()`
2. Trains the model using `train_model(epochs=5, batch_size=128)`
3. Saves the trained model using `save_model(model, 'mnist_model.h5')`
4. Demonstrates model loading using `load_model('mnist_model.h5')`
5. Performs sample predictions on 10 test images using `predict_classes()`
6. Compares predictions against ground truth labels

**Rationale:** The `main.py` script demonstrates best practices by:
- Importing only what's needed from each module
- Following a clear, linear execution flow
- Serving as executable documentation of the ML pipeline
- Being runnable with a simple `python main.py` command

### Reasoning Behind the Modularization

**1. Single Responsibility Principle:** Each module has one clear purpose. `load_data.py` handles data, `create_model.py` defines architecture, `train_model.py` orchestrates training, etc. This makes the codebase easier to understand and maintain.

**2. Reusability:** Functions can be imported and reused across different scripts. For example, `load_mnist_data()` can be used by training scripts, evaluation scripts, or visualization tools without code duplication.

**3. Testability:** Each module can be tested independently. You can test data loading without training a model, or test model architecture creation without loading data.

**4. Separation of Concerns:** Training logic is separate from model architecture, which is separate from data loading. This allows team members to work on different aspects simultaneously without conflicts.

**5. Maintainability:** When changes are needed (e.g., switching from MNIST to a different dataset, or modifying the CNN architecture), modifications are localized to specific modules rather than scattered throughout a monolithic script.

**6. Scalability:** The modular structure makes it easy to extend functionality. For example, adding data augmentation would only require modifying `load_data.py`, or adding a new model architecture would just mean creating a new function in `create_model.py`.

**7. PEP 8 Compliance:** The modularized code follows Python naming conventions (lowercase with underscores for functions and modules), proper import organization, and clear function definitions with appropriate parameter defaults.

### Module Interdependencies

The modules work together through a dependency hierarchy:
- `load_data.py` has no internal dependencies (only external: numpy, keras)
- `create_model.py` imports constants from `load_data.py`
- `train_model.py` imports functions from both `load_data.py` and `create_model.py`
- `model_io.py` has no internal dependencies (only external: keras)
- `predict.py` has no internal dependencies (operates on provided model objects)
- `main.py` imports from all modules to orchestrate the complete pipeline

This hierarchical structure prevents circular dependencies and creates a clean, maintainable architecture.

## 15. pip Requirements File and Virtual Environment

The requirement file was already created in Milestone 1. The file was reviewed to ensure all dependencies were pinned with their fixed version. Below are the dependencies and their corresponding SHA256 hashes which were count on pypi.org:

| **Package Name** | **Version** | **SHA256 Hash Digest** |
| :--- | :--- | :--- |
| absl-py | 2.3.1 | eeecf07f0c2a93ace0772c92e596ace6d3d3996c042b2128459aaae2a76de11d |
| astunparse | 1.6.3 | c2652417f2c8b5bb325c885ae329bdf3f86424075c4fd1a128674bc6fba4b8e8 |
| certifi | 2025.10.5 | 47c09d31ccf2acf0be3f701ea53595ee7e0b8fa08801c6624be771df09ae7b43 |
| charset-normalizer | 3.4.4 | 94537985111c35f28720e43603b8e7b43a6ecfb2ce1d3058bbe955b73404e21a |
| contourpy | 1.3.2 | b6945942715a034c671b7fc54f9588126b0b8bf23db2696e3ca8328f3ff0ab54 |
| cycler | 0.12.1 | 88bb128f02ba341da8ef447245a9e138fae777f6a23943da4540077d3601eb1c |
| flatbuffers | 25.9.23 | 676f9fa62750bb50cf531b42a0a2a118ad8f7f797a511eda12881c016f093b12 |
| fonttools | 4.60.1 | ef00af0439ebfee806b25f24c8f92109157ff3fac5731dc7867957812e87b8d9 |
| gast | 0.6.0 | 88fc5300d32c7ac6ca7b515310862f71e6fdf2c029bbec7c66c0f5dd47b6b1fb |
| google-pasta | 0.2.0 | c9f2c8dfc8f96d0d5808299920721be30c9eec37f2389f28904f454565c8a16e |
| grpcio | 1.76.0 | 7be78388d6da1a25c0d5ec506523db58b18be22d9c37d8d3a32c08be4987bd73 |
| h5py | 3.15.1 | c86e3ed45c4473564de55aa83b6fc9e5ead86578773dfbd93047380042e26b69 |
| idna | 3.11 | 795dafcc9c04ed0c1fb032c2aa73654d8e8c5023a7df64a53f39190ada629902 |
| keras | 3.11.3 | efda616835c31b7d916d72303ef9adec1257320bc9fd4b2b0138840fc65fb5b7 |
| kiwisolver | 1.4.9 | c3b22c26c6fd6811b0ae8363b95ca8ce4ea3c202d3d0975b2914310ceb1bcc4d |
| libclang | 18.1.1 | a1214966d08d73d971287fc3ead8dfaf82eb07fb197680d8b3859dbbbbf78250 |
| Markdown | 3.9 | d2900fe1782bd33bdbbd56859defef70c2e78fc46668f8eb9df3128138f2cb6a |
| markdown-it-py | 4.0.0 | cb0a2b4aa34f932c007117b194e945bd74e0ec24133ceb5bac59009cda1cb9f3 |
| MarkupSafe | 3.0.3 | 722695808f4b6457b320fdc131280796bdceb04ab50fe1795cd540799ebe1698 |
| matplotlib | 3.10.7 | a06ba7e2a2ef9131c79c49e63dad355d2d878413a0376c1727c8b9335ff731c7 |
| mdurl | 0.1.2 | bb413d29f5eea38f31dd4754dd7377d4465116fb207585f97bf925588687c1ba |
| ml_dtypes | 0.5.3 | 95ce33057ba4d05df50b1f3cfefab22e351868a843b3b15a46c65836283670c9 |
| namex | 0.1.0 | 117f03ccd302cc48e3f5c58a296838f6b89c83455ab8683a1e85f2a430aa4306 |
| numpy | 2.2.6 | e29554e2bef54a90aa5cc07da6ce955accb83f21ab5de01a62c8478897b264fd |
| opt_einsum | 3.4.0 | 96ca72f1b886d148241348783498194c577fa30a8faac108586b14f1ba4473ac |
| optree | 0.17.0 | 5335a5ec44479920620d72324c66563bd705ab2a698605dd4b6ee67dbcad7ecd |
| packaging | 25.0 | d443872c98d677bf60f6a1f2f8c1cb748e8fe762d2bf9d3148b5599295b0fc4f |
| pillow | 12.0.0 | 87d4f8125c9988bfbed67af47dd7a953e2fc7b0cc1e7800ec6d2080d490bb353 |
| protobuf | 6.33.0 | 140303d5c8d2037730c548f8c7b93b20bb1dc301be280c378b82b8894589c954 |
| Pygments | 2.19.2 | 636cb2477cec7f8952536970bc533bc43743542f70392ae026374600add5b887 |
| pyparsing | 3.2.5 | 2df8d5b7b2802ef88e8d016a2eb9c7aeaa923529cd251ed0fe4608275d4105b6 |
| python-dateutil | 2.9.0.post0 | 37dd54208da7e1cd875388217d5e00ebd4179249f90fb72437e91a35459a0ad3 |
| requests | 2.32.5 | dbba0bac56e100853db0ea71b82b4dfd5fe2bf6d3754a8893c3af500cec7d7cf |
| rich | 14.2.0 | 73ff50c7c0c1c77c8243079283f4edb376f0f6442433aecb8ce7e6d0b92d1fe4 |
| six | 1.17.0 | ff70335d468e7eb6ec65b95b99d3a2836546063f63acc5171de367e834932a81 |
| tensorboard | 2.20.0 | 9dc9f978cb84c0723acf9a345d96c184f0293d18f166bb8d59ee098e6cfaaba6 |
| tensorboard-data-server | 0.7.2 | 7e0610d205889588983836ec05dc098e80f97b7e7bbff7e994ebb78f578d0ddb |
| tensorflow | 2.20.0 | 47c88e05a07f1ead4977b4894b3ecd4d8075c40191065afc4fd9355c9db3d926 |
| termcolor | 3.2.0 | 610e6456feec42c4bcd28934a8c87a06c3fa28b01561d46aa09a9881b8622c58 |
| typing_extensions | 4.15.0 | 0cea48d173cc12fa28ecabc3b837ea3cf6f38c6d1136f85cbaaf598984861466 |
| urllib3 | 2.5.0 | 3fc47733c7e419d4bc3f6b3dc2b4f890bb743906a30d56ba4a5bfa4bbff92760 |
| Werkzeug | 3.1.3 | 60723ce945c19328679790e3282cc758aa4a6040e4bb330f53d30fa546d44746 |
| wrapt | 2.0.0 | 35a542cc7a962331d0279735c30995b024e852cf40481e384fd63caaa391cbb9 |

### Virtual Environment

Create and activate a virtual environment with

```bash
python3 -m venv venv
source venv/bin/activate   # macOS / Linux
# or
.venv\Scripts\activate     # Windows PowerShell
```

Then install the requirements packages with

```bash
pip install -r requirements.txt
```

## 16. Dockerization

The goal was to dockerize the entire project to prevent the classic "it works on my machine" issue.

**Initial Approach:**
We started with a basic Docker image using Python 3.12:

```dockerfile
FROM python:3.12-slim
WORKDIR /app

RUN pip install uv
RUN uv pip install -r requirements.txt --target /app/.venv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
FROM python:3.12-slim
COPY main.py .
COPY src/ src/

CMD ["python", "main.py"]
```
We used copy src/src/ to preserve the structure of our build, to ensure the import in main.py `from src.train_model import train_model` are working correctly. 

Since TensorFlow is a large library, this approach took considerable time until everything was installed and built—initially around 4 minutes per build.

**Optimized Approach:**
We then switched to installing dependencies using `uv`, which significantly improved build times:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install uv once, globally
RUN pip install --no-cache-dir uv

# Copy requirements.txt for caching
COPY requirements.txt .

# Install deps into the system Python inside the container
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Copy the source code and main.py 
COPY src/ src/
COPY main.py .

CMD ["python", "main.py"]
```

UV is a super fast replacement for pip, built in Rust, which allows for faster downloads and parallelization of installing the requirements.This optimization proved to be a good investment of time, the Docker image now builds in roughly 1/4 of the time. This was especially helpful during debugging sessions. When there were errors in one of the modules or missing libraries in requirements.txt, we no longer had to wait as long as before, however not all of that can be attributed to uv, as some of the installations are cached. 

**Why UV/Python 3.12 and not the TensorFlow image?**
We used the Python 3.12-slim ( an official Image by Docker) image to speed up installations, as the TensorFlow image itself is already huge (around 3GB). Adding a `.dockerignore` file was essential to prevent the image from growing even larger. We excluded items like `.venv` and the `report` directory from the Docker context,not only to save space but mainly to avoid including unnecessary files in the image. The image only needs the `src` directory and `main.py` file to run and the requirements.txt for dependencies. 

## 17. Testing "Dockerized" Code

On Gina's setup the "dockerized" code was tested prior to merging to main branch.
From the feature branch:

1. Build the Docker image

```bash
docker build -t data-mnist:latest .
```

The Docker image was built successfully with no errors. After this we ran the container with -it to show the outputs to verify the code worked, and  —rm to remove the container after it finished.

```bash
docker run --rm -it dsta-mnist:latest
```

The output showed that the dataset loaded, the neural network trained for 5 epochs, the model was evaluated on the test set, the model was saved as a .h5 file in the container, the model was loaded back correctly and run. The final predictions were accurate on unseen data. 

## 18. Tag and Release
Finally, the ReadMe was updated with the current project structure and Docker workflow to run the code.

After merging into `main`, a release tag was created for grading:

```bash
git tag milestone_2
git push origin milestone_2
```
>  **Milestone 2 Summary:**  
> Both team members successfully modularized the MNIST CNN code, created a fully reproducible Dockerized workflow, and ensured the pipeline runs end-to-end (training, saving/loading the model, and performing inference) across any system.
