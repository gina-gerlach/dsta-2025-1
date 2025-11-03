# Milestone Report

**Course:** Data Science Toolkits and Architectures  
**Authors:** Gina Gerlach & Sven Regli  

**Milestone 1** focuses on setting up the development environment, retrieving and running a deep learning model using the MNIST dataset, ensuring reproducibility, and establishing proper Git-based collaboration workflows.


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
- [10. Pull Request and Code Review](#10-pull-request-and-code-review)
- [11. Tag and Release](#11-tag-and-release)
...
* **Milestone 2:**
- [Milestone 2](#milestone2)




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
To be defined
---

