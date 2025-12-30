# Milestone Report
**Course:** Data Science Toolkits and Architectures  
**Authors:** Gina Gerlach & Sven Regli  
**Milestone 1** focuses on setting up the development environment, retrieving and running a deep learning model using the MNIST dataset, ensuring reproducibility, and establishing proper Git-based collaboration workflows.
**Milestone 2** focuses on improving project structure and dependency management, enforcing clean and reproducible development workflows through proper Git practices, virtual environments, and Docker, while expanding the codebase to support modular design, model training, saving/loading, and predictable cross-machine execution.
**Milestone 3** focuses on multi-container Docker applications with PostgreSQL database integration, including image data serialization, relational database design, and container orchestration using Docker Compose with proper health checks and volume persistence.

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

* **Milestone 2:**
- [12. .gitignore Dev Branch and Update](#12-gitignore-dev-branch-and-update)
- [13. Conceptual Questions](#13-conceptual-questions)
- [14. Code Modularization and Refactoring](#task-3-code-modularization-and-refactoring)
- [15. pip Requirement File and Virtual Environment](#15-pip-requirement-file-and-virtual-environment)
- [16. Dockerization](#dockerization)
- [17. Testing "Dockerized" Code](#17-testing-dockerized-code)
- [18. Tag and Release](#18-tag-and-release)

* **Milestone 3:**
- [Task 1: Docker-compose Installation and Questions](#task-1-docker-compose-installation-and-questions)
- [Task 2: PostgreSQL and pgAdmin Questions, Installation and Test](#task-2-postgresql-and-pgadmin-questions-installation-and-test)
- [Task 3: Image Storage in PostgreSQL](#task-3-image-storage-in-postgresql)
  - [The Impedance Mismatch Problem](#the-impedance-mismatch-problem)
  - [MNIST Dataset Structure](#mnist-dataset-structure)
  - [Database Table Design](#database-table-design)
- [Task 4: Multi-Docker Container Application](#task-4-multi-docker-container-application)
  - [Architecture Overview](#architecture-overview)
  - [Database Schema](#database-schema)
  - [Docker Volumes](#docker-volumes)
  - [Application Workflow](#application-workflow)
  - [Docker Compose Startup Order](#docker-compose-startup-order)
  - [Running the Application](#running-the-application)
  - [Key Learnings](#key-learnings)
- [Additional Questions](#additional-questions)
  - [What is an SQL Injection Attack and how can you protect yourself?](#what-is-an-sql-injection-attack-and-how-can-you-protect-yourself)
  - [What is ACID in the context of SQL Databases?](#what-is-acid-in-the-context-of-sql-databases)
  - [What is the difference between a Relational Database and a Document Store?](#what-is-the-difference-between-a-relational-database-and-a-document-store)
  - [What is a SQL Join Operation? What other common SQL statements exist?](#what-is-a-sql-join-operation-what-other-common-sql-statements-exist)
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

# Milestone 3
## Task 1: Docker-compose Installation and Questions

Task 1 introduces Docker Compose, a tool for defining and running multi-container Docker applications. This task explores service orchestration, network communication between containers, and port mapping.

**Docker Compose version installed:** v2.40.3

### Which services are being used for the application (described in the link above)? How do they relate to the host names in terms of computer networks?

The Compose file defines two services: `web` and `redis`. 
`web` runs a Flask application. It handles HTTP requests from the browser and for each request calls Redis to increment a counter. 
`redis` is the server or database. It stores the counter value and listens for connections from web on its default port 6379.

Each service runs in its own container and Docker configures an internal network where each container is reachable by a hostname that’s identical to its service name. 

So, `web` connects to the `redis` service using the hostname `redis` on port 6379, just like computers on a normal network use hostnames to read each other.
 
### What ports are being used (within the application and in the docker-compose file)?

|   | **application**  | **docker-compose**  |
|:---|:---|:---|
| `web`  | 5000 (default port for Flask web server)  | 8000:5000 (host port:container port) | 
| `redis`  | 6379 (default port)  | N/A redis doesn’t talk directly with host computers  |

### How does the host machine (e.g. your computer) communicate with the application inside the
Docker container. Which ports are exposed from the application to the host machine?

The host machine communicates with the Flask app via the mapped ports listed above (8000:5000). The application listens on port 5000 inside the web container while Docker forwards traffic from localhost:8000 to the internal port 5000. That way when you open http:localhost:8000/ in a browser the request reaches the Flask app inside the container. The only port exposed from the application to the host in this example is host port 8000 mapped to container port 5000 on the web service.

### What is localhost, why is it useful in the domain of web applications?

`localhost` is the standard hostname that refers to the local machine. It’s useful because it allows you to test web applications locally before deploying them to remote servers.

## Task 2: PostgreSQL and pgAdmin Questions, Installation and Test

Task 2 focuses on setting up and interacting with a PostgreSQL database running in a Docker container. This includes using Python adapters to programmatically interact with the database and using pgAdmin as a graphical management tool.

### What is PostgreSQL? Is it SQL or no-SQL (why?)

It is an open-sourced object-relational database management system. It is SQL because it organizes data in relational tables with rows, columns and keys and uses SQL for defining and querying data. This is different from no-SQL databases (like Mongo or Redis) as they typically store data in formats like document or graph without fixed table schemas and use non-SQL query models.


### Run a PostgreSQL Server (with the most current version) using a Docker image from the official PostgreSQL Docker Hub page

Pull the latest official image:

```bash
docker pull postgres:latest
```

Create a user-defined Docker network so containers resolve each other by name (important for later steps):

```bash
docker network create pg-net
```

Run a container:

```bash
docker run --name my-postgres --network pg-net \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=postgres \
  -p 5432:5432 \
  -d postgres:latest
```
This code sets the container name, custom network, username, password and default database. It maps the host port 5432 to container port 5432 and runs the container in detached mode.

### Make sure you expose the correct ports when running the Docker container (read the documentation on
Docker Hub)

PostgreSQL listens on port 5432 inside the container by default. The `-p` flag maps the host port 5432 (`localhost:5432`) to the container port 5432. If the host port was in use you could map a different host port by running `-p 5450:5432`. 

### Find an appropriate Python package (Postgres adapter) that allows you to communicate with the
database server

`psycopg2-binary` will be used as the ProgresSQL adaptor. It follows all the requirements for a high-quality package as outlined in Milestone 2.
https://pypi.org/project/psycopg2/

Added to the requirements.txt then installed with
```bash
pip install -r requirements.txt
```

### Write a little python script

In /src I created “postgres_jokes.py” that connects to the database server using "localhost:port”, creates a database called "ms3_jokes”, creates a Table called "jokes". The table should have an attribute "ID" which is it's primary key and another Attribute "JOKE" of character type "TEXT”, inserts your favorite joke into that table, selects your favorite joke (now in the database), and fetches it from the database and prints your favorite joke. 

Checked this script locally 
```bash
Python3 postgres_jokes.py
```
Which resulted in

```
Created database 'ms3_jokes'
Created table 'jokes'
Inserted joke with ID 1
Your joke from database: How much did the pirate pay to get his ears pierced? A buccaneer!
```
### Download the pgADMIN Tool (https://www.pgadmin.org/download/). It also exists as a Docker Image :).
Connect to your running PostgreSQL Database. Can you see your database and table?

Downloaded pgADMIN with
```bash
docker pull dpage/pgadmin4:latest
```

Then ran the pgADMIN container with
```bash
docker run --name pgadmin --network pg-net \
  -p 8080:80 \
  -e PGADMIN_DEFAULT_EMAIL=admin@example.com \
  -e PGADMIN_DEFAULT_PASSWORD=admin \
  -d dpage/pgadmin4
```

Opened with http:/localhost:8080 and added a new server. In the connection section I set all values according to the my-postgres container. After this I was able to see my database and joke.

###  If you stopped and deleted the Docker container running the database and restarted it. Would your joke still be in the database? Why or why not?

If you only stop and start the same container, then yes the joke is still there because the data directory inside that container is preserved. But if you delete the container without a named volume then everything inside it is deleted including the PostgreSQL database and tables.

### Issues and How They Were Solved

Originally when I tried to set up pgAdmin I didn't add the network until later and wound up creating some unnecessary containers. I corrected this by searching the error messages that came up in pgAdmin when I tried to add a new server and then I removed all of my docker containers and networks and started fresh, which included rerunning the python script.

## Task 3: Image Storage in PostgreSQL

Task 3 explores how to store image data in a relational database like PostgreSQL, addressing the "impedance mismatch" problem between object-oriented representations (NumPy arrays) and relational tables.

### Question 1: Dataset Structure

**How is the MNIST data structured?**

The MNIST dataset contains 70,000 28×28 pixel grayscale images. They can be loaded as NumPy arrays with Keras:

```python
from tensorflow import keras
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
```

**Dataset characteristics:**
- Training samples: 60,000 images
- Test samples: 10,000 images
- Image shape: 28×28×1 (height × width × channels)
- Pixel values: Normalized float32 in range [0, 1]
- Labels: Integer 0-9 representing digits

### Question 2: Database Table Design

**How would you define relational database tables to save your data? What kind of data types could you use?**

To store MNIST images in PostgreSQL, we use binary serialization. Images need to be converted to BYTEA (binary data) using serialization, then reversed on retrieval. NumPy's serialization (`np.save()`) provides an effective way to transform image arrays into bytes.

#### Basic Table Schema

| Attribute | Data Type | Purpose |
|-----------|-----------|---------|
| `id` | `SERIAL PRIMARY KEY` | Auto-incrementing unique identifier for each image |
| `image_data` | `BYTEA` | Binary storage of serialized NumPy array (28×28 image) |
| `true_label` | `INTEGER` | The actual digit (0-9) in the image |
| `image_shape` | `VARCHAR(50)` | Shape metadata for validation (e.g., "(28, 28, 1)") |
| `created_at` | `TIMESTAMP` | Timestamp when the image was inserted |

**Why BYTEA?**
This approach is more efficient than flattening into 784 individual columns or storing as text.

### Question 3: Additional Attributes for Querying

**What additional attributes might make sense to easily query your data (e.g., find all pictures of digit 7)?**

Additional attributes that may assist in query may be:

| *Attribute*  | *Data Type*  | *Purpose*  | *Example* |
|---|---|---|---|
| dataset_split  | TEXT  | Indicate if image is training or test set | Find only train images |
| label | INTEGER  | 0-9 digit class | Find all 7s |
| image_index  | INTEGER | Original dataset index  | Track original dataset position |

### The Impedance Mismatch Problem

The "impedance mismatch" problem occurs when trying to store object-oriented or array-based data (like images in Python) into relational databases (which use tables with rows and columns). This is similar to the serialization/deserialization process used when transmitting data over networks.

#### How to Represent/Transform Image Data for Relational Databases

**Problem:**
- Python/NumPy represents images as multi-dimensional arrays (e.g., 28×28×1 for MNIST)
- PostgreSQL stores data in tabular format with specific data types
- Direct storage of NumPy arrays is not possible

**Solution: Binary Serialization**

We transform image data using the following approach:

1. **Serialization (Python → Database):**
   ```python
   def serialize_image(image_array: np.ndarray) -> bytes:
       buffer = io.BytesIO()
       np.save(buffer, image_array)
       return buffer.getvalue()
   ```
   - Convert NumPy array to bytes using NumPy's built-in serialization
   - Store in PostgreSQL `BYTEA` (binary data) column

2. **Deserialization (Database → Python):**
   ```python
   def deserialize_image(image_bytes: bytes) -> np.ndarray:
       buffer = io.BytesIO(image_bytes)
       return np.load(buffer, allow_pickle=False)
   ```
   - Retrieve bytes from database
   - Reconstruct NumPy array using NumPy's load function

**Alternative Approaches:**
- **JSON encoding:** Convert array to JSON (less efficient for large arrays)
- **Base64 encoding:** Encode bytes as text but this is less efficient
- **External file storage:** Store images as files, reference paths in database
- **Specialized databases:** Use PostgreSQL extensions like `cube` or dedicated image databases


**How Data is Loaded:**
```python
from tensorflow import keras
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
```
- Downloaded automatically from Keras
- Cached locally for subsequent runs
- Preprocessed: normalized and reshaped

### Database Table Design

#### PostgreSQL Data Types Used

| Data Type | Usage | Rationale |
|-----------|-------|-----------|
| `SERIAL` | Primary keys (`id`) | Auto-incrementing integer, ensures uniqueness |
| `BYTEA` | Image data, probability arrays | Binary data storage for serialized NumPy arrays |
| `INTEGER` | Labels, predictions | Digit classification (0-9) |
| `REAL` | Confidence scores | Floating-point for probability values |
| `VARCHAR(50)` | Image shape metadata | Store shape as string for validation |
| `TIMESTAMP` | Creation timestamps | Track when data was inserted |

#### Table Attributes for Querying

**Indexes for Performance:**
```sql
-- Find all images of a specific digit
CREATE INDEX idx_input_data_label ON input_data(true_label);

-- Find predictions for a specific digit
CREATE INDEX idx_predictions_label ON predictions(predicted_label);

-- Link predictions to input data efficiently
CREATE INDEX idx_predictions_input ON predictions(input_data_id);
```
- **Behavior:** Waits until the container exits with status code 0
- **Use Case:** Initialization scripts, database migrations
- **Example:** One-time setup containers

**Query Examples:**
```sql
-- Find all images of the digit 7
SELECT * FROM input_data WHERE true_label = 7;

-- Find all incorrect predictions
SELECT p.*, i.true_label
FROM predictions p
JOIN input_data i ON p.input_data_id = i.id
WHERE p.predicted_label != i.true_label;

-- Find predictions with low confidence
SELECT * FROM predictions WHERE confidence < 0.8;
```
 
## Task 4: Multi-Docker Container Application

### Architecture Overview

The application consists of two containerized services:

1. **PostgreSQL Service (`db`):** Stateful database server
2. **Python Application Service (`app`):** Stateless prediction service

### Database Schema

#### Entity-Relationship Diagram

```
┌─────────────────────────────────────┐
│          input_data                 │
├─────────────────────────────────────┤
│  id (SERIAL, PRIMARY KEY)           |
│  image_data (BYTEA)                 │
│  image_data (BYTEA)                 │
│  true_label (INTEGER)               │
│  image_shape (VARCHAR)              │
│  created_at (TIMESTAMP)             │
└─────────────────┬───────────────────┘
                  │
                  │ 1
                  │
                  │ links to (Foreign Key)
                  │
                  │ *
                  ▼
┌─────────────────────────────────────┐
│         predictions                 │
├─────────────────────────────────────┤
│ id (SERIAL, PRIMARY KEY)            |
│ input_data_id (INTEGER, FK)         │
│ predicted_label (INTEGER)           │
│ confidence (REAL)                   │
│ prediction_probabilities (BYTEA)    │
│ created_at (TIMESTAMP)              │
└─────────────────────────────────────┘
```

#### Relationship: One-to-Many
- One `input_data` record can have many `predictions`
- Each `prediction` must reference exactly one `input_data` record
- Foreign key constraint: `predictions.input_data_id` → `input_data.id`
- Cascade delete: If input data is deleted, associated predictions are also deleted

#### Table: `input_data`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | SERIAL | PRIMARY KEY | Auto-incrementing unique identifier |
| `image_data` | BYTEA | NOT NULL | Serialized NumPy array of image |
| `true_label` | INTEGER | NOT NULL | Actual digit (0-9) in the image |
| `image_shape` | VARCHAR(50) | NOT NULL | Shape metadata for validation |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Insertion timestamp |

**Purpose:** Store input images and their ground truth labels

#### Table: `predictions`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | SERIAL | PRIMARY KEY | Auto-incrementing unique identifier |
| `input_data_id` | INTEGER | FOREIGN KEY → input_data(id) | Links to input image |
| `predicted_label` | INTEGER | NOT NULL | Predicted digit (0-9) |
| `confidence` | REAL | NOT NULL | Max probability (confidence score) |
| `prediction_probabilities` | BYTEA | NOT NULL | Full probability distribution (10 values) |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Prediction timestamp |

**Purpose:** Store model predictions with references to input data

### Docker Volumes

#### `postgres_data` Volume
**Purpose:** Persist PostgreSQL database
```yaml
volumes:
  postgres_data:/var/lib/postgresql/data
```
- **Persistence:** Data survives container restarts/removals
- **Location:** PostgreSQL's data directory
- **Contents:** Database files, tables, indexes

#### `model_data` Volume
**Purpose:** Share trained neural network model
```yaml
volumes:
  model_data:/app/models
```
- **Persistence:** Model survives container restarts
- **Sharing:** Can be pre-loaded externally or trained separately
- **Format:** Keras SavedModel format (`.keras` or `.h5`)

### Application Workflow

The Python application (`db_app.py`) executes the following steps:

1. **Wait for Database:** Retries connection until PostgreSQL is ready
2. **Initialize Database:** Creates `milestone_3` database if it doesn't exist
3. **Create Tables:** Creates `input_data` and `predictions` tables
4. **Load Model:** Loads trained neural network from volume
5. **Load Sample:** Loads one MNIST test image
6. **Serialize & Store:** Converts image to bytes and inserts into `input_data`
7. **Retrieve & Deserialize:** Loads image back from database, converts to NumPy array
8. **Verify:** Confirms deserialization matches original image
9. **Predict:** Runs neural network inference on retrieved image
10. **Store Prediction:** Inserts prediction into `predictions` table with foreign key


## Docker Compose Startup Order

### The Startup Order Problem

**Issue:** Docker Compose starts containers in parallel by default. The Python app might try to connect to the database before PostgreSQL is ready, causing a crash.

### Three `condition` Options

Docker Compose provides three dependency conditions:

#### 1. `service_started` (Default)
```yaml
depends_on:
  db:
    condition: service_started
```
- **Behavior:** Waits until the container starts
- **Problem:** Container "started" ≠ service "ready"
- **Risk:** PostgreSQL container might be running but not accepting connections
- **Use Case:** When dependent service has no health check

#### 2. `service_healthy`
```yaml
depends_on:
  db:
    condition: service_healthy
```
- **Behavior:** Waits until the health check passes
- **Requirement:** Database must have a `healthcheck` defined
- **Advantage:** Ensures service is actually ready to accept connections
- **Use Case:** **Recommended for databases and critical services**

#### 3. `service_completed_successfully`
```yaml
depends_on:
  init:
    condition: service_completed_successfully
```
- **Behavior:** Waits until the container exits with status code 0
- **Use Case:** Initialization scripts, database migrations
- **Example:** One-time setup containers

### Our Implementation: `service_healthy`

**Database Health Check:**
```yaml
db:
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U postgres"]
    interval: 5s
    timeout: 5s
    retries: 5
```

**How it works:**
1. PostgreSQL container starts
2. Every 5 seconds, Docker runs `pg_isready -U postgres`
3. If command succeeds, health check passes
4. After 5 consecutive failures (5s × 5 = 25s), container is marked unhealthy
5. Only after health check passes does the `app` container start

**Application Dependency:**
```yaml
app:
  depends_on:
    db:
      condition: service_healthy
```

**Benefits:**
- Prevents race conditions
- No manual connection retry logic needed (though we include it for extra safety)
- Explicit dependency declaration
- Self-documenting service requirements



### Additional Safety: Application-Level Retry

Even with `service_healthy`, we implement retry logic in `db_app.py`:

```python
def wait_for_db(max_retries: int = 30, retry_delay: int = 2):
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(...)
            return True
        except Exception:
            time.sleep(retry_delay)
    raise Exception("Database not ready")
```


---

## Running the Application

### Prerequisites

1. **Train and save a model:**
   ```bash
   # Option 1: Train locally first
   python main.py

   # Then copy model to volume location
   docker volume create milestone3_model_data
   docker run --rm -v $(pwd)/models:/src -v milestone3_model_data:/dest \
     alpine sh -c "cp /src/mnist_model.keras /dest/"
   ```

2. **Or use the included model training:**
   The Dockerfile can be extended to train the model during build.

### Start the Application

```bash
# Start both services
docker-compose up

# Or run in detached mode
docker-compose up -d
```

### Verify the Database

**Option 1: Using psql**
```bash
docker exec -it milestone3_postgres psql -U postgres -d milestone_3

# Query input data
SELECT id, true_label, created_at FROM input_data;

# Query predictions with join
SELECT p.id, p.predicted_label, p.confidence, i.true_label
FROM predictions p
JOIN input_data i ON p.input_data_id = i.id;
```

**Option 2: Using pgAdmin**
1. Install pgAdmin or use web version
2. Connect to `localhost:5432`
3. Username: `postgres`, Password: `postgres`
4. Navigate to `milestone_3` database
5. Inspect `input_data` and `predictions` tables

### Clean Up

```bash
# Stop containers
docker-compose down

# Stop and remove volumes (fresh start)
docker-compose down -v
```

**Note:** To start fresh, you must remove the `postgres_data` volume. Otherwise, the database will persist and the application will detect it already exists.


## Key Learnings

1. **Impedance Mismatch:** Binary serialization (NumPy → bytes → BYTEA) solves the object-relational mapping problem for images

2. **Data Persistence:** Docker volumes provide stateful storage for stateless containers

3. **Service Dependencies:** Health checks prevent race conditions in multi-container applications

4. **Database Design:** Foreign keys maintain referential integrity between predictions and input data

5. **Microservices:** Separation of database and application services enables independent scaling and deployment

## Additional questions:

## What is an SQL Injection Attack and how can you protect yourself?

A **SQL Injection Attack** is a malicious attack where an attacker injects harmful SQL code into an input field (for example a login form or search field) in order to manipulate the database.

Instead of normal input, an attacker could submit something like:

' OR 1=1; --

or even destructive commands such as:

DROP DATABASE;

If the application directly concatenates user input into SQL queries, this code may be executed by the database. This can lead to unauthorized access, data leakage, or complete data loss and is therefore a **critical security vulnerability**.

**Protection measures:**
- Use **prepared statements / parameterized queries** so user input is treated as data, not executable SQL
- Apply **input validation and type enforcement** (e.g. only integers for age fields)
- **Never allow users to directly communicate with the database**; always use a backend server
- Use **ORM frameworks** (e.g. SQLAlchemy, Hibernate)
- Apply the **principle of least privilege** for database users
- Avoid **dynamic SQL string concatenation**

## What is ACID in the context of SQL Databases?

**ACID** describes four properties that guarantee reliable database transactions:

- **Atomicity**  
  A transaction is all-or-nothing. Either all operations succeed or none are applied.

- **Consistency**  
  A transaction moves the database from one valid state to another while respecting all constraints and rules.

- **Isolation**  
  Concurrent transactions do not interfere with each other; results are as if transactions were executed sequentially.

- **Durability**  
  Once a transaction is committed, the changes are permanently stored, even in case of crashes or power failures.

## What is the difference between a Relational Database and a Document Store?  
### In which scenarios would you use which technology?

### Relational Databases
- Enforce a **fixed schema** (tables, columns, data types)
- Store data in **rows and columns**
- Support **joins**, constraints, and complex queries
- Provide strong **ACID guarantees**

**Typical use cases:**
- Structured, tabular data
- Financial and accounting systems
- Applications requiring strong consistency and relationships

Examples: PostgreSQL, MySQL, SQL Server

---

### Document Stores
- Use **flexible or schema-less documents** (typically JSON)
- Support nested data structures
- Little or no join support
- Easier horizontal scaling

**Typical use cases:**
- Semi-structured or evolving data
- JSON-heavy applications
- User profiles, logs, content management systems

Examples: MongoDB, CouchDB

## What is a SQL Join Operation? What other common SQL statements exist?

A **SQL Join** operation combines rows from two or more tables using a related column (usually a foreign key).

**Common join types:**
- **INNER JOIN** – returns only matching rows from both tables
- **LEFT JOIN** – returns all rows from the left table and matching rows from the right
- **RIGHT JOIN** – returns all rows from the right table and matching rows from the left
- **FULL OUTER JOIN** – returns all rows from both tables, matched where possible

**Other common SQL statements:**
- `SELECT` – retrieve data
- `INSERT` – add new rows
- `UPDATE` – modify existing rows
- `DELETE` – remove rows
- `WHERE` – filter records
- `GROUP BY` – aggregate data
- `HAVING` – filter aggregated results
- `ORDER BY` – sort query results
- `CREATE`, `ALTER`, `DROP` – manage database structures
- `INDEX` – improve query performance
- `TRANSACTION`, `COMMIT`, `ROLLBACK` – control transactions


