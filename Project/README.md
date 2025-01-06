# CelebA Dataset Visualization Tool

## Introduction
This project is an interactive visualization and analysis tool for the **CelebA dataset**, a large-scale dataset containing over 200,000 celebrity images annotated with facial attributes. The CelebA dataset is widely used in computer vision and machine learning, especially for tasks like facial recognition, attribute prediction, and facial similarity detection. Our tool focuses on a subset of this dataset, consisting of **30,012 images**, to provide insightful analysis and functionality.

### What is the CelebA Dataset?
The **CelebFaces Attributes Dataset (CelebA)** includes:
- Images of celebrities annotated with **39 binary facial attributes** (e.g., "bald", "male", "smiling").
- High-dimensional embeddings (512 dimensions) representing each image in latent space.

This application enables users to interact with the dataset through data analysis, similarity matching, clustering, and embedding exploration.

---

## Use Cases
Here are some practical examples of how this tool can be applied:

1. **Facial Recognition:**
   - Analyze and cluster facial embeddings for improved facial recognition models.
   - Discover patterns and relationships between facial attributes.

2. **Dating Apps and Social Platforms:**
   - Use similarity matching to find visually similar individuals, an essential feature for dating apps and friend-matching platforms.

3. **Data Exploration and Visualization:**
   - Conduct exploratory data analysis on attributes and embeddings for research or commercial purposes.
   - Visualize relationships between features and explore the dataset interactively.

---

## Project Structure

Below is an overview of the project's folder and file structure:

### Root Directory
- **`README.md`**: Provides an overview of the project.
- **`requirements.txt`**: Contains Python dependencies for running the application.

### `pages/`
Contains the main scripts for the Dash app:
- **`home.py`**: Displays an overview of the dataset with randomly sampled images shown every 3 seconds.
- **`data_analysis.py`**: Implements tools for analyzing dataset attributes and visualizing trends with static and dynamic statistics.
- **`similarity_embeddings.py`**: Enables facial similarity matching with customizable distance metrics and K-nearest neighbors search.
- **`embeddings.py`**: Allows for advanced embedding exploration, including data filtering, clustering, and dimensionality reduction.

### `PDF/`
Includes reference and supporting documents:
- **`Evaluation-Grid.pdf`**: A guide for evaluating the tool's features.
- **`Guideline-Project.pdf`**: Comprehensive guidelines for the project.
- **`Report-Project.pdf`**: Detailed project report including objectives, implementation, and results.

### `utils/`
Contains utility scripts used throughout the project:
- **`app_utils.py`**: Provides helper functions for navigation and app-level operations.
- **`clustering_utils.py`**: Implements clustering algorithms (DBScan, Agglomerative Clustering, K-Means).
- **`data_analysis_utils.py`**: Contains functions for attribute analysis and data visualization.
- **`embeddings_utils.py`**: Handles embedding manipulation and similarity computations.
- **`similarity_embeddings_utils.py`**: Offers functions for similarity searches using customizable distance metrics.

---

## Features and Functionality

### 1. Home Page
- **Introduction to the Dataset**: Presents a brief overview of the CelebA dataset.
- **Image Sampler**: Randomly samples and displays three images from the dataset every 3 seconds to showcase its diversity.
- **Navigation**: Features buttons to access:
  - Data Analysis
  - Similarity Search
  - Embedding Exploration

---

### 2. Similarity Matching
The **Similarity Matching Page** allows users to:
- **Search for Similar Faces**:
  - Upload or select an image from the dataset.
  - Use one of two distance metrics:
    - **Euclidean Distance**
    - **Cosine Similarity**
  - Specify the number of nearest neighbors to retrieve (K).

---

### 3. Data Analysis
The **Data Analysis Page** provides a detailed exploration of the CelebA dataset:
- **Static and Dynamic Statistics**: Displays key metrics such as:
  - Total number of images
  - Embedding's dimension
  - Attribute presence / absence with adjustable thresholds.
  - (In)Frequent attributes.
  - ...
- **Visualizations**: Includes five interactive charts:
  1. **Correlation Matrix**: Highlights linear relationships between features.
  2. **Co-occurrence Matrix**: Shows how often features appear together.
  3. **Facial Feature Distributions**: Displays the balance of facial attributes in the dataset.
  4. **Embedding Distributions**: Plots the range and distribution of embedding values.
  5. **Feature-Embedding Correlation**: Analyzes the relationship between features and embeddings.

---

### 4. Embedding Exploration
The **Embedding Exploration Page** offers advanced tools for understanding and visualizing facial embeddings:
1. **Weighted Similarity Search**:
   - Assign weights to individual facial features.
   - Generate a synthetic embedding based on the weighted average.
   - Find the most similar image to this embedding.
   
2. **Data Filtering**:
   - Use **AND** or **OR** filters to refine data subsets based on facial attributes.
   
3. **Clustering**:
   - Apply clustering algorithms to group similar images:
     - **DBScan**
     - **K-Means**
     - **Agglomerative Clustering**
   - Customize key parameters like the number of clusters or density thresholds.
   
4. **Dimensionality Reduction (DR)**:
   - Reduce embeddings to 2D space for visualization using:
     - **PCA**
     - **t-SNE**
     - **MDS**
     - **LLE**
     - **ISOMAP**
   - Modify algorithm-specific parameters for tailored results.

---

## Setup Instructions

To get started with the project, follow these steps:

1. **Install Dependencies**
   Make sure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare the Dataset**
   - Create a folder named `Datas`.
   - Download the CelebA dataset from [here](https://www.kaggle.com/jessicali9530/celeba-dataset).
   - Rename the dataset file to `celeba_buffalo_l.csv` and place it inside the `Datas` folder.

3. **Prepare the Assets**
   - Create a folder named `assets`.
   - Place all the images from the CelebA dataset into this folder.

4. **Run the Application**
   Start the Dash app by running:
   ```bash
   python app.py
   ```

---

For questions or contributions, feel free to create an issue or submit a pull request.