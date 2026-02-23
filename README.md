# AAI 540 — Group 6: Cat Pose Estimation & Classification Pipeline

> **Course:** AAI 540 — Machine Learning Operations  
> **Team:** Group 6  
> **Platform:** AWS SageMaker

---

## Overview

This project implements an end-to-end MLOps pipeline for **human pose estimation and action classification** using AWS cloud services. The pipeline covers everything from raw data ingestion through model training, monitoring, and live endpoint deployment — following production-grade MLOps best practices.

The system combines two complementary models:
- A **keypoint regression model** that predicts human body keypoint coordinates from image data
- A **classification model** that maps those keypoints (or raw features) to action/pose class labels

---

## Repository Structure

```
AAI_540_Grp_6/
├── src/                              # Supporting Python modules and scripts
├── 1_Setup.ipynb                     # AWS environment setup (IAM, S3, SageMaker config)
├── 2_Ingestion.ipynb                 # Data ingestion and S3 upload
├── 3_Create_Athena_EDA.ipynb         # Athena table creation and exploratory data analysis
├── 4_Feature_store.ipynb             # SageMaker Feature Store ingestion and management
├── 5_Classification_Model.ipynb      # Training and evaluation of the classification model
├── 5-1_Keypoint_Regression_Model.ipynb  # Training and evaluation of the keypoint regression model
├── 6_Monitoring.ipynb                # SageMaker Model Monitor setup and drift detection
├── 7_Endpoint_Demo.ipynb             # Live endpoint deployment and inference demo
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## Pipeline Walkthrough

### 1. Setup (`1_Setup.ipynb`)
Configures the AWS environment: IAM roles, S3 bucket creation, and SageMaker session initialization. Run this notebook first to establish all prerequisites.

### 2. Data Ingestion (`2_Ingestion.ipynb`)
Loads the raw dataset (pose/keypoint images and annotations) and uploads it to S3 for downstream processing.

### 3. Athena EDA (`3_Create_Athena_EDA.ipynb`)
Creates an AWS Glue/Athena table over the S3 data and performs exploratory data analysis — querying distributions, inspecting label balance, and visualizing keypoint statistics.

### 4. Feature Store (`4_Feature_store.ipynb`)
Ingests processed features into the **SageMaker Feature Store**, enabling reproducible, versioned feature retrieval for model training and serving.

### 5a. Classification Model (`5_Classification_Model.ipynb`)
Trains a classifier to predict action/pose categories. Covers data preparation, training job configuration, hyperparameter settings, and evaluation metrics.

### 5b. Keypoint Regression Model (`5-1_Keypoint_Regression_Model.ipynb`)
Trains a regression model to predict (x, y) coordinates for human body keypoints. Covers model architecture, training, and evaluation (e.g., RMSE, normalized error).

### 6. Monitoring (`6_Monitoring.ipynb`)
Sets up **SageMaker Model Monitor** to capture inference traffic, establish data quality baselines, and schedule drift detection jobs against the deployed endpoint.

### 7. Endpoint Demo (`7_Endpoint_Demo.ipynb`)
Deploys the trained model(s) to a SageMaker real-time endpoint and demonstrates end-to-end inference — sending sample inputs and visualizing predictions.

---

## Getting Started

### Prerequisites

- Python 3.8+
- An AWS account with SageMaker, S3, Glue, and Athena access
- Appropriate IAM role with SageMaker execution permissions
- AWS CLI configured locally (or run notebooks inside SageMaker Studio)

### Installation

```bash
git clone https://github.com/lpam-sd/AAI_540_Grp_6.git
cd AAI_540_Grp_6
pip install -r requirements.txt
```

### Running the Pipeline

Execute the notebooks **in order**:

```
1_Setup → 2_Ingestion → 3_Create_Athena_EDA → 4_Feature_store
       → 5_Classification_Model / 5-1_Keypoint_Regression_Model
       → 6_Monitoring → 7_Endpoint_Demo
```

> **Tip:** The notebooks are designed to be run inside **Amazon SageMaker Studio** for seamless AWS integration. Running locally requires valid AWS credentials and may incur costs for SageMaker training jobs and endpoints.

---

## AWS Services Used

| Service | Purpose |
|---|---|
| Amazon S3 | Raw data and artifact storage |
| AWS Glue + Amazon Athena | Metadata catalog and SQL-based EDA |
| SageMaker Feature Store | Centralized, versioned feature management |
| SageMaker Training Jobs | Managed model training (classification & regression) |
| SageMaker Endpoints | Real-time model serving |
| SageMaker Model Monitor | Data quality and drift monitoring |

---

## Models

| Model | Type | Task |
|---|---|---|
| Classification Model | Multi-class classifier | Predict pose/action label from features |
| Keypoint Regression Model | Coordinate regression | Predict (x, y) body keypoint locations |

---

## Team

**AAI 540 — Group 6**  
University of San Diego — Applied Artificial Intelligence Program

---

## License

This project was created for academic purposes as part of the AAI 540 course. Please contact the contributors before reusing or redistributing any content.