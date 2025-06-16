# BERT Genre Classification MLOps Pipeline

MLOps pipeline for training and deploying BERT-based text classification models using AWS SageMaker, MLflow, and automated CI/CD workflows.

## 🎯 Project Overview

This pipeline implements an end-to-end machine learning operations (MLOps) solution for text genre classification using BERT models. It features:

- **Automated Training**: SageMaker-based distributed training with hyperparameter optimization
- **Model Versioning**: MLflow integration for experiment tracking and model registry
- **CI/CD Pipeline**: GitHub Actions workflows for automated building, training, and deployment
- **Infrastructure as Code**: CloudFormation templates for inference
- **Production Deployment**: EC2-based LitServe inference API
- **Quality Assurance**: Automated smoke testing and model validation


## 🚀 Quick Start

### Prerequisites

- AWS Account with appropriate permissions
- GitHub repository with Actions enabled
- Python 3.11+

### Initial Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Configure GitHub Secrets**
   ```
   AWS_ACCESS_KEY_ID          # AWS access key
   AWS_SECRET_ACCESS_KEY      # AWS secret key
   AWS_REGION                 # AWS region (e.g., us-east-1)
   SAGEMAKER_EXECUTION_ROLE   # ARN of SageMaker execution role
   MLFLOW_TRACKING_SERVER_ARN # MLflow tracking server ARN
   S3_BUCKET_NAME            # S3 bucket for artifacts
   VPC_ID                    # VPC ID for EC2 deployment
   SUBNET_ID                 # Subnet ID for EC2 deployment
   EC2_SSH_KEY               # SSH private key for EC2 access
   GH_TOKEN                  # GitHub token for tagging
   ```

3. **Deploy Infrastructure**
   ```bash
   # Deploy EC2 infrastructure
   # Trigger the "Deploy EC2 Instance" workflow in GitHub Actions
   # Select environment: staging or prod
   ```

4. **Train Model**
   ```bash
   # Trigger the "SageMaker Training Pipeline" workflow
   # Or push changes to train/ directory
   # Upload new training data to S3
   ```

5. **Deploy Model**
   ```bash
   # Trigger the "Deploy inference" workflow
   # Model will be deployed after successful smoke tests
   ```

## 📦 Pipeline Components

### 1. Training Pipeline (`train/`)

**Purpose**: Trains BERT models for text classification using SageMaker.

**Key Files**:
- `train/train.py`: Main training script with MLflow integration
- `train/Dockerfile.train`: Training container definition
- `train/requirements.txt`: Python dependencies

**Workflow**: `.github/workflows/build-training-image.yml`
- Triggers on push to `train/` directory
- Builds and pushes Docker image to ECR
- Tags with semantic versioning

### 2. Inference Pipeline (`infer/`)

**Purpose**: Serves trained models via REST API using LitServe.

**Key Files**:
- `infer/app/genre_classifier.py`: API implementation
- `infer/Dockerfile.infer`: Inference container definition
- `infer/tests/unit_tests.py`: Comprehensive unit tests
- `infer/tests/smoke_test.py`: End-to-end validation

**Workflow**: `.github/workflows/build-inference-image.yml`
- Builds optimized inference container
- Downloads model artifacts from MLflow
- Runs unit tests before building

### 3. Infrastructure (`cloudformation/`)

**Purpose**: Defines AWS infrastructure as code.

**Key Files**:
- `cloudformation/ec2-instance.yaml`: EC2 instance configuration
  - Includes CloudWatch agent for monitoring
  - Configures security groups and IAM roles
  - Supports both staging and production environments

**Features**:
- Auto-scaling ready
- CloudWatch monitoring enabled

### 4. Deployment Workflows

**Training Workflow** (`.github/workflows/launch_training.yml`):
```yaml
Inputs:
- instance_type: SageMaker instance type
- epochs: Number of training epochs
- use_spot: Use spot instances for cost savings
```
- model training can either be triggered manually
- otherwise it monitors an S3 bucket daily for new training data

### Deployment Workflow (`.github/workflows/deploy-inference.yml`):
1. Deploy to staging
2. Run smoke tests with proper environment variables
3. If successful, promote to production
4. Alias model as "champion" in MLflow

**Important**: The deployment workflow must set environment variables correctly for the smoke test:
```yaml
- name: Perform smoke testing
  env:
    HOST: ${{ env.PUBLIC_IP_STAGING }}  # Must match smoke test expectation
    MLFLOW_TRACKING_SERVER_ARN: ${{ secrets.MLFLOW_TRACKING_SERVER_ARN }}
    EC2_INSTANCE_ID: ${{ env.INSTANCE_ID_STAGING }}
    MODEL_NAME: "bert-genre-classifier"
    PUBLIC_IP_STAGING: ${{ env.PUBLIC_IP_STAGING }}  # Required by smoke test
    USERNAME: ubuntu
  run: |
    python infer/tests/smoke_test.py
```

## ⚙️ Configuration

### Model Configuration

**Supported Models**:
- bert-base-uncased (default)
- Any HuggingFace BERT variant

**Hyperparameters**:
```json
{
  "model-name": "bert-base-uncased",
  "max-length": 256,
  "num-train-epochs": 3,
  "per-device-train-batch-size": 8,
  "learning-rate": 2e-5,
  "weight-decay": 0.01,
  "warmup-steps": 500
}
```

## 📊 Usage Guide

### Training a New Model

1. **Prepare Training Data**
   Format: `ID ::: TITLE ::: GENRE ::: DESCRIPTION`
   ```
   1 ::: The Matrix ::: Action ::: A computer hacker learns...
   2 ::: The Notebook ::: Romance ::: A poor yet passionate...
   ```

2. **Upload to S3**
   ```bash
   aws s3 cp train_data.txt s3://<bucket>/training-data/
   ```

3. **Trigger Training**
   - Via GitHub Actions UI
   - Or push changes to `train/` directory

4. **Monitor Progress**
   - Check SageMaker console
   - View MLflow experiments
   - Review CloudWatch logs

### Deploying a Model

1. **Model Selection**
   - Models are automatically registered in MLflow
   - Latest model becomes "challenger"

2. **Staging Deployment**
   ```bash
   # Automatic via workflow
   # Deploys to staging EC2 instance
   ```

3. **Validation**
   - Smoke tests run automatically
   - Checks latency, accuracy, and resource usage
   - Results logged to MLflow

4. **Production Promotion**
   - If tests pass, model becomes "champion"
   - Automatic deployment to production

### API Usage

**Endpoint**: `http://<ec2-public-ip>:8000/predict`

**Request**:
```bash
curl -X POST http://<ec2-public-ip>:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": "A thrilling adventure movie about space exploration..."
  }'
```

**Response**:
```json
{
  "main_category": ["Adventure", "Sci-Fi", "Action"],
  "probabilities": [0.75, 0.15, 0.10]
}
```

## 🔍 Monitoring & Debugging

### CloudWatch Metrics

**System Metrics**:
- CPU utilization
- Memory usage
- Disk I/O
- Network traffic

**Application Metrics**:
- Request latency
- Success rate
- Model inference time

### Debugging Commands

**Check EC2 Status**:
```bash
# SSH into instance
ssh -i <key-file> ubuntu@<public-ip>

# Check Docker containers
docker ps

# View application logs
docker logs <container-id>

# Check CloudWatch agent
sudo systemctl status amazon-cloudwatch-agent
```

**SageMaker Training Logs**:
```bash
# View in AWS Console
# Or use AWS CLI
aws sagemaker describe-training-job --training-job-name <job-name>
```

## ⚠️ Current Limitations

This section outlines the current limitations of the MLOps pipeline

### 1. **No Manual Approval Gates**
- **Impact**: Models are automatically promoted without human review
- **Risk**: Potentially problematic models could reach production without oversight

### 2. **Public Endpoints Only**
- **Impact**: Inference API is exposed to the public internet without authentication
- **Security Risk**: 
  - No rate limiting
  - No authentication/authorization

### 3. **No High Availability or Load Balancing**
- **Impact**: Single point of failure for inference service
- **Current Architecture**:
  - One EC2 instance per environment (staging/prod)
  - No automatic failover
  - No load distribution
  - Manual scaling only
- **Risks**:
  - Service unavailable if instance fails
  - Cannot handle traffic spikes
  - No zero-downtime deployments
  - Potential for data loss during failures


### Common Issues

1. **Training Job Fails**
   - Check IAM role permissions
   - Verify S3 bucket access
   - Review training logs in CloudWatch
   - Ensure ECR repository exists
   - Validate training data format

2. **Deployment Fails**
   - Ensure EC2 instance is running
   - Check security group rules (port 8000 open)
   - Verify Docker daemon is active
   - Check environment variable configuration
   - Validate ECR image accessibility

3. **Smoke Tests Fail**
   - Review test logs in GitHub Actions
   - Check model performance metrics
   - Verify EC2 resource availability
   - Ensure correct environment variables:
     ```bash
     export PUBLIC_IP_STAGING=<staging-ip>
     export EC2_INSTANCE_ID=<instance-id>
     export MODEL_NAME="bert-genre-classifier"
     ```

4. **MLflow Connection Issues**
   - Verify MLFLOW_TRACKING_SERVER_ARN is correct
   - Check IAM permissions for SageMaker MLflow
   - Ensure VPC endpoints are configured
   - Test connection:
     ```python
     import mlflow
     mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_SERVER_ARN"])
     print(mlflow.list_experiments())
     ```

5. **Model Loading Errors**
   - Verify model artifacts downloaded correctly
   - Check file permissions in container
   - Validate label_mappings.json format
   - Ensure sufficient memory for model loading
