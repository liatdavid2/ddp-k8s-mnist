# Distributed PyTorch Training on Kubernetes (DDP) – MNIST Example

This project demonstrates how to train a PyTorch model inside Kubernetes using containerized workloads, with support for Distributed Data Parallel (DDP). It shows how machine learning training jobs can run as Kubernetes workloads using Docker images and scalable infrastructure.

The goal of this project is to illustrate core ML platform engineering concepts:

- Containerized ML training with Docker
- Running training workloads inside Kubernetes Pods
- Saving model checkpoints from containerized jobs
- Preparing the architecture for Distributed Data Parallel training
- Production-style training workflow similar to real ML platforms


# Architecture Overview

Developer Laptop
      │
      │ kubectl
      ▼
Kubernetes Cluster
      │
      ▼
Training Pod (Container)
      │
      ▼
PyTorch Training Script
      │
      ▼
Model Checkpoint Saved (/outputs)


Distributed Training Architecture (DDP):

              Kubernetes Cluster

        ┌─────────────────────┐
        │   ddp-master Pod    │  rank 0
        │   PyTorch process   │
        └─────────┬───────────┘
                  │ gradient synchronization
        ┌─────────▼───────────┐
        │   ddp-worker Pod    │  rank 1
        │   PyTorch process   │
        └─────────────────────┘


# Project Structure

ddp-k8s-mnist/

src/
    train.py                PyTorch training script with DDP support

docker/
    Dockerfile              Docker image definition for training

k8s/
    ddp-master.yaml         Kubernetes Job for master process
    ddp-worker.yaml         Kubernetes Job for worker process

outputs/                    Saved model checkpoints

requirements.txt
.gitignore
.dockerignore
README.md


# Training Workflow

Step 1 – Build Docker image

docker build -t mnist-ddp:latest -f docker/Dockerfile .

Or using minikube:

minikube image build -t mnist-ddp:latest -f docker/Dockerfile .


Step 2 – Run training inside Kubernetes

Apply the Kubernetes training jobs:

kubectl apply -f k8s/ddp-master.yaml
kubectl apply -f k8s/ddp-worker.yaml
kubectl apply -f k8s/ddp-service.yaml


Check pod status:

kubectl get pods


Example output:

NAME              READY   STATUS      RESTARTS   AGE
ddp-master-xxx    1/1     Running     0          30s
ddp-worker-xxx    1/1     Running     0          30s


View logs from master pod:

kubectl logs ddp-master-xxx


Example training output:

[INIT] world_size=2 backend=gloo
[RANK 0] host=ddp-master
[RANK 1] host=ddp-worker
[EPOCH 1/2] loss=0.1677 samples/sec=856.0
[EPOCH 2/2] loss=0.0469 samples/sec=1047.0
[DONE] Saved checkpoint: /outputs/mnist_cnn_ddp.pt


Step 3 – Verify saved model

kubectl exec <pod-name> -- ls -lh /outputs


Example output:

mnist_cnn_ddp.pt


# Training Script Features

The training script supports:

- CPU training
- Distributed Data Parallel (DDP)
- Multiple ranks (master / worker)
- Checkpoint saving
- Automatic dataset download
- Kubernetes-friendly execution

Core DDP initialization:

dist.init_process_group(
    backend="gloo",
    rank=rank,
    world_size=world_size
)


# Technologies Used

- Python
- PyTorch
- Distributed Data Parallel (DDP)
- Docker
- Kubernetes
- Minikube
- Containerized ML workloads

# Example Kubernetes Training Job

kubectl apply -f k8s/ddp-master.yaml

Kubernetes schedules the training workload automatically as a batch job.


# Example Training Result

[EPOCH 1/2] loss=0.1677
[EPOCH 2/2] loss=0.0469
[DONE] Saved checkpoint: /outputs/mnist_cnn_ddp.pt


# Future Improvements

- Multi-node distributed training with multiple pods
- GPU support
- Persistent volume for checkpoint storage
- Integration with MLflow for experiment tracking
- Automated retraining pipelines
- Kubernetes autoscaling


# Key Learning Outcomes

This project demonstrates:

- Running PyTorch training inside Kubernetes
- Building Docker images for ML workloads
- Using Kubernetes Jobs for batch training
- Preparing infrastructure for distributed training
- Saving and managing model checkpoints
- Understanding containerized ML workflows
