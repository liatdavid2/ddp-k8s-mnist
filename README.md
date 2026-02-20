# Distributed PyTorch Training on Kubernetes (DDP + torchrun)

## Overview

This project demonstrates real multi-node Distributed Data Parallel (DDP) training using:

* PyTorch
* torchrun
* Kubernetes Jobs
* Headless Service for rendezvous
* Minikube (local cluster)

Two separate Kubernetes pods run one process each, synchronize gradients using AllReduce, and train a CNN on MNIST.

This simulates how distributed training works in production ML platforms.

---
# Architecture

```
Kubernetes Cluster
│
├── Pod 1: ddp-master
│     torchrun
│     rank = 0
│     trains model
│
├── Pod 2: ddp-worker
      torchrun
      rank = 1
      trains model
```

During every training batch:

```
Pod 0 gradients ──┐
                  ├── AllReduce ──> synchronized model
Pod 1 gradients ──┘
```

Both pods compute gradients independently.
PyTorch performs an AllReduce operation to synchronize gradients across pods.
Each pod updates identical model weights.



-------
## Project Structure

```text
ddp-k8s-mnist/
│
├── docker/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .dockerignore
│
├── k8s/
│   ├── ddp-master-job.yaml
│   ├── ddp-worker-job.yaml
│   └── headless-svc.yaml
│
├── src/
│   └── train.py
│
├── .gitignore
├── README.md
├── Commands
└── kubectl
```
---

# What Was Implemented

* Containerized PyTorch training script
* Multi-pod distributed execution using torchrun
* Static rendezvous configuration
* Headless Kubernetes Service for DNS-based coordination
* Gradient synchronization using DistributedDataParallel
* Checkpoint saving after training

---

# Kubernetes Components

## 1. Headless Service (for rendezvous)

Allows pods to discover each other via DNS:

```
ddp-master-svc:29500
```

## 2. Master Job

Runs:

```
torchrun \
  --nnodes=2 \
  --node_rank=0 \
  --rdzv_backend=static \
  --rdzv_endpoint=ddp-master-svc:29500 \
  /app/src/train.py
```

## 3. Worker Job

Runs:

```
torchrun \
  --nnodes=2 \
  --node_rank=1 \
  --rdzv_backend=static \
  --rdzv_endpoint=ddp-master-svc:29500 \
  /app/src/train.py
```

---

# How It Works Internally

1. Each pod runs torchrun.
2. torchrun assigns:

   * RANK (0 or 1)
   * WORLD_SIZE=2
   * MASTER_ADDR
3. `dist.init_process_group("gloo")` initializes distributed communication.
4. Each pod trains independently on its data shard.
5. After backward pass:

   * PyTorch performs AllReduce.
   * Gradients are averaged.
6. Model weights remain synchronized across pods.

---

# Commands That Worked

## Build Docker image directly inside Minikube

```
minikube image build -t ddp-mnist:0.1 -f docker/Dockerfile .
```

## Deploy all Kubernetes resources

```
kubectl apply -f k8s/
```

## Verify pods

```
kubectl get pods
```

## View logs

```
kubectl logs -f <pod-name>
```

Example:

```
kubectl logs -f ddp-master-2b26w
```

## Delete and redeploy

```
kubectl delete -f k8s/
kubectl apply -f k8s/
```

---

# Example Output

```
TRAIN STARTED
RANK: 0
WORLD_SIZE: 2
[INIT] world_size=2 backend=gloo

[EPOCH 1/2] loss=0.3660 samples/sec=641.3
[EPOCH 2/2] loss=0.0757 samples/sec=663.5
[DONE] Saved checkpoint: /outputs/mnist_cnn_ddp.pt
```

This confirms:

* Two ranks participated
* Training executed successfully
* Gradients synchronized
* Checkpoint saved

---

## Distributed Training Results (Kubernetes + PyTorch DDP)

This model was trained using **PyTorch DistributedDataParallel (DDP)** across multiple Kubernetes pods in a containerized environment.

---

### Training Performance

| Epoch | Loss   | Throughput (samples/sec) |
|-------|--------|--------------------------|
| 1/2   | 0.3660 | 592.6                    |
| 2/2   | 0.0757 | 798.0                    |

The model converged rapidly while maintaining high distributed throughput due to synchronized gradient updates across pods.

---

### Evaluation Metrics

| Metric     | Value  |
|------------|--------|
| Accuracy   | 0.9824 |
| Precision  | 0.9825 |
| Recall     | 0.9824 |
| F1-score   | 0.9824 |

---

### Classification Report

```

```
          precision    recall  f1-score   support

       0       0.97      1.00      0.98       451
       1       0.99      0.99      0.99       591
       2       0.97      0.99      0.98       501
       3       0.99      0.99      0.99       511
       4       0.99      0.98      0.98       480
       5       0.98      0.98      0.98       458
       6       0.99      0.97      0.98       499
       7       0.98      0.98      0.98       519
       8       0.97      0.98      0.97       466
       9       0.99      0.96      0.98       524

accuracy                           0.98      5000
```

macro avg       0.98      0.98      0.98      5000
weighted avg       0.98      0.98      0.98      5000

```

---

### Model Artifact

The trained checkpoint is saved for reproducibility and deployment:

```

/outputs/mnist_cnn_ddp.pt

```

---




