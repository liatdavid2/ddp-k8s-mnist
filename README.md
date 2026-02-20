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

[EPOCH 1/2] loss=0.3272 samples/sec=641.3
[EPOCH 2/2] loss=0.0712 samples/sec=663.5
[DONE] Saved checkpoint: /outputs/mnist_cnn_ddp.pt
```

This confirms:

* Two ranks participated
* Training executed successfully
* Gradients synchronized
* Checkpoint saved

---

# Key Takeaways

* Demonstrates real distributed training across multiple Kubernetes pods
* Uses production-style rendezvous and static backend
* Shows gradient synchronization via AllReduce
* Reproduces ML platform distributed training architecture


