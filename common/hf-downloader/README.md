.# Working with Shared File System in Nebius

This guide provides instructions on how to work with a shared file system mounted to a managed kubernetes cluster in Nebius. A shared file system allows multiple pods and containers to access and share the same set of files.

## Prerequisites

Before you start, make sure you have the following:
- A Nebius account.
- The Nebius CLI installed and configured.
- A k8s cluster provisioned with attached shared FS.

## Overview

When working with shared FS mounted to Nebius mk8s cluster, please keep in mind the following:
- The shared FS is mounted to all nodes at "/mnt/data" by default (if specified during the cluster creation).
- When using the CSI driver with with shared FS, the driver will create a subdirectory per each Persistent Volume. Thus, to copy the files in a specific Persisten Volume, you need to allways reuse the same PV-PVC combination.

## Example: Downloading models from Hugging Face to the shared FS

When training or finetuning models, you often need to download large datasets and models to some local storage in order to avoid loosing time on repeated downloads. In this example, we will download model weights from Hugging Face to the shared FS.

### Create a Persistent Volume and Persistent Volume Claim

Create a [Persistent Volume (PV) and Persistent Volume Claim (PVC) pair](../shared-filesystem-mount/README.md) to mount the shared FS.

### Create a Pod to download the data

Create a pod that mounts the PVC and uses the Hugging Face Hub CLI to download the data:
```bash
kubectl apply -f pod.yaml
```
Connect to the pod:
```bash
kubectl exec -it hf-downloader -- /bin/bash
```
The PV corresponding to shared FS is mounted at `/persistent-storage`.

Let's download `meta-llama/Llama-3.1-8B` base model weights (Note that the model requires permission to download from Hugging Face):
```bash
# Llama model from Meta requires authentication to download
huggingface-cli login
mkdir -p /persistent-storage/models/meta-llama--Llama-3.1-8B
huggingface-cli download meta-llama/Llama-3.1-8B --local-dir /persistent-storage/models/meta-llama--Llama-3.1-8B --exclude="original/*"
```

### Clean up

To delete the pod, run the following command:
```bash
kubectl delete -f pod.yaml
```
