#!/bin/bash

PROJECT_NAME="demo"
GPU_PER_NODE=1
NUM_WORKERS=2

runai mpi submit demo-nccl-test-16-h200 -p $PROJECT_NAME \
    --image cr.eu-north1.nebius.cloud/nebius-benchmarks/nccl-tests:2.23.4-ubu22.04-cu12.4 \
    --environment "NCCL_DEBUG=version" \
    --environment "NCCL_DEBUG_SUBSYS=init,env" \
    --environment "NCCL_SOCKET_IFNAME=eth0" \
    --environment "UCX_NET_DEVICES=eth0" \
    --environment "NCCL_IB_HCA=mlx5" \
    --environment "NCCL_BUFFSIZE=8388608" \
    --environment "NCCL_IB_QPS_PER_CONNECTION=2" \
    --environment "OMPI_MCA_coll_hcoll_enable=0" \
    --environment "OMPI_MCA_coll_ucc_enable=0" \
    --large-shm \
    --capability "IPC_LOCK" \
    --workers $NUM_WORKERS \
    --gpu-devices-request $GPU_PER_NODE \
    --slots-per-worker $GPU_PER_NODE \
    --host-path "path=/mnt/data,mount=/sfs,mount-propagation=HostToContainer"\
    --master-no-pvcs \
    --master-command "mpirun" \
    --master-args  "-np $(($NUM_WORKERS*$GPU_PER_NODE)) --map-by ppr:$GPU_PER_NODE:node --allow-run-as-root bash -c '/opt/nccl_tests/build/all_reduce_perf -b 1024M -e 16G -f 2 -g 1'" \
    --command -- bash -c "/usr/sbin/sshd -De"

