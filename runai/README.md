#Run:ai examples

##Run:ai CLI setup
To configure the CLI please follow the procedure 
https://run-ai-docs.nvidia.com/self-hosted/2.21/reference/cli/install-cli

When the CLI is installed you need to authenticate in the cluster with CLI, running this command
```
runai login
```

##Submit the job

To submit the MPI job, naviage to the `examples/wrokloads` folder and to run `nccl-test` run the one of the scripts in `mpi` folder for desired GPU generation `H100/H200` or `B200` 