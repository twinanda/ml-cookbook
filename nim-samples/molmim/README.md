# NVIDIA MolMIM

This [example](MolMIMOracleControlledGeneration.ipynb) has been adapted from [NVIDIA digital-biology-examples](https://github.com/NVIDIA/digital-biology-examples/tree/main/examples/nims/molmim/MolMIMOracleControlledGeneration.ipynb) to fit Nebius deployment. 

## Setup

### Preparing the MolMIM NIM

Install your MolMIM NIM through Nebius Applications Catalog. You can verify that the NIM has started successfully by querying its status. 
After replacing the username, password, and URL in the following command to the exposed NIM endpoint, it will return `{"status":"ready"}` when the NIM is ready:

```bash
   curl -X 'GET' \
     'https://{username}:{password}@{url}/v1/health/ready' \
     -H 'accept: application/json'
```

### Jupyter Lab Environment

Prepare your JupyterLab environment locally, or use JupyterLab provided by Nebius Standalone Applications. 
Load the [notebook](MolMIMOracleControlledGeneration.ipynb) into the JupyterLab environment. 
