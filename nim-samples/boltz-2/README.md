# Boltz-2 Python Client Examples

**The following code examples have been modified (on 16 September 2025) from the [NVIDIA digital biology examples](https://github.com/nvidia/digital-biology-examples/tree/main/examples/nims/boltz-2/examples) to align with the Nebius deployment framework and facilitate customer integration. For the unmodified and complete examples, please refer to the original repository.**

## 🚀 Quick Start

### Pre-requisites
Install `boltz2_client` through PyPI. 

```bash
pip install boltz2-python-client
```

### Boltz2 NIM URL


#### Python API

The `boltz2_client` Python package assumes that your NIM instance is deployed either on localhost or on NVIDIA-hosted infrastructure. We have adjusted all the python samples here to work with the deployed Boltz2 NIM on Nebius. Go to [`constants.py`](constants.py) and replace the username, password, and the endpoint accordingly. 

```python
username = '<username>'
password = '<password>'
endpoint = '<endpoint>'
base_url = f'https://{username}:{password}@{endpoint}'
```

#### CLI with custom URL

Similarly, the `boltz2` CLI command also has the same assumption. To mitigate the issue, specify the URL to Boltz2 NIM deployed on Nebius as `--base-url` in the command line. For example, use the following command to check the health of the NIM deployment.

```bash
boltz2 --base-url <base_url> health 
```

### Run All Examples

```bash
# Run individual examples
python 01_basic_protein_folding.py
python 02_protein_structure_prediction_with_msa.py
# ... etc

# Or use the CLI examples
boltz2 examples
```

## 📁 Example Files Overview

In some examples, some additional changes are made to ensure the scripts run without other issues. For examples with additional changes, they are detailed in the description below. 

### 1. **Basic Protein Folding** (`01_basic_protein_folding.py`)
- Simple protein structure prediction from sequence
- Basic parameter usage
- Structure output handling
- Confidence score interpretation

**Run:** `python 01_basic_protein_folding.py`

### 2. **Protein Structure Prediction with MSA** (`02_protein_structure_prediction_with_msa.py`)
- Multiple Sequence Alignment (MSA) integration
- Direct comparison between basic and MSA-guided predictions
- MSA file handling and validation
- Confidence score analysis and interpretation
- Educational approach to understanding MSA benefits
Nebius changes:
- The structure of `msa-kras-g12x_combined.a3m` has been updated to address `422 Error` from the server. 

**Run:** `python 02_protein_structure_prediction_with_msa.py`

### 3. **Protein-Ligand Complex** (`03_protein_ligand_complex.py`)
- SMILES-based ligand binding
- CCD (Chemical Component Dictionary) ligands
- Pocket-constrained binding
- Multiple ligand examples (aspirin, acetate, ATP)

**Run:** `python 03_protein_ligand_complex.py`

### 4. **Covalent Bonding** (`04_covalent_bonding.py`)
- Protein-ligand covalent bonds
- Disulfide bond formation (intra-protein)
- Multiple simultaneous bonds
- Flexible atom-to-atom bonding

**Run:** `python 04_covalent_bonding.py`

### 5. **DNA-Protein Complex** (`05_dna_protein_complex.py`)
- DNA-protein interactions
- RNA-protein complexes
- Multi-protein DNA complexes
- Nuclease-DNA systems with recognition sequences

**Run:** `python 05_dna_protein_complex.py`

### 6. **YAML Configurations** (`06_yaml_configurations.py`)
- Official Boltz YAML format compatibility
- Programmatic YAML config creation
- MSA file integration
- Parameter override examples
- Affinity prediction configurations

**Run:** `python 06_yaml_configurations.py`

### 7. **Advanced Parameters** (`07_advanced_parameters.py`)
- Diffusion parameter exploration
- Quality vs. speed trade-offs
- Complex molecular system configurations
- JSON configuration files
- Specialized prediction options
Nebius changes:
- Commented out `without_potentials` parameter from `predict()` function since the client API does not seem to support it. 

**Run:** `python 07_advanced_parameters.py`

### 8. **Affinity Prediction** (`08_affinity_prediction.py`)
- Predict binding affinity (IC50/pIC50) for protein-ligand complexes
- Binary binding probability estimation
- Model-specific predictions from ensemble
- Molecular weight correction options
- Complete kinase-inhibitor example

**Run:** `python 08_affinity_prediction.py`

**Quick Test:** `python 08_affinity_prediction_simple.py` (simplified version)

### 9. **Virtual Screening** (`09_virtual_screening.py`)
- High-level API for drug discovery campaigns
- Parallel compound screening with progress tracking
- Automatic result analysis and ranking
- Support for CSV/JSON compound libraries
- Pocket constraint specification
- Batch processing for large libraries
Nebius changes:
- Overrid the `quick_screen()` function to allow instantiation of `Boltz2Client` with custom URL.
- Commented out `pocket_residues` from the `screener.screen()` function because it is causing some pydantic issues.
- Removed asynchronous example due to `await` error.

**Run:** `python 09_virtual_screening.py`


## 📊 Example Categories by Use Case

### **Research & Development**
- `01_basic_protein_folding.py` - Quick structure predictions
- `02_protein_structure_prediction_with_msa.py` - High-accuracy predictions
- `07_advanced_parameters.py` - Parameter optimization

### **Drug Discovery**
- `03_protein_ligand_complex.py` - Drug-target interactions
- `04_covalent_bonding.py` - Covalent drug design
- `06_yaml_configurations.py` - Affinity predictions

### **Structural Biology**
- `05_dna_protein_complex.py` - Nucleic acid interactions
- `04_covalent_bonding.py` - Disulfide bond analysis
- `07_advanced_parameters.py` - Complex molecular systems

### **Production Deployment**
- `06_yaml_configurations.py` - Batch processing
- `07_advanced_parameters.py` - Performance tuning

## 🔧 Configuration Examples

### **YAML Configuration Files**
```yaml
# protein_ligand.yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: "MKTVRQERLK..."
  - ligand:
      id: B
      smiles: "CC(=O)O"
```

### **JSON Configuration Files**
```json
{
  "polymers": [
    {
      "id": "A",
      "molecule_type": "protein",
      "sequence": "MKTVRQERLK..."
    }
  ],
  "recycling_steps": 5,
  "sampling_steps": 100
}
```

## 📈 Parameter Guidelines

### **Quality Levels**
- **Fast (Low Quality)**: `recycling_steps=1, sampling_steps=10`
- **Standard**: `recycling_steps=3, sampling_steps=50`
- **High Quality**: `recycling_steps=5, sampling_steps=100`
- **Maximum Quality**: `recycling_steps=6, sampling_steps=200`

### **Diversity Control**
- **High Diversity**: `step_scale=0.5-0.8`
- **Standard**: `step_scale=1.638`
- **Focused**: `step_scale=2.0-3.0`

## 🔗 API Request Variations Covered

### **Molecular Types**
- ✅ Proteins (with/without MSA)
- ✅ DNA sequences
- ✅ RNA sequences
- ✅ Small molecule ligands (SMILES/CCD)

### **Complex Types**
- ✅ Single proteins
- ✅ Protein-ligand complexes
- ✅ Covalent complexes
- ✅ DNA-protein complexes
- ✅ RNA-protein complexes
- ✅ Multi-polymer systems

### **Constraint Types**
- ✅ Pocket constraints
- ✅ Covalent bonds
- ✅ Disulfide bonds
- ✅ Multiple simultaneous constraints

### **Configuration Methods**
- ✅ Programmatic (Python objects)
- ✅ YAML files (official Boltz format)
- ✅ JSON files (advanced parameters)
- ✅ CLI commands

### **Endpoint Types**
- ✅ Local deployments
- ✅ NVIDIA hosted endpoints
- ✅ Custom endpoints
- ✅ Endpoint failover

## 🎯 Example Selection Guide

| **Goal** | **Recommended Examples** |
|----------|-------------------------|
| Learn basics | `01`, `03`, `06` |
| Improve accuracy | `02`, `07` |
| Drug discovery | `03`, `04`, `08`, `09` |
| Affinity prediction | `08` |
| Virtual screening | `09` |
| Complex systems | `05`, `07` |
| Production setup | `06`, `07` |
| Parameter tuning | `07` |
| Batch processing | `06`, `07`, `09` |

## 📚 Additional Resources

- [**Original Boltz2 code samples module**](https://github.com/nvidia/digital-biology-examples/tree/main/examples/nims/boltz-2)
- **CLI Help**: `boltz2 --help`

## 🐛 Troubleshooting

1. **Service not running**: Ensure you point to the right URL of the deployed Boltz-2 NIM. Refer to [this section](#boltz2-nim-url).
2. **Timeout errors**: Increase `timeout` parameter for complex predictions
3. **Memory issues**: Reduce `diffusion_samples` or `sampling_steps`
