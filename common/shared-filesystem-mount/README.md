# Create PV and PVC for Shared Filesystem

Based on [csi-mounted-fs-path](https://github.com/nebius/nebius-solution-library/tree/main/modules/csi-mounted-fs-path) from Nebius Solution Library. Same limitations apply.

### To provision:

The following command will create a PersistentVolume (PV) and a PersistentVolumeClaim (PVC) in default namespace for a shared filesystem:

```bash
kubectl apply -f pv-pvc.yaml
```

### Cleanup:

To delete the PV and PVC, run:

```bash
kubectl delete -f pv-pvc.yaml
```