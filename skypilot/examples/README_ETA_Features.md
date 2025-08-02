# Enhanced S3 Migration with Advanced ETA Calculation

This repository contains enhanced versions of the S3 migration scripts with advanced ETA (Estimated Time of Arrival) calculation features for SkyPilot-based distributed data migration.

## Files Overview

### 1. `s3_to_s3_migration.yaml` (Current Enhanced Version)
- **Advanced ETA calculation** with per-node and cluster-wide progress tracking
- **Real-time speed monitoring** in MBps (megabytes per second)
- **Comprehensive debugging system** with DEBUG variable control
- **Resource monitoring** (RAM disk and disk space usage)
- **Automatic RAM disk cleanup** when space is low
- **Enhanced error diagnostics** with detailed failure information
- **Progress percentage** for both objects and data size
- **Completion time estimation** with human-readable format
- **Periodic progress updates** (15 seconds for node, 60 seconds for cluster)
- **Streaming transfers** to eliminate RAM disk size limitations
- **Transfer speed breakdown** per object (download/upload/average)
- **Advanced error handling** with retry mechanisms and detailed logging

## Key ETA Features

### 🕐 Real-Time ETA Calculation
- **Dynamic speed calculation** based on actual transfer rates
- **Adaptive ETA updates** as transfer speeds change
- **Speed metrics in MBps** (megabytes per second) for intuitive file transfer monitoring
- **Completion time prediction** with exact date/time
- **Cluster-wide ETA** accounting for parallel processing across nodes

### 📊 Progress Tracking
- **Object-based progress**: X/Y objects completed
- **Size-based progress**: X GB remaining of Y GB total
- **Percentage completion** for both metrics
- **Elapsed time tracking** with human-readable format
- **Cluster-wide progress aggregation** from all nodes

### 🌐 Cluster-Wide Monitoring
- **Distributed progress tracking** across all nodes
- **Cluster-level ETA** based on overall progress
- **Node synchronization** via S3-based coordination
- **Real-time cluster statistics** with completion and elapsed time
- **Parallel processing awareness** in ETA calculations

### ⚡ Speed Analysis
- **Download speed** per object in MB/s
- **Upload speed** per object in MB/s
- **Average transfer speed** calculation in MBps
- **Speed trend analysis** over time
- **Resource-aware speed monitoring**

### 🔧 Advanced Debugging & Monitoring
- **DEBUG variable control** for detailed output
- **Resource usage monitoring** (RAM disk and disk space)
- **Automatic cleanup** when RAM disk is 95%+ full
- **Emergency cleanup** when RAM disk is 98%+ full
- **Detailed error diagnostics** with exit codes and command output
- **File verification** with size checking
- **Network connectivity testing** before operations

## Usage Instructions

### Prerequisites
1. **AWS Profiles**: Configure both source and target AWS profiles
2. **Credentials**: Set up `~/.aws/config` and `~/.aws/credentials`
3. **Environment Variables**: Configure the required environment variables

### Required Environment Variables
```yaml
SOURCE_AWS_PROFILE: us-central1
SOURCE_ENDPOINT_URL: https://storage.us-central1.nebius.cloud:443
SOURCE_BUCKET: your-source-bucket
TARGET_AWS_PROFILE: eu-north1
TARGET_ENDPOINT_URL: https://storage.eu-north1.nebius.cloud:443
TARGET_BUCKET: your-target-bucket
NUM_CONCURRENT: 4
DEBUG: false  # Set to true for detailed debug output
```

## ETA Calculation Methodology

### 1. Speed Calculation
```
Average Speed (bytes/sec) = Total Bytes Transferred / Elapsed Time
Speed (MBps) = bytes/sec / (1024 * 1024)  # Megabytes per second
```

### 2. ETA Calculation
```
Remaining Bytes = Total Bytes - Transferred Bytes
ETA (seconds) = Remaining Bytes / Average Speed (bytes/sec)
```

### 3. Progress Percentage
```
Object Progress % = (Processed Objects / Total Objects) * 100
Size Progress % = (Transferred Bytes / Total Bytes) * 100
```

### 4. Cluster ETA (Parallel Processing)
```
Cluster Speed = Sum of all node speeds
Cluster ETA = Remaining Objects / (Cluster Speed * Number of Nodes)
```

## Sample Output

### Node Progress Report
```
📊 Node 1 Progress Report:
   Objects: 150/500 (30.0%)
   Size: 45.2 GB remaining (65.3% complete)
   Speed: 26.58 MBps
   ETA: 2h 15m 30s
   Completion: 2024-01-15 14:30:45
   Elapsed: 1h 30m 15s
---
```

### Cluster Progress Report
```
🌐 Cluster Progress: 1250/5000 objects
   Cluster ETA: 2h 15m
   Completion: 2024-01-15 16:45:30
   Elapsed: 1h 30m 15s
---
```

### Debug Output (when DEBUG=true)
```
Debug mode: true
🔍 Processing: part-000029.zip (661027678 bytes)
   Source: s3://source-bucket/part-000029.zip
   Target: s3://target-bucket/part-000029.zip
💾 Resource Usage:
   RAM disk: 45% used (8.2G available)
   Root disk: 12% used (15.8G available)
   Checking if object exists in source...
   ✅ Object exists in source
   Streaming transfer (download → upload)...
   Command: s5cmd --profile us-central1 --endpoint-url https://storage.us-central1.nebius.cloud:443 cp s3://source-bucket/part-000029.zip - | s5cmd --profile eu-north1 --endpoint-url https://storage.eu-north1.nebius.cloud:443 cp - s3://target-bucket/part-000029.zip
   ✅ Streaming transfer completed in 23s (35.61MB/s)
✅ part-000029.zip (661027678 bytes) - Streaming: 35.61MB/s
```

### Final Statistics
```
======================================================
📊 Final Migration Statistics:
======================================================
⏱️  Total migration time: 6h 45m 30s
📦 Source bucket: 5000 objects
📦 Target bucket: 5000 objects
💾 Total data transferred: 1.2 TB
🚀 Average speed: 180.5 GB/h
======================================================
```

## Performance Optimizations

### 1. RAM Disk Management
- **16GB RAM disk** for temporary storage (increased from 8GB)
- **Automatic cleanup** when space exceeds 95%
- **Emergency cleanup** when space exceeds 98%
- **Faster I/O** compared to local disk
- **Resource monitoring** with warnings at 90%+ usage

### 2. Parallel Processing
- **Configurable concurrency** (default: 4)
- **Sequential processing** for reliability
- **Load balancing** across multiple nodes
- **Cluster-wide coordination** via S3

### 3. Network Optimization
- **s5cmd** for high-performance S3 operations
- **Concurrent transfers** for maximum throughput
- **Retry mechanisms** for failed transfers
- **Detailed error diagnostics** with command output

## Monitoring and Debugging

### Progress Monitoring
- **Real-time updates** every 15 seconds (node) / 60 seconds (cluster)
- **Detailed speed metrics** per transfer in MBps
- **Error tracking** with retry logic and detailed diagnostics

### Debugging Features
- **DEBUG variable control** for detailed output
- **Resource usage monitoring** (RAM disk and disk space)
- **Transfer speed breakdown** per object
- **Error categorization** (download vs upload failures)
- **Command output capture** for debugging
- **File verification** with size checking
- **Network connectivity testing**

### Resource Monitoring
```
💾 Resource Usage:
   RAM disk: 45% used (8.2G available)
   Root disk: 12% used (15.8G available)
```

### Automatic Cleanup
```
🧹 Cleaning up RAM disk (96% full)...
   Removing old files:
     Removing: /mnt/ramdisk/s3migration/temp/part-000001.zip
     Removing: /mnt/ramdisk/s3migration/temp/part-000002.zip
✅ Cleanup complete. RAM disk now 45% full
```

## Error Handling

### Automatic Retries
- **Exponential backoff** for failed operations
- **Configurable retry limits** (5 attempts)
- **Graceful degradation** for persistent failures

### Error Reporting
- **Detailed error messages** with context
- **Failed object tracking** for manual review
- **Exit code reporting** for debugging
- **Command output capture** for analysis
- **Resource status** during failures

### Enhanced Error Diagnostics
```
❌ Download failed: part-000029.zip
   Debug: Check source bucket permissions, object existence, and connectivity
   Download exit code: 1
   Download output: ERROR "cp s3://bucket/file.zip": given object not found
   Local file path: /mnt/ramdisk/s3migration/temp/part-000029.zip
   File exists: No
   Available RAM disk space: 8.2G
   Available root disk space: 15.8G
```

## Best Practices

### 1. Resource Planning
- **Monitor memory usage** during migration
- **Adjust RAM disk size** based on available memory (16GB default)
- **Scale concurrency** based on network capacity
- **Enable DEBUG mode** for troubleshooting

### 2. Network Considerations
- **Monitor bandwidth usage** across nodes
- **Adjust update intervals** for large datasets
- **Consider timezone differences** for completion estimates
- **Test connectivity** before large migrations

### 3. Monitoring
- **Regular progress checks** via logs
- **Speed trend analysis** for optimization
- **Cluster health monitoring** during migration
- **Resource usage tracking** for optimization

## Troubleshooting

### Common Issues

#### 1. RAM Disk Space Issues
```
⚠️ WARNING: RAM disk is 96% full!
🧹 Cleaning up RAM disk (96% full)...
```
- **Solution**: Automatic cleanup is enabled, but you can increase RAM disk size
- **Prevention**: Monitor resource usage and adjust cleanup thresholds

#### 2. Slow Transfer Speeds
- **Check network bandwidth** between source and target
- **Adjust concurrency settings** based on available resources
- **Monitor system resources** (CPU, memory, disk I/O)
- **Enable DEBUG mode** to see detailed transfer information

#### 3. ETA Inaccuracy
- **Speed variations** due to network conditions
- **File size distribution** affecting average speeds
- **Cluster load balancing** impacting individual node speeds
- **Parallel processing** affecting cluster ETA calculations

#### 4. Debug Mode Usage
```yaml
# Enable detailed debugging
DEBUG: true

# Disable for production (cleaner logs)
DEBUG: false
```

**Debug Output Troubleshooting:**
- **Check YAML file**: Ensure `DEBUG: true` is set in the environment variables
- **Look for debug indicator**: Should see "Debug mode: true" at job start
- **Use sky logs**: Run `sky logs <cluster-name>` to see all output
- **Debug messages appear**: Between progress updates, not replacing them
- **Resource monitoring**: Debug shows RAM disk and disk usage every operation

### Debug Commands
```bash
# Check progress monitor process
ps aux | grep progress

# Monitor RAM disk usage
df -h /mnt/ramdisk

# Check network speed
iftop -i eth0

# Monitor system resources
htop

# Check AWS credentials
ls -la ~/.aws/
```

## Configuration Options

### Environment Variables
```yaml
# AWS Configuration
SOURCE_AWS_PROFILE: us-central1
SOURCE_ENDPOINT_URL: https://storage.us-central1.nebius.cloud:443
SOURCE_BUCKET: your-source-bucket
TARGET_AWS_PROFILE: eu-north1
TARGET_ENDPOINT_URL: https://storage.eu-north1.nebius.cloud:443
TARGET_BUCKET: your-target-bucket

# Performance Settings
NUM_CONCURRENT: 4  # Number of concurrent transfers
DEBUG: false        # Enable detailed debugging

# Resource Settings
# RAM disk size: 16GB (configured in setup)
# Progress update intervals: 15s (node), 60s (cluster)
```

### Resource Management
- **RAM Disk Size**: 16GB (configurable in setup section)
- **Cleanup Thresholds**: 95% (normal), 98% (emergency)
- **Progress Intervals**: 15s (node), 60s (cluster)
- **Retry Limits**: 5 attempts with exponential backoff

## Future Enhancements

### Planned Features
- **Web-based dashboard** for real-time monitoring
- **Email/Slack notifications** for completion
- **Resume capability** for interrupted migrations
- **Advanced analytics** for transfer optimization
- **Multi-cloud support** for additional providers
- **Enhanced streaming transfers** with compression and chunking

### Performance Improvements
- **Adaptive concurrency** based on network conditions
- **Intelligent chunking** for large files
- **Compression options** for bandwidth optimization
- **Parallel verification** of transferred objects
- **Dynamic resource allocation** based on workload

## Contributing

When contributing to the ETA calculation features:

1. **Test thoroughly** with various file sizes and network conditions
2. **Document changes** to ETA calculation logic
3. **Maintain backward compatibility** with existing configurations
4. **Add unit tests** for new ETA calculation functions
5. **Update documentation** for new features
6. **Test debugging features** with DEBUG mode
7. **Verify resource monitoring** under various conditions

## License

This project follows the same license as the original SkyPilot project. 