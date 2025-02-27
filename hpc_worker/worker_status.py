import os
from datetime import datetime, timezone
from typing import Dict

import yaml
import numpy as np

from hpc_worker.ray_cluster import check_ray_cluster

def format_time(last_deployed_time_s: int, tz: timezone = timezone.utc) -> Dict:
    current_time = datetime.now(tz)
    last_deployed_time = datetime.fromtimestamp(last_deployed_time_s, tz)
    
    duration = current_time - last_deployed_time
    days = duration.days
    seconds = duration.seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60

    duration_parts = []
    if days > 0:
        duration_parts.append(f"{days}d")
    if hours > 0:
        duration_parts.append(f"{hours}h")
    if minutes > 0:
        duration_parts.append(f"{minutes}m")
    if remaining_seconds > 0:
        duration_parts.append(f"{remaining_seconds}s")

    return {
        "timestamp": last_deployed_time.strftime("%Y/%m/%d %H:%M:%S"),
        "timezone": str(tz),
        "duration_since": " ".join(duration_parts) if duration_parts else "0s"
    }

def load_dataset_info(dataset_path: str) -> Dict:
    """Load dataset information from info.npz file"""
    info_path = os.path.join(dataset_path, 'info.npz')
    if not os.path.exists(info_path):
        return None
    
    info = np.load(info_path)
    return {
        'name': os.path.basename(dataset_path),
        'samples': int(info['length'])
    }

def process_model_info(image: str) -> Dict:
    """Process docker image string into model info"""
    repo, version = image.split(':')
    name = repo.split('/')[-1]
    return {
        'name': name,
        'image': image,
        'version': version
    }

def worker_status(config_path: str, registered_at: int, context: dict) -> Dict:
    """Get complete worker status"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Process datasets
    datasets = []
    for path in config['dataset_paths']:
        if os.path.exists(path):
            info = load_dataset_info(path)
            if info:
                datasets.append(info)
    
    # Process models
    models = [process_model_info(image) for image in config['trusted_models']]
    
    # Get ray cluster status
    ray_status = check_ray_cluster()
    
    # Format time information
    time_info = format_time(registered_at)
    
    status = {
        'machine': {
            'name': config['machine_name'],
            'max_gpus': config['max_gpus']
        },
        'ray_cluster': {
            'head_node_running': ray_status['head_running'],
            'active_workers': ray_status['worker_count']
        },
        'registration': {
            'registered_at': time_info['timestamp'],
            'timezone': time_info['timezone'],
            'uptime': time_info['duration_since']
        },
        'datasets': datasets,
        'trusted_models': models
    }
    
    return status
