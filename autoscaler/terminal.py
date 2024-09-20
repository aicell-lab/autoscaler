import subprocess
import logging
from config import Config
from typing import List

logger = logging.getLogger(__name__)

def run_command(args: List[str]) -> str:
    try:
        result = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return result.stdout.decode().strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Process error: {e.stderr}, args: {args}")
        return None
    except FileNotFoundError:
        logger.error("The 'squeue' command was not found. Ensure SLURM is installed and accessible.")
        return None
    except Exception as e:
        logger.error(f"Unkown error: {e}")
        return None

def launch_worker_node() -> int:
    try:
        output = run_command(args=[Config.WORKER_LAUNCH_PATH])
        job_id = int(output.split()[-1])
        logger.info(f"Job submitted successfully with Job ID: {job_id}")
        return job_id
    except subprocess.CalledProcessError as e:
        logger.error(f"Error submitting SLURM job: {e.stderr}")
        return -1