"""
Simplified environment file that doesn't require AWS keys.
"""

import logging
import os
from pathlib import Path
import re
import typing as tp

import omegaconf

logger = logging.getLogger(__name__)


class AudioCraftEnvironment:
    """Environment configuration for VoiceCraft.
    
    This is a simplified version that doesn't require AWS keys.
    """
    _instance = None
    DEFAULT_TEAM = "default"

    def __init__(self) -> None:
        """Loads configuration."""
        self.team = "default"
        self.cluster = "local"
        
        # Create a minimal config that provides the necessary values
        self.config = omegaconf.OmegaConf.create({
            "local": {
                "dora_dir": "/tmp/dora",
                "reference_dir": "/tmp/reference",
                "dataset_mappers": {},
            }
        })
        
        self._dataset_mappers = []

    def _get_cluster_config(self) -> omegaconf.DictConfig:
        return self.config["local"]

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Clears the environment and forces a reload on next invocation."""
        cls._instance = None

    @classmethod
    def get_team(cls) -> str:
        """Gets the selected team."""
        return cls.instance().team

    @classmethod
    def get_cluster(cls) -> str:
        """Gets the detected cluster."""
        return cls.instance().cluster

    @classmethod
    def get_dora_dir(cls) -> Path:
        """Gets the path to the dora directory."""
        cluster_config = cls.instance()._get_cluster_config()
        dora_dir = os.getenv("AUDIOCRAFT_DORA_DIR", cluster_config["dora_dir"])
        
        # Create the directory if it doesn't exist
        os.makedirs(dora_dir, exist_ok=True)
        
        logger.warning(f"Dora directory: {dora_dir}")
        return Path(dora_dir)

    @classmethod
    def get_reference_dir(cls) -> Path:
        """Gets the path to the reference directory."""
        cluster_config = cls.instance()._get_cluster_config()
        ref_dir = os.getenv("AUDIOCRAFT_REFERENCE_DIR", cluster_config["reference_dir"])
        
        # Create the directory if it doesn't exist
        os.makedirs(ref_dir, exist_ok=True)
        
        return Path(ref_dir)

    @classmethod
    def get_slurm_exclude(cls) -> tp.Optional[str]:
        """Get the list of nodes to exclude for that cluster."""
        return None

    @classmethod
    def get_slurm_partitions(cls, partition_types: tp.Optional[tp.List[str]] = None) -> str:
        """Gets the requested partitions for the current team and cluster as a comma-separated string."""
        return ""

    @classmethod
    def resolve_reference_path(cls, path: tp.Union[str, Path]) -> Path:
        """Converts reference placeholder in path with configured reference dir to resolve paths."""
        path = str(path)

        if path.startswith("//reference"):
            reference_dir = cls.get_reference_dir()
            logger.warning(f"Reference directory: {reference_dir}")
            path = re.sub("^//reference", str(reference_dir), path)

        return Path(path)

    @classmethod
    def apply_dataset_mappers(cls, path: str) -> str:
        """Applies dataset mapping regex rules as defined in the configuration."""
        return path 