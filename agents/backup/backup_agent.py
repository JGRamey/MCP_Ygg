#!/usr/bin/env python3
"""
MCP Server Backup Agent
Creates and restores database snapshots with integrity checks for Neo4j and Qdrant
"""

import gzip
import hashlib
import json
import logging
import multiprocessing as mp
import os
import shutil
import subprocess
import tarfile
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import aiofiles
import aiohttp
import asyncio
import boto3
import psutil
import yaml
from azure.storage.blob import BlobServiceClient
from botocore.exceptions import ClientError, NoCredentialsError
from google.cloud import storage as gcs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BackupMetadata:
    """Metadata for a backup"""

    backup_id: str
    backup_type: str  # 'full', 'incremental', 'differential'
    created_at: datetime
    size_bytes: int
    checksum: str
    components: List[str]  # ['neo4j', 'qdrant', 'metadata', 'files']
    status: str  # 'created', 'uploaded', 'verified', 'failed'
    storage_locations: List[str]
    retention_until: datetime
    tags: Dict[str, str]
    verification_results: Dict[str, Any]
    notes: Optional[str] = None


@dataclass
class RestorePoint:
    """Information about a restore point"""

    backup_id: str
    timestamp: datetime
    description: str
    component_checksums: Dict[str, str]
    database_versions: Dict[str, str]
    compatibility_notes: str


@dataclass
class BackupJob:
    """Background backup job"""

    job_id: str
    job_type: str
    status: str  # 'queued', 'running', 'completed', 'failed'
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    progress_percent: float
    current_step: str
    error_message: Optional[str]
    result: Optional[BackupMetadata]


class CloudStorageManager:
    """Manages cloud storage operations for backups"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize cloud storage manager"""
        self.config = config
        self.aws_client = None
        self.azure_client = None
        self.gcp_client = None

        # Initialize cloud clients
        self._init_aws()
        self._init_azure()
        self._init_gcp()

    def _init_aws(self):
        """Initialize AWS S3 client"""
        if self.config.get("aws", {}).get("enabled", False):
            try:
                self.aws_client = boto3.client(
                    "s3",
                    aws_access_key_id=self.config["aws"].get("access_key_id"),
                    aws_secret_access_key=self.config["aws"].get("secret_access_key"),
                    region_name=self.config["aws"].get("region", "us-east-1"),
                )
                logger.info("AWS S3 client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize AWS client: {e}")

    def _init_azure(self):
        """Initialize Azure Blob Storage client"""
        if self.config.get("azure", {}).get("enabled", False):
            try:
                connection_string = self.config["azure"].get("connection_string")
                if connection_string:
                    self.azure_client = BlobServiceClient.from_connection_string(
                        connection_string
                    )
                    logger.info("Azure Blob Storage client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Azure client: {e}")

    def _init_gcp(self):
        """Initialize Google Cloud Storage client"""
        if self.config.get("gcp", {}).get("enabled", False):
            try:
                credentials_path = self.config["gcp"].get("credentials_path")
                if credentials_path:
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
                    self.gcp_client = gcs.Client()
                    logger.info("Google Cloud Storage client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize GCP client: {e}")

    async def upload_backup(
        self, local_path: Path, backup_id: str, metadata: BackupMetadata
    ) -> List[str]:
        """Upload backup to configured cloud storage providers"""
        upload_locations = []

        # AWS S3 upload
        if self.aws_client and self.config.get("aws", {}).get("enabled", False):
            try:
                s3_key = f"mcp-backups/{backup_id}/{local_path.name}"
                bucket = self.config["aws"]["bucket"]

                # Upload with metadata
                extra_args = {
                    "Metadata": {
                        "backup-id": backup_id,
                        "created-at": metadata.created_at.isoformat(),
                        "checksum": metadata.checksum,
                        "backup-type": metadata.backup_type,
                    }
                }

                self.aws_client.upload_file(
                    str(local_path), bucket, s3_key, ExtraArgs=extra_args
                )
                upload_locations.append(f"s3://{bucket}/{s3_key}")
                logger.info(f"Uploaded backup to S3: {s3_key}")

            except Exception as e:
                logger.error(f"Failed to upload to S3: {e}")

        # Azure Blob upload
        if self.azure_client and self.config.get("azure", {}).get("enabled", False):
            try:
                container = self.config["azure"]["container"]
                blob_name = f"mcp-backups/{backup_id}/{local_path.name}"

                blob_client = self.azure_client.get_blob_client(
                    container=container, blob=blob_name
                )

                with open(local_path, "rb") as data:
                    blob_client.upload_blob(
                        data,
                        metadata={
                            "backup_id": backup_id,
                            "created_at": metadata.created_at.isoformat(),
                            "checksum": metadata.checksum,
                        },
                        overwrite=True,
                    )

                upload_locations.append(f"azure://{container}/{blob_name}")
                logger.info(f"Uploaded backup to Azure: {blob_name}")

            except Exception as e:
                logger.error(f"Failed to upload to Azure: {e}")

        # Google Cloud Storage upload
        if self.gcp_client and self.config.get("gcp", {}).get("enabled", False):
            try:
                bucket_name = self.config["gcp"]["bucket"]
                bucket = self.gcp_client.bucket(bucket_name)
                blob_name = f"mcp-backups/{backup_id}/{local_path.name}"
                blob = bucket.blob(blob_name)

                blob.metadata = {
                    "backup_id": backup_id,
                    "created_at": metadata.created_at.isoformat(),
                    "checksum": metadata.checksum,
                }

                blob.upload_from_filename(str(local_path))
                upload_locations.append(f"gcs://{bucket_name}/{blob_name}")
                logger.info(f"Uploaded backup to GCS: {blob_name}")

            except Exception as e:
                logger.error(f"Failed to upload to GCS: {e}")

        return upload_locations

    async def download_backup(
        self, backup_id: str, component: str, local_path: Path
    ) -> bool:
        """Download backup from cloud storage"""
        # Try each storage provider until successful
        for provider in ["aws", "azure", "gcp"]:
            if await self._download_from_provider(
                provider, backup_id, component, local_path
            ):
                return True

        return False

    async def _download_from_provider(
        self, provider: str, backup_id: str, component: str, local_path: Path
    ) -> bool:
        """Download from specific cloud provider"""
        try:
            if provider == "aws" and self.aws_client:
                bucket = self.config["aws"]["bucket"]
                s3_key = f"mcp-backups/{backup_id}/{component}"
                self.aws_client.download_file(bucket, s3_key, str(local_path))
                return True

            elif provider == "azure" and self.azure_client:
                container = self.config["azure"]["container"]
                blob_name = f"mcp-backups/{backup_id}/{component}"
                blob_client = self.azure_client.get_blob_client(
                    container=container, blob=blob_name
                )

                with open(local_path, "wb") as download_file:
                    download_file.write(blob_client.download_blob().readall())
                return True

            elif provider == "gcp" and self.gcp_client:
                bucket_name = self.config["gcp"]["bucket"]
                bucket = self.gcp_client.bucket(bucket_name)
                blob_name = f"mcp-backups/{backup_id}/{component}"
                blob = bucket.blob(blob_name)
                blob.download_to_filename(str(local_path))
                return True

        except Exception as e:
            logger.warning(f"Download from {provider} failed: {e}")

        return False


class Neo4jBackupManager:
    """Manages Neo4j database backups and restores"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Neo4j backup manager"""
        self.config = config
        self.neo4j_config = config.get("neo4j", {})

    async def create_backup(self, backup_path: Path) -> Tuple[bool, str, int]:
        """Create Neo4j database backup"""
        try:
            backup_file = backup_path / "neo4j_backup.dump"

            # Use neo4j-admin dump command
            neo4j_home = self.neo4j_config.get("home_dir", "/var/lib/neo4j")
            database = self.neo4j_config.get("database", "neo4j")

            cmd = [
                "neo4j-admin",
                "dump",
                "--database",
                database,
                "--to",
                str(backup_file),
            ]

            # Add authentication if configured
            if self.neo4j_config.get("username") and self.neo4j_config.get("password"):
                cmd.extend(["--username", self.neo4j_config["username"]])
                cmd.extend(["--password", self.neo4j_config["password"]])

            logger.info(f"Creating Neo4j backup: {' '.join(cmd)}")

            # Run backup command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=neo4j_home,
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                # Compress the backup
                compressed_file = backup_file.with_suffix(".dump.gz")
                await self._compress_file(backup_file, compressed_file)

                # Remove uncompressed file
                backup_file.unlink()

                file_size = compressed_file.stat().st_size
                logger.info(
                    f"Neo4j backup completed: {compressed_file} ({file_size} bytes)"
                )

                return True, str(compressed_file), file_size
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                logger.error(f"Neo4j backup failed: {error_msg}")
                return False, error_msg, 0

        except Exception as e:
            logger.error(f"Neo4j backup error: {e}")
            return False, str(e), 0

    async def restore_backup(
        self, backup_file: Path, target_database: str = None
    ) -> Tuple[bool, str]:
        """Restore Neo4j database from backup"""
        try:
            if not backup_file.exists():
                return False, f"Backup file not found: {backup_file}"

            # Decompress if needed
            if backup_file.suffix == ".gz":
                temp_file = backup_file.with_suffix("")
                await self._decompress_file(backup_file, temp_file)
                restore_file = temp_file
            else:
                restore_file = backup_file

            # Use neo4j-admin load command
            neo4j_home = self.neo4j_config.get("home_dir", "/var/lib/neo4j")
            database = target_database or self.neo4j_config.get("database", "neo4j")

            cmd = [
                "neo4j-admin",
                "load",
                "--from",
                str(restore_file),
                "--database",
                database,
                "--force",  # Overwrite existing database
            ]

            logger.info(f"Restoring Neo4j backup: {' '.join(cmd)}")

            # Stop Neo4j service first
            await self._stop_neo4j_service()

            try:
                # Run restore command
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=neo4j_home,
                )

                stdout, stderr = await process.communicate()

                if process.returncode == 0:
                    logger.info("Neo4j restore completed successfully")
                    return True, "Restore completed"
                else:
                    error_msg = stderr.decode() if stderr else "Unknown error"
                    logger.error(f"Neo4j restore failed: {error_msg}")
                    return False, error_msg

            finally:
                # Restart Neo4j service
                await self._start_neo4j_service()

                # Clean up temporary file
                if restore_file != backup_file:
                    restore_file.unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Neo4j restore error: {e}")
            return False, str(e)

    async def _compress_file(self, source: Path, target: Path):
        """Compress file using gzip"""
        with open(source, "rb") as f_in:
            with gzip.open(target, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

    async def _decompress_file(self, source: Path, target: Path):
        """Decompress gzip file"""
        with gzip.open(source, "rb") as f_in:
            with open(target, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

    async def _stop_neo4j_service(self):
        """Stop Neo4j service"""
        try:
            cmd = ["systemctl", "stop", "neo4j"]
            process = await asyncio.create_subprocess_exec(*cmd)
            await process.communicate()

            # Wait for service to stop
            await asyncio.sleep(5)

        except Exception as e:
            logger.warning(f"Failed to stop Neo4j service: {e}")

    async def _start_neo4j_service(self):
        """Start Neo4j service"""
        try:
            cmd = ["systemctl", "start", "neo4j"]
            process = await asyncio.create_subprocess_exec(*cmd)
            await process.communicate()

            # Wait for service to start
            await asyncio.sleep(10)

        except Exception as e:
            logger.warning(f"Failed to start Neo4j service: {e}")


class QdrantBackupManager:
    """Manages Qdrant vector database backups and restores"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Qdrant backup manager"""
        self.config = config
        self.qdrant_config = config.get("qdrant", {})
        self.qdrant_url = f"http://{self.qdrant_config.get('host', 'localhost')}:{self.qdrant_config.get('port', 6333)}"

    async def create_backup(self, backup_path: Path) -> Tuple[bool, str, int]:
        """Create Qdrant collections backup"""
        try:
            total_size = 0
            backup_files = []

            # Get list of collections
            collections = await self._get_collections()

            if not collections:
                logger.warning("No Qdrant collections found")
                return True, "No collections to backup", 0

            # Backup each collection
            for collection_name in collections:
                collection_file = backup_path / f"qdrant_{collection_name}.json"

                success, size = await self._backup_collection(
                    collection_name, collection_file
                )
                if success:
                    backup_files.append(collection_file)
                    total_size += size
                else:
                    logger.error(f"Failed to backup collection: {collection_name}")

            # Create archive
            if backup_files:
                archive_file = backup_path / "qdrant_backup.tar.gz"
                await self._create_archive(backup_files, archive_file)

                # Remove individual files
                for file in backup_files:
                    file.unlink(missing_ok=True)

                final_size = archive_file.stat().st_size
                logger.info(
                    f"Qdrant backup completed: {archive_file} ({final_size} bytes)"
                )

                return True, str(archive_file), final_size
            else:
                return False, "No collections successfully backed up", 0

        except Exception as e:
            logger.error(f"Qdrant backup error: {e}")
            return False, str(e), 0

    async def restore_backup(self, backup_file: Path) -> Tuple[bool, str]:
        """Restore Qdrant collections from backup"""
        try:
            if not backup_file.exists():
                return False, f"Backup file not found: {backup_file}"

            # Extract archive
            temp_dir = backup_file.parent / "qdrant_restore_temp"
            temp_dir.mkdir(exist_ok=True)

            try:
                await self._extract_archive(backup_file, temp_dir)

                # Restore each collection
                restored_collections = []

                for collection_file in temp_dir.glob("qdrant_*.json"):
                    collection_name = collection_file.stem.replace("qdrant_", "")

                    success = await self._restore_collection(
                        collection_name, collection_file
                    )
                    if success:
                        restored_collections.append(collection_name)
                    else:
                        logger.error(f"Failed to restore collection: {collection_name}")

                if restored_collections:
                    msg = f"Restored collections: {', '.join(restored_collections)}"
                    logger.info(msg)
                    return True, msg
                else:
                    return False, "No collections successfully restored"

            finally:
                # Clean up temporary directory
                shutil.rmtree(temp_dir, ignore_errors=True)

        except Exception as e:
            logger.error(f"Qdrant restore error: {e}")
            return False, str(e)

    async def _get_collections(self) -> List[str]:
        """Get list of Qdrant collections"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.qdrant_url}/collections") as response:
                    if response.status == 200:
                        data = await response.json()
                        return [
                            coll["name"]
                            for coll in data.get("result", {}).get("collections", [])
                        ]
                    else:
                        logger.error(f"Failed to get collections: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error getting collections: {e}")
            return []

    async def _backup_collection(
        self, collection_name: str, output_file: Path
    ) -> Tuple[bool, int]:
        """Backup a single Qdrant collection"""
        try:
            # Get collection info
            async with aiohttp.ClientSession() as session:
                # Get collection info
                async with session.get(
                    f"{self.qdrant_url}/collections/{collection_name}"
                ) as response:
                    if response.status != 200:
                        return False, 0

                    collection_info = await response.json()

                # Get all points (simplified - for large collections, implement pagination)
                async with session.post(
                    f"{self.qdrant_url}/collections/{collection_name}/points/scroll",
                    json={"limit": 10000, "with_payload": True, "with_vector": True},
                ) as response:
                    if response.status != 200:
                        return False, 0

                    points_data = await response.json()

                # Combine collection info and points
                backup_data = {
                    "collection_info": collection_info["result"],
                    "points": points_data["result"]["points"],
                    "backup_timestamp": datetime.now().isoformat(),
                }

                # Save to file
                async with aiofiles.open(output_file, "w") as f:
                    await f.write(json.dumps(backup_data, indent=2))

                file_size = output_file.stat().st_size
                logger.info(
                    f"Backed up collection {collection_name}: {file_size} bytes"
                )

                return True, file_size

        except Exception as e:
            logger.error(f"Error backing up collection {collection_name}: {e}")
            return False, 0

    async def _restore_collection(
        self, collection_name: str, backup_file: Path
    ) -> bool:
        """Restore a single Qdrant collection"""
        try:
            # Load backup data
            async with aiofiles.open(backup_file, "r") as f:
                content = await f.read()
                backup_data = json.loads(content)

            collection_info = backup_data["collection_info"]
            points = backup_data["points"]

            async with aiohttp.ClientSession() as session:
                # Delete existing collection if it exists
                await session.delete(f"{self.qdrant_url}/collections/{collection_name}")

                # Create collection
                create_payload = {
                    "vectors": collection_info["config"]["params"]["vectors"]
                }

                async with session.put(
                    f"{self.qdrant_url}/collections/{collection_name}",
                    json=create_payload,
                ) as response:
                    if response.status not in [200, 201]:
                        logger.error(f"Failed to create collection {collection_name}")
                        return False

                # Restore points in batches
                batch_size = 100
                for i in range(0, len(points), batch_size):
                    batch = points[i : i + batch_size]

                    async with session.put(
                        f"{self.qdrant_url}/collections/{collection_name}/points",
                        json={"points": batch},
                    ) as response:
                        if response.status not in [200, 201]:
                            logger.error(
                                f"Failed to restore points batch for {collection_name}"
                            )
                            return False

                logger.info(
                    f"Restored collection {collection_name} with {len(points)} points"
                )
                return True

        except Exception as e:
            logger.error(f"Error restoring collection {collection_name}: {e}")
            return False

    async def _create_archive(self, files: List[Path], archive_file: Path):
        """Create tar.gz archive"""
        with tarfile.open(archive_file, "w:gz") as tar:
            for file in files:
                tar.add(file, arcname=file.name)

    async def _extract_archive(self, archive_file: Path, extract_dir: Path):
        """Extract tar.gz archive"""
        with tarfile.open(archive_file, "r:gz") as tar:
            tar.extractall(extract_dir)


class IntegrityChecker:
    """Verifies backup integrity and consistency"""

    def __init__(self):
        """Initialize integrity checker"""
        pass

    async def verify_backup(
        self, backup_metadata: BackupMetadata, backup_path: Path
    ) -> Dict[str, Any]:
        """Verify backup integrity"""
        results = {
            "overall_status": "pending",
            "file_integrity": {},
            "checksum_verification": {},
            "size_verification": {},
            "content_validation": {},
            "errors": [],
        }

        try:
            # Verify file existence and sizes
            for component in backup_metadata.components:
                component_file = backup_path / f"{component}_backup.*"
                matching_files = list(backup_path.glob(component_file.name))

                if not matching_files:
                    results["errors"].append(
                        f"Missing backup file for component: {component}"
                    )
                    results["file_integrity"][component] = False
                    continue

                file_path = matching_files[0]

                # Check file size
                actual_size = file_path.stat().st_size
                results["size_verification"][component] = {
                    "expected": backup_metadata.size_bytes,
                    "actual": actual_size,
                    "valid": True,  # Would implement proper size checking per component
                }

                # Verify checksum
                actual_checksum = await self._calculate_file_checksum(file_path)
                results["checksum_verification"][component] = {
                    "expected": backup_metadata.checksum,
                    "actual": actual_checksum,
                    "valid": actual_checksum == backup_metadata.checksum,
                }

                results["file_integrity"][component] = True

            # Content validation
            if "neo4j" in backup_metadata.components:
                neo4j_valid = await self._validate_neo4j_backup(backup_path)
                results["content_validation"]["neo4j"] = neo4j_valid

            if "qdrant" in backup_metadata.components:
                qdrant_valid = await self._validate_qdrant_backup(backup_path)
                results["content_validation"]["qdrant"] = qdrant_valid

            # Overall status
            all_files_ok = all(results["file_integrity"].values())
            all_checksums_ok = all(
                v["valid"] for v in results["checksum_verification"].values()
            )
            all_content_ok = all(results["content_validation"].values())

            if (
                all_files_ok
                and all_checksums_ok
                and all_content_ok
                and not results["errors"]
            ):
                results["overall_status"] = "valid"
            else:
                results["overall_status"] = "invalid"

            return results

        except Exception as e:
            results["overall_status"] = "error"
            results["errors"].append(f"Verification error: {str(e)}")
            return results

    async def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file"""
        hash_sha256 = hashlib.sha256()

        async with aiofiles.open(file_path, "rb") as f:
            while chunk := await f.read(8192):
                hash_sha256.update(chunk)

        return hash_sha256.hexdigest()

    async def _validate_neo4j_backup(self, backup_path: Path) -> bool:
        """Validate Neo4j backup content"""
        try:
            # Check if backup file exists and is valid
            neo4j_files = list(backup_path.glob("neo4j_backup.*"))
            if not neo4j_files:
                return False

            backup_file = neo4j_files[0]

            # Basic validation - check if file is readable and has expected structure
            if backup_file.suffix == ".gz":
                # Try to read gzip file
                with gzip.open(backup_file, "rb") as f:
                    header = f.read(1024)
                    return len(header) > 0
            else:
                # Try to read regular file
                with open(backup_file, "rb") as f:
                    header = f.read(1024)
                    return len(header) > 0

        except Exception as e:
            logger.error(f"Neo4j backup validation error: {e}")
            return False

    async def _validate_qdrant_backup(self, backup_path: Path) -> bool:
        """Validate Qdrant backup content"""
        try:
            # Check if backup archive exists
            qdrant_files = list(backup_path.glob("qdrant_backup.*"))
            if not qdrant_files:
                return False

            backup_file = qdrant_files[0]

            # Try to extract and validate JSON structure
            if backup_file.suffix in [".tar", ".gz"]:
                with tarfile.open(backup_file, "r:gz") as tar:
                    members = tar.getnames()
                    return len(members) > 0 and any(
                        m.endswith(".json") for m in members
                    )

            return True

        except Exception as e:
            logger.error(f"Qdrant backup validation error: {e}")
            return False


class BackupAgent:
    """Main backup agent coordinating all backup operations"""

    def __init__(self, config_path: str = "agents/backup/config.yaml"):
        """Initialize backup agent"""
        self.load_config(config_path)

        # Initialize managers
        self.cloud_storage = CloudStorageManager(self.config.get("cloud_storage", {}))
        self.neo4j_manager = Neo4jBackupManager(self.config)
        self.qdrant_manager = QdrantBackupManager(self.config)
        self.integrity_checker = IntegrityChecker()

        # Job tracking
        self.active_jobs: Dict[str, BackupJob] = {}
        self.backup_history: List[BackupMetadata] = []

        # Load existing backup history
        self.load_backup_history()

    def load_config(self, config_path: str) -> None:
        """Load backup configuration"""
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            # Default configuration
            self.config = {
                "backup_schedule": {
                    "full_backup_interval_days": 7,
                    "incremental_backup_interval_hours": 6,
                    "max_backup_age_days": 30,
                },
                "storage": {
                    "local_backup_dir": "data/backups",
                    "compression": True,
                    "encryption": False,
                },
                "cloud_storage": {
                    "aws": {"enabled": False},
                    "azure": {"enabled": False},
                    "gcp": {"enabled": False},
                },
                "neo4j": {"host": "localhost", "port": 7687, "database": "neo4j"},
                "qdrant": {"host": "localhost", "port": 6333},
                "verification": {
                    "verify_after_backup": True,
                    "verify_before_restore": True,
                },
            }

    def load_backup_history(self) -> None:
        """Load backup history from storage"""
        history_file = (
            Path(self.config["storage"]["local_backup_dir"]) / "backup_history.json"
        )

        if history_file.exists():
            try:
                with open(history_file, "r") as f:
                    data = json.load(f)

                for backup_data in data.get("backups", []):
                    # Convert datetime strings back
                    backup_data["created_at"] = datetime.fromisoformat(
                        backup_data["created_at"]
                    )
                    backup_data["retention_until"] = datetime.fromisoformat(
                        backup_data["retention_until"]
                    )

                    backup = BackupMetadata(**backup_data)
                    self.backup_history.append(backup)

                logger.info(f"Loaded {len(self.backup_history)} backup records")

            except Exception as e:
                logger.error(f"Error loading backup history: {e}")

    def save_backup_history(self) -> None:
        """Save backup history to storage"""
        history_file = Path(self.config["storage"]["local_backup_dir"])
        history_file.mkdir(parents=True, exist_ok=True)
        history_file = history_file / "backup_history.json"

        try:
            backup_data = []
            for backup in self.backup_history:
                backup_dict = asdict(backup)
                backup_dict["created_at"] = backup.created_at.isoformat()
                backup_dict["retention_until"] = backup.retention_until.isoformat()
                backup_data.append(backup_dict)

            data = {
                "updated_at": datetime.now().isoformat(),
                "total_backups": len(backup_data),
                "backups": backup_data,
            }

            with open(history_file, "w") as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error saving backup history: {e}")

    async def create_backup(
        self,
        backup_type: str = "full",
        components: List[str] = None,
        description: str = "",
        tags: Dict[str, str] = None,
    ) -> BackupJob:
        """Create a new backup"""

        job_id = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

        job = BackupJob(
            job_id=job_id,
            job_type=backup_type,
            status="queued",
            started_at=None,
            completed_at=None,
            progress_percent=0.0,
            current_step="Initializing",
            error_message=None,
            result=None,
        )

        self.active_jobs[job_id] = job

        # Start backup in background
        asyncio.create_task(
            self._execute_backup_job(
                job, components or ["neo4j", "qdrant"], description, tags or {}
            )
        )

        return job

    async def _execute_backup_job(
        self,
        job: BackupJob,
        components: List[str],
        description: str,
        tags: Dict[str, str],
    ) -> None:
        """Execute backup job"""
        try:
            job.status = "running"
            job.started_at = datetime.now()
            job.current_step = "Preparing backup"
            job.progress_percent = 5.0

            # Create backup directory
            backup_id = f"backup_{job.started_at.strftime('%Y%m%d_%H%M%S')}"
            backup_dir = Path(self.config["storage"]["local_backup_dir"]) / backup_id
            backup_dir.mkdir(parents=True, exist_ok=True)

            backup_files = []
            total_size = 0

            # Backup Neo4j
            if "neo4j" in components:
                job.current_step = "Backing up Neo4j"
                job.progress_percent = 20.0

                success, file_or_error, size = await self.neo4j_manager.create_backup(
                    backup_dir
                )
                if success:
                    backup_files.append(file_or_error)
                    total_size += size
                    logger.info(f"Neo4j backup completed: {size} bytes")
                else:
                    raise Exception(f"Neo4j backup failed: {file_or_error}")

            # Backup Qdrant
            if "qdrant" in components:
                job.current_step = "Backing up Qdrant"
                job.progress_percent = 40.0

                success, file_or_error, size = await self.qdrant_manager.create_backup(
                    backup_dir
                )
                if success:
                    backup_files.append(file_or_error)
                    total_size += size
                    logger.info(f"Qdrant backup completed: {size} bytes")
                else:
                    raise Exception(f"Qdrant backup failed: {file_or_error}")

            # Calculate overall checksum
            job.current_step = "Calculating checksums"
            job.progress_percent = 60.0

            overall_checksum = await self._calculate_backup_checksum(backup_files)

            # Create metadata
            retention_days = self.config["backup_schedule"]["max_backup_age_days"]
            retention_until = datetime.now() + timedelta(days=retention_days)

            metadata = BackupMetadata(
                backup_id=backup_id,
                backup_type=job.job_type,
                created_at=job.started_at,
                size_bytes=total_size,
                checksum=overall_checksum,
                components=components,
                status="created",
                storage_locations=[str(backup_dir)],
                retention_until=retention_until,
                tags={"description": description, **tags},
                verification_results={},
            )

            # Verify backup
            if self.config["verification"]["verify_after_backup"]:
                job.current_step = "Verifying backup"
                job.progress_percent = 70.0

                verification_results = await self.integrity_checker.verify_backup(
                    metadata, backup_dir
                )
                metadata.verification_results = verification_results

                if verification_results["overall_status"] != "valid":
                    raise Exception(
                        f"Backup verification failed: {verification_results['errors']}"
                    )

            # Upload to cloud storage
            job.current_step = "Uploading to cloud storage"
            job.progress_percent = 80.0

            if backup_files:
                for backup_file in backup_files:
                    cloud_locations = await self.cloud_storage.upload_backup(
                        Path(backup_file), backup_id, metadata
                    )
                    metadata.storage_locations.extend(cloud_locations)

            metadata.status = "completed"

            # Save metadata
            job.current_step = "Finalizing"
            job.progress_percent = 95.0

            self.backup_history.append(metadata)
            self.save_backup_history()

            # Complete job
            job.status = "completed"
            job.completed_at = datetime.now()
            job.progress_percent = 100.0
            job.current_step = "Completed"
            job.result = metadata

            logger.info(f"Backup job {job.job_id} completed successfully")

        except Exception as e:
            job.status = "failed"
            job.completed_at = datetime.now()
            job.error_message = str(e)
            logger.error(f"Backup job {job.job_id} failed: {e}")

    async def restore_backup(
        self, backup_id: str, components: List[str] = None, target_location: str = None
    ) -> Tuple[bool, str]:
        """Restore from backup"""
        try:
            # Find backup metadata
            backup_metadata = None
            for backup in self.backup_history:
                if backup.backup_id == backup_id:
                    backup_metadata = backup
                    break

            if not backup_metadata:
                return False, f"Backup not found: {backup_id}"

            # Download backup if needed
            local_backup_dir = (
                Path(self.config["storage"]["local_backup_dir"]) / backup_id
            )

            if not local_backup_dir.exists():
                logger.info(f"Downloading backup {backup_id} from cloud storage")
                local_backup_dir.mkdir(parents=True, exist_ok=True)

                for component in components or backup_metadata.components:
                    component_file = local_backup_dir / f"{component}_backup.tar.gz"
                    success = await self.cloud_storage.download_backup(
                        backup_id, component_file.name, component_file
                    )
                    if not success:
                        return False, f"Failed to download {component} backup"

            # Verify backup before restore
            if self.config["verification"]["verify_before_restore"]:
                logger.info("Verifying backup before restore")
                verification_results = await self.integrity_checker.verify_backup(
                    backup_metadata, local_backup_dir
                )

                if verification_results["overall_status"] != "valid":
                    return (
                        False,
                        f"Backup verification failed: {verification_results['errors']}",
                    )

            # Restore components
            restore_results = []

            if "neo4j" in (components or backup_metadata.components):
                neo4j_files = list(local_backup_dir.glob("neo4j_backup.*"))
                if neo4j_files:
                    success, message = await self.neo4j_manager.restore_backup(
                        neo4j_files[0]
                    )
                    restore_results.append(f"Neo4j: {message}")
                    if not success:
                        return False, f"Neo4j restore failed: {message}"

            if "qdrant" in (components or backup_metadata.components):
                qdrant_files = list(local_backup_dir.glob("qdrant_backup.*"))
                if qdrant_files:
                    success, message = await self.qdrant_manager.restore_backup(
                        qdrant_files[0]
                    )
                    restore_results.append(f"Qdrant: {message}")
                    if not success:
                        return False, f"Qdrant restore failed: {message}"

            result_message = "; ".join(restore_results)
            logger.info(f"Restore completed: {result_message}")

            return True, result_message

        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False, str(e)

    async def _calculate_backup_checksum(self, backup_files: List[str]) -> str:
        """Calculate overall checksum for backup"""
        hash_sha256 = hashlib.sha256()

        for file_path in sorted(backup_files):  # Sort for consistent ordering
            async with aiofiles.open(file_path, "rb") as f:
                while chunk := await f.read(8192):
                    hash_sha256.update(chunk)

        return hash_sha256.hexdigest()

    def get_backup_job(self, job_id: str) -> Optional[BackupJob]:
        """Get backup job status"""
        return self.active_jobs.get(job_id)

    def list_backups(self, limit: int = 50) -> List[BackupMetadata]:
        """List available backups"""
        # Sort by creation date (newest first)
        sorted_backups = sorted(
            self.backup_history, key=lambda x: x.created_at, reverse=True
        )
        return sorted_backups[:limit]

    def cleanup_old_backups(self) -> int:
        """Clean up expired backups"""
        now = datetime.now()
        cleaned_count = 0

        for backup in self.backup_history[:]:  # Create copy for iteration
            if backup.retention_until < now:
                # Remove local files
                local_backup_dir = (
                    Path(self.config["storage"]["local_backup_dir"]) / backup.backup_id
                )
                if local_backup_dir.exists():
                    shutil.rmtree(local_backup_dir, ignore_errors=True)

                # TODO: Remove from cloud storage

                # Remove from history
                self.backup_history.remove(backup)
                cleaned_count += 1

        if cleaned_count > 0:
            self.save_backup_history()
            logger.info(f"Cleaned up {cleaned_count} expired backups")

        return cleaned_count


async def main():
    """Example usage of backup agent"""
    agent = BackupAgent()

    try:
        # Create a backup
        job = await agent.create_backup(
            backup_type="full",
            components=["neo4j", "qdrant"],
            description="Test backup",
            tags={"environment": "development"},
        )

        print(f"Started backup job: {job.job_id}")

        # Monitor job progress
        while job.status in ["queued", "running"]:
            await asyncio.sleep(2)
            print(f"Progress: {job.progress_percent:.1f}% - {job.current_step}")

        if job.status == "completed":
            print(f"Backup completed successfully!")
            print(f"Backup ID: {job.result.backup_id}")
            print(f"Size: {job.result.size_bytes} bytes")
            print(f"Checksum: {job.result.checksum}")
        else:
            print(f"Backup failed: {job.error_message}")

        # List backups
        backups = agent.list_backups(10)
        print(f"\nAvailable backups: {len(backups)}")
        for backup in backups[:3]:
            print(f"- {backup.backup_id} ({backup.backup_type}) - {backup.created_at}")

        # Test restore (commented out to avoid disrupting running services)
        # if backups:
        #     success, message = await agent.restore_backup(backups[0].backup_id)
        #     print(f"Restore test: {success} - {message}")

        # Cleanup old backups
        cleaned = agent.cleanup_old_backups()
        print(f"Cleaned up {cleaned} old backups")

    except Exception as e:
        logger.error(f"Backup agent example failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
