import azure.batch.batch_service_client as batch
import azure.batch.batch_auth as batch_auth
import azure.storage.blob as blob
from aztk_sdk import error
from aztk_sdk.version import __version__
from azure.common.credentials import ServicePrincipalCredentials
from typing import Optional


class BatchConfig:
    def __init__(self, service_url: str, account_key: Optional[str]=None, account_name: Optional[str]=None,
                 resource_url: Optional[str]=None, tenant_id: Optional[str]=None, client_id: Optional[str]=None,
                 credential: Optional[str]=None):
        self.service_url = service_url
        self.account_key = account_key
        self.account_name = account_name
        self.resource_url = resource_url
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.credential = credential


class BlobConfig:
    def __init__(self, account_key: str, account_name: str, account_suffix: str):
        self.account_key = account_key
        self.account_name = account_name
        self.account_suffix = account_suffix


def _validate_batch_config(batch_config: BatchConfig):
    if batch_config.service_url is None:
        raise error.AzureApiInitError("Batch service url is not set in secrets.yaml config")
    if batch_config.resource_url is None:
        if batch_config.account_name is None:
            raise error.AzureApiInitError("Neither Batch resource url nor account name is not set in secrets.yaml config")
        if batch_config.account_key is None:
            raise error.AzureApiInitError("Neither Batch resource url nor account key is not set in secrets.yaml config")
    else:
        if batch_config.account_name is not None:
            raise error.AzureApiInitError("Both Batch resource url and account name are set in secrets.yaml config, can use either shared secret or service principal login, not both")
        if batch_config.account_key is not None:
            raise error.AzureApiInitError("Both Batch resource url and account key are set in secrets.yaml config, can use either shared secret or service principal login, not both")
        if batch_config.tenant_id is None:
            raise error.AzureApiInitError("Batch resource url is set but serviceprincipal.tenantid is not set in secrets.yaml config")
        if batch_config.client_id is None:
            raise error.AzureApiInitError("Batch resource url is set but serviceprincipal.clientid is not set in secrets.yaml config")
        if batch_config.credential is None:
            raise error.AzureApiInitError("Batch resource url is set but serviceprincipal.credential is not set in secrets.yaml config")


def make_batch_client(batch_config: BatchConfig):
    """
        Creates a batch client object
        :param str batch_account_key: batch account key
        :param str batch_account_name: batch account name
        :param str batch_service_url: batch service url
    """
    # Validate the given config
    _validate_batch_config(batch_config)

    if batch_config.resource_url is None:
        # Set up SharedKeyCredentials
        credentials = batch_auth.SharedKeyCredentials(
            batch_config.account_name,
            batch_config.account_key)
    else:
        # Set up ServicePrincipalCredentials
        credentials = ServicePrincipalCredentials(
            client_id=batch_config.client_id,
            secret=batch_config.credential,
            tenant=batch_config.tenant_id,
            resource=batch_config.resource_url)

    # Set up Batch Client
    batch_client = batch.BatchServiceClient(
        credentials,
        base_url=batch_config.service_url)

    # Set retry policy
    batch_client.config.retry_policy.retries = 5
    batch_client.config.add_user_agent('aztk/{}'.format(__version__))

    return batch_client


def _validate_blob_config(blob_config: BlobConfig):
    if blob_config.account_key is None:
        raise error.AzureApiInitError("Storage account key is not set in secrets.yaml config")
    if blob_config.account_name is None:
        raise error.AzureApiInitError("Storage account name is not set in secrets.yaml config")
    if blob_config.account_suffix is None:
        raise error.AzureApiInitError("Storage account suffix is not set in secrets.yaml config")


def make_blob_client(blob_config: BlobConfig):
    """
        Creates a blob client object
        :param str storage_account_key: storage account key
        :param str storage_account_name: storage account name
        :param str storage_account_suffix: storage account suffix
    """
    # Validate Blob config
    _validate_blob_config(blob_config)

    # Set up BlockBlobStorage
    blob_client = blob.BlockBlobService(
        account_name=blob_config.account_name,
        account_key=blob_config.account_key,
        endpoint_suffix=blob_config.account_suffix)

    return blob_client
