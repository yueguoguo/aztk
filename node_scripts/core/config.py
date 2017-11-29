import os
import logging
import azure.batch.batch_service_client as batch
import azure.batch.batch_auth as batchauth
from azure.common.credentials import ServicePrincipalCredentials

account_name = os.environ["AZ_BATCH_ACCOUNT_NAME"]
account_key = os.environ["BATCH_ACCOUNT_KEY"]
service_url = os.environ["BATCH_SERVICE_URL"]
resource_url = os.environ["BATCH_RESOURCE_URL"]
tenant_id = os.environ["BATCH_TENANT_ID"]
client_id = os.environ["BATCH_CLIENT_ID"]
credential = os.environ["BATCH_CREDENTIAL"]

pool_id = os.environ["AZ_BATCH_POOL_ID"]
node_id = os.environ["AZ_BATCH_NODE_ID"]
is_dedicated = os.environ["AZ_BATCH_NODE_IS_DEDICATED"]

spark_web_ui_port = os.environ["SPARK_WEB_UI_PORT"]
spark_worker_ui_port = os.environ["SPARK_WORKER_UI_PORT"]
spark_jupyter_port = os.environ["SPARK_JUPYTER_PORT"]
spark_job_ui_port = os.environ["SPARK_JOB_UI_PORT"]

def get_client() -> batch.BatchServiceClient:
    if not resource_url:
        credentials = batchauth.SharedKeyCredentials(
            account_name,
            account_key)
    else:
        credentials = ServicePrincipalCredentials(
            client_id=client_id,
            secret=credential,
            tenant=tenant_id,
            resource=resource_url)
    return batch.BatchServiceClient(credentials, base_url=service_url)

batch_client = get_client()

logging.info("Pool id is %s", pool_id)
logging.info("Node id is %s", node_id)
logging.info("Batch account name %s", account_name)
logging.info("Is dedicated %s", is_dedicated)
