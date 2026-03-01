import sys
import json
import os
import datetime
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, command
from azure.ai.ml import Input, Output


# ====== CONFIG (from App Service env) ======
SUBSCRIPTION_ID = os.environ.get("AZ_SUBSCRIPTION_ID")
RESOURCE_GROUP = os.environ.get("AZ_RESOURCE_GROUP")
WORKSPACE_NAME = os.environ.get("AZ_ML_WORKSPACE")

COMPUTE_NAME = os.environ.get("AZ_ML_COMPUTE", "RGN-Compute-Cluster")
ENV_NAME = os.environ.get("AZ_ML_ENV", "RGN_Env:1")  # 推荐加版本号
DATASTORE_NAME = os.environ.get("AZ_ML_DATASTORE", "rgnresults_ds")

# settings 已经由 server.js 保存到 run/settings.json
SETTINGS_URI = os.environ.get(
    "AML_SETTINGS_URI",
    "azureml://datastores/rgnresults_ds/paths/run/settings.json"
)


def main():
    # ---- 1) read settings from stdin (optional, mostly for logging) ----
    try:
        settings = json.loads(sys.stdin.read() or "{}")
    except Exception:
        settings = {}

    # ---- 2) generate job name ----
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    job_name = f"webrun_{timestamp}"

    # ---- 3) connect to AML ----
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential,
        SUBSCRIPTION_ID,
        RESOURCE_GROUP,
        WORKSPACE_NAME,
    )

    # ---- 4) define job ----
    job = command(
        ccode="git+https://github.com/JiyaoPu/RuleGeneration.git#main:aml_component_train", 
        command=(
            "pip install -r requirements.txt && "
            "python main.py "
            "--settings_json ${{inputs.settings_json}} "
            "--outputs_dir ${{outputs.outputs_dir}}"
        ),
        inputs={
            "settings_json": Input(
                type="uri_file",
                path=SETTINGS_URI
            )
        },
        outputs={
            "outputs_dir": Output(
                type="uri_folder",
                path=f"azureml://datastores/{DATASTORE_NAME}/paths/{job_name}/"
            )
        },
        environment=ENV_NAME,
        compute=COMPUTE_NAME,
        experiment_name="web_run",
        display_name=job_name,
    )

    # ---- 5) submit ----
    returned_job = ml_client.jobs.create_or_update(job)

    studio_url = returned_job.studio_url

    # ---- 6) return JSON to Node ----
    print(json.dumps({
        "job_name": job_name,
        "studio_url": studio_url
    }))


if __name__ == "__main__":
    main()