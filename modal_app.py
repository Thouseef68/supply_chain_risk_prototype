import modal


APP_NAME = "supplychainiq"

RELEASE_BASE_URL = (
    "https://github.com/Thouseef68/"
    "supply_chain_risk_prototype/releases/download/v1.0.0"
)


# Persistent storage
models_volume = modal.Volume.from_name(
    "supplychainiq-models",
    create_if_missing=True
)

data_volume = modal.Volume.from_name(
    "supplychainiq-data",
    create_if_missing=True
)

outputs_volume = modal.Volume.from_name(
    "supplychainiq-outputs",
    create_if_missing=True
)


# Python environment
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(
        "backend/requirements.txt"
    )
    .add_local_dir(
        "backend",
        "/root/backend",
        copy=False
    )
)


app = modal.App(
    APP_NAME,
    image=image
)


@app.function(
    memory=4096,
    cpu=2,
    timeout=3600,
    volumes={
        "/root/models": models_volume,
        "/root/data": data_volume,
        "/root/outputs": outputs_volume,
    },
    env={
        "MODEL_RELEASE_BASE_URL": RELEASE_BASE_URL,
    },
)
@modal.asgi_app()
def fastapi_app():
    import sys

    sys.path.insert(
        0,
        "/root/backend"
    )

    from main import app as api

    # Persist newly downloaded runtime assets.
    models_volume.commit()
    data_volume.commit()

    return api