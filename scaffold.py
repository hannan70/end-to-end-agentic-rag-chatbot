import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")

list_of_files = [
    "artifacts/model/*",
    "artifacts/data/*",
    "notebooks/experiments.ipynb",
    "docker/Dockerfile",
    "docker/docker-compose.yml",
    ".env",
    "requirements.txt",
    "static/*",
    "templates/index.html",
    "app.py",
    "setup.py",
    "test.py",
]


for filepath in list_of_files:
    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)

    # make dir
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    # # make file
    if str(filepath).endswith("*"):
        pass
    else:
        if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
            with open(filepath, "w") as f:
                pass
                logging.info(f"Creating empty file: {filepath}")

        else:
            logging.info(f"{filename} is already exists")
