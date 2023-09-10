setup:
    pdm venv create --force --with-pip 3.10
    pdm use .venv
    pdm install