FROM python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

COPY requirements.txt /opt/toolsmith/requirements.txt
RUN python -m pip install --no-cache-dir --no-compile --disable-pip-version-check -r /opt/toolsmith/requirements.txt \
    && mkdir -p /workspace/src /workspace/tests \
    && chmod 755 /workspace /workspace/src /workspace/tests
COPY runner.py /opt/toolsmith/runner.py
RUN chmod 555 /opt/toolsmith/runner.py

USER 65532:65532
WORKDIR /workspace
CMD ["python", "-I", "/opt/toolsmith/runner.py", "visible"]
