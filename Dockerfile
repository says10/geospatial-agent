# --- Stage 1: The Builder ---
FROM python:3.10-slim-bookworm AS builder

# Set up non-interactive environment for package installation
ENV DEBIAN_FRONTEND=noninteractive

# --- The HTTPS Fix ---
# 1. Install the tool that allows apt to speak HTTPS.
# 2. Change the package source list to use https:// instead of http://
# 3. Then, run the update and install commands. This is the most robust method.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates apt-transport-https && \
    sed -i 's/http:/https:/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install and configure Poetry
RUN pip install "poetry==1.8.2"
RUN poetry config virtualenvs.in-project true

# Copy dependency files and install them.
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-interaction --no-ansi --no-root --no-dev


# --- Stage 2: The Final Production Image ---
FROM python:3.10-slim-bookworm

# Also apply the HTTPS fix in the final stage before installing gdal-bin
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates apt-transport-https && \
    sed -i 's/http:/https:/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the virtual environment created in the builder stage
COPY --from=builder /build/.venv /app/.venv

# Add the venv to the PATH.
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Copy our application source code and tests
COPY ./src /app/src
COPY ./tests /app/tests

# Set a default command to an interactive shell for debugging
CMD ["bash"]