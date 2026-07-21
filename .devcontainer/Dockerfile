FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

# System dependencies for building pyoomph from source (see INSTALL.md)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ccache \
    git \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    libopenmpi-dev \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libginac-dev \
    libcln-dev \
    libgmp-dev \
    libglu1-mesa \
    libxcursor1 \
    libopengl0 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

USER vscode

RUN python3 -m venv /home/vscode/venv
ENV PATH="/home/vscode/venv/bin:${PATH}"

RUN pip install --no-cache-dir \
    gmsh mkl mpi4py matplotlib numpy petsc4py nanobind pygmsh scipy meshio \
    scikit-build-core pytest
