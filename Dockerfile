# Taken from: https://sourcery.ai/blog/python-docker/

# From https://groups.google.com/u/0/g/recsys-challenge2021/c/_PSoFy9Lflw/m/H4zuvs1hAQAJ
# to answer some questions, a container from your image will be launched with
# workdir '/home/{submission_identifier}', where {submission_identifier} is a
# unique hash. All files will be put in this location. That means that
# `/home/{submission_identifier}/test` contains the test set, and
# `/home/{submission_identifier}/` contains `run` and all other files / folders
# in your zip.
# results.csv should also be written to /home/{submission_identifier}/results.csv

# docker build -t malto-submission-image . && docker run --name smol-boi --rm --mount type=bind,source="$(pwd)",destination=/home/cafebabe,readonly --workdir="/home/cafebabe" malto-submission-image:latest pwd

################################################################################
FROM python:3.7-slim as base

# Setup env
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONFAULTHANDLER=1

################################################################################
# Stage 1: python-deps
FROM base AS python-deps

# Install pipenv and compilation dependencies
RUN pip install pipenv
RUN apt-get update && apt-get install -y --no-install-recommends gcc

# Install python dependencies in /.venv
COPY Pipfile .
COPY Pipfile.lock .
RUN PIPENV_VENV_IN_PROJECT=1 pipenv install --deploy

################################################################################
# Stage 2: runtime
FROM base AS runtime

# Install Java 8
RUN mkdir -p /usr/share/man/man1/ && \
    apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    apt-add-repository 'deb http://security.debian.org/debian-security stretch/updates main' && \
    apt-get update && \
    apt-get install -y --no-install-recommends openjdk-8-jre-headless && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

# Copy virtual env from python-deps stage
COPY --from=python-deps /.venv /.venv
ENV PATH="/.venv/bin:$PATH"

# No ENTRYPOINT and no CMD
