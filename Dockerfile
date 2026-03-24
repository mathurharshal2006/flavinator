# Dockerfile
#
# Recipe for building the Flavinator image.
# Docker reads this file top to bottom and executes each instruction.
#
# WHY EACH INSTRUCTION EXISTS:
#   FROM    - which base image to start from
#   WORKDIR - where inside the container our code lives
#   COPY    - copy files from your machine into the container
#   RUN     - execute a command while building the image
#   EXPOSE  - tell Docker which port this container uses
#   CMD     - what to run when container starts

# START FROM official Python 3.12 on slim Ubuntu
# WHY slim: full Ubuntu image is 900MB, slim is 150MB
# We only need Python, not the entire OS
FROM python:3.12-slim

# SET working directory inside container
# All future commands run from this folder
# WHY /app: industry standard name for application directory
WORKDIR /app

# COPY requirements first, before copying code
# WHY THIS ORDER:
#   Docker caches each step. If requirements.txt has not changed,
#   Docker skips the pip install step (saves 5 minutes every build).
#   If we copied all code first, any code change would
#   invalidate the cache and reinstall ALL packages every time.
COPY requirements.txt .

# INSTALL packages
# --no-cache-dir: do not store download cache (saves disk space)
# --upgrade pip: make sure pip itself is current
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# COPY rest of the code into container
# The . . means: copy everything from current folder to /app
COPY . .

# EXPOSE the ports our services use
# WHY EXPOSE: documents which ports the container listens on
# Does not actually open the port - docker-compose does that
EXPOSE 8000
EXPOSE 8501

# DEFAULT command when container starts
# This runs the FastAPI server
# Can be overridden in docker-compose.yml for Streamlit container
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]