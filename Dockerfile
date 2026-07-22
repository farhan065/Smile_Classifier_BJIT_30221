# ---- Base image ---------------------------------------------------------- #
# Official, minimal Python 3.12. "slim" = small footprint.
FROM python:3.12-slim

# ---- Environment settings ------------------------------------------------ #
# Don't write .pyc files, and don't buffer stdout/stderr (so logs appear
# immediately in `docker logs` — important for debugging containers).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---- System dependencies ------------------------------------------------- #
# psycopg2 and Pillow need a few system libraries. We install them, then
# clean up the apt cache in the SAME layer to keep the image small.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        libjpeg62-turbo \
        zlib1g \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Working directory --------------------------------------------------- #
# All following commands run inside /app in the image.
WORKDIR /app

# ---- Python dependencies (cached layer) ---------------------------------- #
# Copy ONLY requirements first so this layer is cached and not rebuilt
# every time application code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- Application code ----------------------------------------------------- #
# Now copy the rest of the project (respecting .dockerignore).
COPY . .

# ---- Non-root user (security best practice) ------------------------------ #
# Create an unprivileged user and hand over ownership of /app, then switch
# to that user so the app doesn't run as root.
RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app
USER appuser

# ---- Runtime ------------------------------------------------------------- #
# Document the port the app listens on.
EXPOSE 8000

# Start the production server. NOTE: no --reload (that's a dev-only feature).
# We bind to 0.0.0.0 so the server is reachable from OUTSIDE the container,
# not just from within it (127.0.0.1 would be unreachable from your browser).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]