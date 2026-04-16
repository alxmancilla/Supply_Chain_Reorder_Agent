# ── Reorder Alert Agent ────────────────────────────────────────────────────
# Single image used by all four services (seeder / agent / simulator / app).
# Each service overrides CMD in docker-compose.yml.
# ---------------------------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

# Install Python dependencies before copying code so this layer is cached
# unless requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Streamlit port (only bound by the `app` service)
EXPOSE 8501

# No default command — each service in docker-compose.yml supplies its own.
