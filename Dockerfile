FROM python:3.10-slim

# Set up a new user named "user" with UID 1000 (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user

# Set working directory
WORKDIR /home/user/app

# Install dependencies as user
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all repository files
COPY --chown=user . .

# Switch to non-root user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860

# Expose default Hugging Face Spaces port
EXPOSE 7860

# Launch the ConG-MoE v2 web app
CMD ["python", "run_web.py"]
