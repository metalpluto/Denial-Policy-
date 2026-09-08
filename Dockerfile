# Lightweight base image  Slim variant keeps the image small
# while still having what scikit-learn needs to install cleanly.
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first, separately from the app code. Docker
# caches each instruction as a layer as long as requirements.txt
# hasn't changed, this layer is reused on rebuilds instead of
# reinstalling everything every time you change a .py file.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual application code
COPY . .

# GOOGLE_API_KEY is intentionally NOT set here  it should never be
# baked into an image. Pass it at run time instead (see README),
# so the key never ends up committed to a Docker layer or pushed to
# a registry by accident.

ENTRYPOINT ["python", "main.py"]
# Default query if none is provided at `docker run` time — this is
# fully overridden by any arguments you pass after the image name.
CMD ["--query", "How long do I have to file an appeal?", "--no-generate"]
