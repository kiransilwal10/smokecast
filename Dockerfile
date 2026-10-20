FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir numpy pandas pyyaml fastapi "uvicorn[standard]"

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

ENV SMOKECAST_BUNDLE=/app/serving
EXPOSE 8000
CMD ["uvicorn", "smokecast.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
