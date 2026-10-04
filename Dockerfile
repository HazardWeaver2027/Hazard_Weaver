FROM python:3.11-slim
WORKDIR /workspace/hazardweaver
COPY . .
RUN pip install --no-cache-dir -e ".[dev]"
RUN make verify && make smoke && make reproduce-paper
