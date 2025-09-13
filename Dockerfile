FROM python:3.13-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen

COPY . .

EXPOSE 3000

ENV PORT=3000

CMD ["uv", "run", "python", "main.py"]
