FROM python:3.11

# Instalar Node.js (>=18) e ferramentas necessárias
RUN apt-get update \
  && apt-get install -y --no-install-recommends nodejs npm \
  && rm -rf /var/lib/apt/lists/*

# Copiar uv da imagem oficial
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

WORKDIR /app

# Copiar arquivos de dependências primeiro para aproveitar o cache
COPY package.json package-lock.json ./
COPY frontend/package.json frontend/package-lock.json ./frontend/
COPY backend/pyproject.toml backend/uv.lock ./backend/

# Instalar dependências (Node + Python)
RUN npm ci \
  && npm ci --prefix frontend \
  && cd backend && uv sync --frozen

# Copiar código-fonte do projeto
COPY . .

EXPOSE 3000 5001

# Iniciar frontend e backend simultaneamente (modo de desenvolvimento)
CMD ["npm", "run", "dev"]