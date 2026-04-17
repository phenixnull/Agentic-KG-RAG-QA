# GPU-flavored API/worker image for local embedding and reranker acceleration.
FROM python:3.12-slim
COPY --from=node:20-slim /usr/local/bin /usr/local/bin
COPY --from=node:20-slim /usr/local/lib/node_modules /usr/local/lib/node_modules
COPY --from=node:20-slim /usr/local/include /usr/local/include
COPY --from=node:20-slim /usr/local/share /usr/local/share

WORKDIR /app

ENV TZ=Asia/Shanghai \
    UV_PROJECT_ENVIRONMENT="/usr/local" \
    UV_COMPILE_BYTECODE=1 \
    UV_DEFAULT_INDEX="https://pypi.org/simple" \
    PIP_INDEX_URL="https://pypi.org/simple" \
    DEBIAN_FRONTEND=noninteractive

RUN npm config set registry https://registry.npmmirror.com --global \
    && npm cache clean --force

RUN set -ex \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i 's|http://deb.debian.org/debian|https://mirrors.ustc.edu.cn/debian|g; s|http://deb.debian.org/debian-security|https://mirrors.ustc.edu.cn/debian-security|g' /etc/apt/sources.list.d/debian.sources; \
      fi \
    && apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install -y --no-install-recommends --fix-missing \
        ca-certificates \
        curl \
        git \
        libpq5 \
        libsm6 \
        libxext6 \
    && python -m pip install --no-cache-dir uv==0.7.2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY ../backend/pyproject.toml /app/pyproject.toml
COPY ../backend/.python-version /app/.python-version
COPY ../backend/uv.lock /app/uv.lock
COPY ../backend/package /app/package

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --group test --no-dev --frozen \
    && python -m pip install --no-cache-dir --force-reinstall \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.8.0 torchvision==0.23.0

ENV PATH="/app/.venv/bin:$PATH"

COPY ../backend/server /app/server
