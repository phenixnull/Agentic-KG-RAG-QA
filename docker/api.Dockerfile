# 使用轻量级Python基础镜像
FROM python:3.12-slim
COPY --from=node:20-slim /usr/local/bin /usr/local/bin
COPY --from=node:20-slim /usr/local/lib/node_modules /usr/local/lib/node_modules
COPY --from=node:20-slim /usr/local/include /usr/local/include
COPY --from=node:20-slim /usr/local/share /usr/local/share

# 设置工作目录
WORKDIR /app

# 环境变量设置
ENV TZ=Asia/Shanghai \
    UV_PROJECT_ENVIRONMENT="/usr/local" \
    UV_COMPILE_BYTECODE=1 \
    UV_DEFAULT_INDEX="https://pypi.org/simple" \
    PIP_INDEX_URL="https://pypi.org/simple" \
    DEBIAN_FRONTEND=noninteractive

# 设置 npm 镜像源，为 MCP 和 Skills 安装依赖
RUN npm config set registry https://registry.npmmirror.com --global \
    && npm cache clean --force

# 设置代理和时区，安装系统依赖 - 合并为一个RUN减少层数
RUN set -ex \
    # (A) 设置时区
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    # (B) 切换到更稳定的 Debian HTTPS 镜像，避免 deb.debian.org 间歇性 502
    && if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i 's|http://deb.debian.org/debian|https://mirrors.ustc.edu.cn/debian|g; s|http://deb.debian.org/debian-security|https://mirrors.ustc.edu.cn/debian-security|g' /etc/apt/sources.list.d/debian.sources; \
      fi \
    # (B) 安装必要的系统库
    && apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install -y --no-install-recommends --fix-missing \
        ca-certificates \
        curl \
        git \
        libpq5 \
        libsm6 \
        libxext6 \
    && python -m pip install --no-cache-dir uv==0.7.2 \
    # (C) 清理垃圾，减小体积
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制项目配置文件
COPY ../backend/pyproject.toml /app/pyproject.toml
COPY ../backend/.python-version /app/.python-version
COPY ../backend/uv.lock /app/uv.lock

# 先复制 package 目录，因为 pyproject.toml 中 yuxi = { path = "package", editable = true }
COPY ../backend/package /app/package

# 如果网络还是不好，可以在后面添加 --index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --group test --no-dev --frozen

# 激活虚拟环境并添加到PATH
ENV PATH="/app/.venv/bin:$PATH"

# 复制 server 代码
COPY ../backend/server /app/server
