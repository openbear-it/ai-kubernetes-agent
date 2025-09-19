
# syntax=docker/dockerfile:1.4
FROM --platform=$BUILDPLATFORM debian:bullseye-slim
ARG TARGETARCH
ARG MCP_VERSION=latest
WORKDIR /app

# Download the correct binary for the target architecture and version
ADD https://github.com/containers/kubernetes-mcp-server/releases/${MCP_VERSION}/download/kubernetes-mcp-server-linux-${TARGETARCH} /usr/local/bin/kubernetes-mcp-server
RUN chmod +x /usr/local/bin/kubernetes-mcp-server

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/kubernetes-mcp-server"]
CMD ["--port", "8080"]
