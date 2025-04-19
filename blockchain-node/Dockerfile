# Build stage
FROM golang:1.20-alpine AS builder

# Install build dependencies
RUN apk add --no-cache gcc musl-dev

# Set working directory
WORKDIR /app

# Copy go mod and sum files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=1 GOOS=linux go build -o phantomx-node

# Final stage
FROM alpine:latest

# Install runtime dependencies
RUN apk add --no-cache ca-certificates tzdata

# Create non-root user
RUN adduser -D -g '' phantomx

# Set working directory
WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/phantomx-node .
COPY --from=builder /app/config.yaml .

# Create data directory
RUN mkdir -p /app/data && chown -R phantomx:phantomx /app

# Switch to non-root user
USER phantomx

# Expose ports
EXPOSE 30303

# Set environment variables
ENV PHANTOMX_NETWORK_ID=1
ENV PHANTOMX_DATA_DIR=/app/data

# Run the application
CMD ["./phantomx-node"]