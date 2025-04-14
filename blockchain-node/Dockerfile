# Build stage
FROM golang:1.20-alpine AS builder

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache gcc musl-dev linux-headers git

# Copy go mod files
COPY go.mod ./
COPY go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=1 GOOS=linux go build -a -installsuffix cgo -o phantomx-node .

# Final stage
FROM alpine:latest

WORKDIR /app

# Install runtime dependencies
RUN apk add --no-cache ca-certificates

# Copy binary from builder
COPY --from=builder /app/phantomx-node .
COPY config.yaml .

# Create data directory
RUN mkdir -p /data

# Expose ports
EXPOSE 8545 8546 8547 30301/tcp 30301/udp

# Set environment variables
ENV PHANTOMX_ENV=production

# Run the application
CMD ["./phantomx-node"]