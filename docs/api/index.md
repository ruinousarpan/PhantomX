# API Overview

The Python AI Core API provides a comprehensive set of endpoints for interacting with the system. This section provides an overview of the available endpoints and how to use them.

## Authentication

All API endpoints require authentication using a JWT token. Include the token in the `Authorization` header:

```
Authorization: Bearer your-jwt-token
```

## Base URL

The base URL for all API endpoints is:

```
https://api.python-ai-core.com/v1
```

## API Endpoints

The API is organized into the following categories:

### Data Validation

Endpoints for validating data against schemas and rules.

### Data Lineage

Endpoints for tracking and querying data lineage information.

### Data Versioning

Endpoints for managing data versions and rollbacks.

### Data Backup

Endpoints for creating and restoring backups.

### Data Monitoring

Endpoints for monitoring system health and performance.

### Data Analytics

Endpoints for performing analytics and generating insights.

### Data Reporting

Endpoints for generating and managing reports.

## Response Format

All API responses are in JSON format and follow this structure:

```json
{
  "status": "success",
  "data": {
    // Response data
  },
  "message": "Optional message"
}
```

## Error Handling

Errors are returned with appropriate HTTP status codes and a JSON response:

```json
{
  "status": "error",
  "error": {
    "code": "ERROR_CODE",
    "message": "Error message"
  }
}
```

## Rate Limiting

API requests are rate-limited to prevent abuse. The current limits are:

- 100 requests per minute for authenticated users
- 10 requests per minute for unauthenticated users

Rate limit information is included in the response headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1620000000
```

## Pagination

Endpoints that return lists of items support pagination using the following query parameters:

- `page`: Page number (default: 1)
- `per_page`: Items per page (default: 20, max: 100)

Example:

```
GET /api/v1/data?page=2&per_page=50
```

## Versioning

The API is versioned using the URL path. The current version is v1.

## SDKs and Libraries

We provide official SDKs for the following languages:

- Python
- JavaScript
- Java
- Go

See the [SDKs and Libraries](sdk.md) page for more information. 