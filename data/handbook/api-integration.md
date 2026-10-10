# API integration

## Authentication
The REST API is served at https://api.kestrel-parcel.example/v2. Each request carries an API key in the X-Kestrel-Key header. Keys are created in the portal under Settings, Developers. An account may have two active keys at a time, so a key can be rotated without downtime. Rotating keys every 90 days is recommended.

## Rate limits
Each API key may send 120 requests per minute. Above that the API answers with HTTP 429 and a Retry-After header that gives the number of seconds to wait. Bulk label creation should use the batch endpoint, which accepts up to 100 shipments in one request.

## Creating shipments
POST /shipments creates a shipment and returns the tracking number and the label. The label format is chosen with the label_format field: PDF or ZPL. A shipment can be cancelled with DELETE /shipments/{id} until its first scan.

## Idempotency
To make retries safe, send an Idempotency-Key header with a unique value on every POST. The API stores the key for 24 hours and returns the original response when it sees the same key again.

## Sandbox
A sandbox is available at https://sandbox.kestrel-parcel.example/v2. It uses separate keys, creates no real shipments and never bills. Sandbox data is wiped every Sunday night.
