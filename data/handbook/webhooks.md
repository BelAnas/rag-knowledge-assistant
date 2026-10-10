# Webhooks

## Events
Kestrel can notify your system when a shipment changes state. The events are shipment.created, shipment.in_transit, shipment.delivered and shipment.exception. Each event is a JSON document sent by HTTP POST to the URL registered in the portal.

## Verifying the signature
Every webhook request is signed. The X-Kestrel-Signature header contains an HMAC-SHA256 of the raw request body, computed with the webhook secret. Reject any request whose signature does not match.

## Retries
Your endpoint must answer with a 2xx status within 5 seconds. If it does not, the event is retried 6 times with exponential backoff, spread over 24 hours.

## Automatic disabling
An endpoint that fails every delivery for 3 consecutive days is disabled, and the account owner is told by email. It has to be re-enabled by hand in the portal.

## Replaying events
Events from the last 14 days can be replayed from the portal, one by one or for a date range. Replayed events carry the same event id, so the receiving system should ignore ids it has already processed.
