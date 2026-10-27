# Security Controls

## Upload validation

Every uploaded document is checked against an allow list of file types, and
file size limits are enforced before the body is read into memory. Files
that fail validation are rejected with an explicit error.

## Prompt injection

Retrieved documents are untrusted input. An attacker can embed instructions
in a document hoping the model will follow them, which is known as prompt
injection. The system prompt tells the model to treat document content as
data, and answers must cite their sources so reviewers can trace claims back
to the original text.

## Secrets and credentials

API keys and database credentials are provided through environment variables
and never stored in source control. Logs exclude document text and secrets
by default; only metadata such as document identifiers and latency is
recorded.

## Multi-tenant isolation

Retrieval filters by document ownership before any chunk leaves the store,
so one user's documents can never appear in another user's results.
Retention policies delete expired documents and their vectors together.
