# Security policy

ChemApp processes untrusted instrument archives and can optionally send selected
experimental summaries to third-party language-model providers. Treat every
upload, filename, metadata field, model response, and imported model artifact as
untrusted input.

## Deployment requirements

- Bind the development server to loopback unless an authenticated reverse proxy
  is in front of it.
- Configure separate `CHEMAPP_ACCESS_TOKEN` and `CHEMAPP_ADMIN_TOKEN` values
  before enabling remote access. API calls accept the access token through
  `X-ChemApp-Access-Token` or Bearer authentication; destructive, model-import,
  and training routes additionally require `X-ChemApp-Admin-Token`.
- Configure `CHEMAPP_REVIEWER_TOKENS` as a JSON map of stable reviewer subjects
  to distinct high-entropy secrets. Do not reuse access/admin secrets. Review
  writes have no local identity bypass and ignore reviewer names in payloads;
  the subject is resolved only from `X-ChemApp-Reviewer-Token`.
- Configure `CHEMAPP_REVIEW_ADMIN_SUBJECT` as a stable pseudonymous audit
  subject distinct from every reviewer subject. Review administration fails
  closed when it is missing, malformed, or overlaps a reviewer. Restrict
  `CHEMAPP_REVIEW_ALLOWED_LICENSES` to licenses approved for the deployment.
- Keep `CHEMAPP_LOCAL_ADMIN_BYPASS=0` outside a single-user workstation.
- Keep the runtime database, raw spectra, indexes, checkpoints, `.env`, and API
  keys outside Git and back them up separately.
- Use HTTPS and an explicit `CHEMAPP_LLM_ALLOWED_HOSTS` allowlist. Do not enable
  private or insecure LLM endpoints unless the network and provider are trusted.
- Apply request-body limits at the reverse proxy in addition to application ZIP
  and upload limits.
- Run parsing and model training under an unprivileged account with bounded CPU,
  memory, storage, and GPU access.

## Data sent to AI providers

AI features are optional. Before sending data, users should review the selected
spectra and provider. The payload can include sample names, peak positions,
integrals, multiplets, and analysis metrics. Do not send confidential data to a
provider whose retention and training policy has not been approved.

## Reporting a vulnerability

Do not place secrets, malicious archives, or sensitive spectra in a public
issue. Contact the project owner privately with the affected version, minimal
reproduction steps, impact, and any suggested mitigation.
