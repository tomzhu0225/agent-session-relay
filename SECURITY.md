# Security policy

Agent Session Relay reads local coding-agent histories and may pass recovered
context to a different provider selected by the user. Treat session histories,
recovery bundles, Git diffs, and skill paths as sensitive local data.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's security-advisory
feature for this repository. Do not open a public issue containing session
content, credentials, private paths, or a working exploit.

## Important boundaries

- Relay does not upload data on its own. Cross-provider disclosure begins only
  when the user launches a different target agent.
- Secret redaction is defense in depth, not a guarantee that arbitrary history
  text contains no sensitive information. Inspect a `--dry-run` recovery bundle
  when the source session is sensitive.
- Git diffs are preserved verbatim so they remain useful. Use `--no-git-diff`
  when the working tree may contain credentials or other sensitive changes.
- Relay does not interpret a skill as permission. The receiving agent remains
  subject to its normal filesystem, network, and approval controls.
- Vendor history formats are not stable APIs. An adapter can omit or misclassify
  content after a vendor update; use `--dry-run` and report incompatibilities.

Only the latest release is supported with security fixes.
