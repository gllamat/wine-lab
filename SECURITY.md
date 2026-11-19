# Security and sensitive content

- **Secrets.** Never commit API keys or tokens. Keys belong in a local `.env` file (ignored by git). If you find a secret in the history, report it privately (GitHub: Security → Report a vulnerability) so it can be revoked and the history cleaned.
- **Vulnerabilities.** Report them privately through GitHub's security advisories, not in public issues.
- **Rights concerns.** For copyright or licence concerns (for example review text, WSET material or data licences) use the *Takedown / rights request* issue template, or see [DISCLAIMER.md](DISCLAIMER.md) section 5.
- The dashboard is a static site with no server, no cookies and no user accounts. It stores only your audience choice and theme in the browser's local storage.
