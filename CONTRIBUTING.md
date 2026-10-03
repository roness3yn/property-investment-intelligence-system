# Contributing

1. Pick an issue with an owner and acceptance criteria.
2. Create a short branch: `feature/<issue>-<summary>`, `fix/<issue>-<summary>` or `docs/<summary>`.
3. Keep reusable logic in `src/`; notebooks explain analysis and call that logic.
4. Record data sources, split rules, feature changes and experiment settings.
5. Clear notebook outputs that expose personal data, credentials or large tables.
6. Open a pull request with the purpose, changed behavior, verification and limitations.
7. Request team review before merging.

Never commit secrets, downloaded datasets or binary model artifacts. Pin dependencies once tested. Do not report planned metrics as achieved results. Use meaningful tests for transformations, calculations, prediction interfaces and agent tools as they are implemented.

