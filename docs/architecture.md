# Architecture decisions

## Primary scope

Follow the uploaded charter and Airbnb-only architecture: short-term-rental research using Inside Airbnb. The earlier working document's Zillow-led U.S. market analysis is Plan B and requires a revised modelling target.

## Responsibilities and contracts

Data pipeline produces validated snapshots and provenance. Features produce a versioned feature matrix. Models return nightly-price predictions. Analytics returns comparable/neighborhood evidence and ranking with quality flags. The application renders those outputs. The agent consumes the same services through read-only tools.

Every tool result should carry stable evidence IDs, city, listing identity, snapshot date, units and quality flags. Opportunity scores need a documented formula and sensitivity checks; price residual alone does not prove an investment opportunity.

The agent is optional at runtime. On failure, display analytical outputs and an explicit investigation status. Define tool bounds, output schema and evidence validation before integrating an LLM.

## Open decisions

- Select initial city, snapshot and currency after inspecting accessible files.
- Confirm target, success thresholds and grouping/temporal split.
- Choose tested dependencies, dashboard framework and LLM provider.
- Define comparable relevance and evidence/confidence policies.
- Verify source usage conditions and data minimization.

No infrastructure, cloud provider or paid service is selected by this scaffold.

