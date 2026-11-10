"""Local short-term-rental regulation as data and code.

    amsterdam_rules.json                the rules, each with its official article, source and dates
    amsterdam_15_night_wijken.geojson   the eight wijken with a 15-night cap from 1 April 2026
    compliance.py                       deterministic checks: which rules apply to an opportunity,
                                        its night cap per listing, and the revenue within the cap

The rules are curated from the official texts (Huisvestingsverordening Amsterdam 2024 and the
15-night designation) and carry a `verified_on` date. The agents read them through two tools
(regulation_rules, compliance_check) and cite them like any other evidence. Rules change: the
web search of the Data Enrichment agent is the way to check for anything newer.
"""
from agents.regulation.compliance import (load_rules, night_caps, regime_for, rules_for_city,
                                          segment_compliance)

__all__ = ["load_rules", "night_caps", "regime_for", "rules_for_city", "segment_compliance"]
