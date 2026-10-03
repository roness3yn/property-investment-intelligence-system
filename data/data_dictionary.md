# Data dictionary

**Status: planned fields, not an inspected schema.** Data owner: Erin.

Complete this from actual source files. For each column record original name, type, meaning, units, missing values, cleaning and source/snapshot.

| Planned table | Grain / key | Candidate fields | Validation |
| --- | --- | --- | --- |
| Listings | city + snapshot date + listing ID | id, latitude, longitude, neighbourhood_cleansed, room_type, accommodates, bedrooms, bathrooms_text, price | Confirm schema; unique composite key; currency parsing and positive price |
| Calendar | city + snapshot date + listing ID + calendar date | listing_id, date, available, price | Confirm coverage; parse dates and flags; unavailable does not mean booked |
| Reviews | city + snapshot date + review ID | id, listing_id, date | Aggregate activity where justified; exclude personal text/identifiers from published outputs |
| Features | listing identity and feature snapshot | property/location features, comparable statistics | Training-only aggregations; no target-derived leakage |
| Predictions | listing identity and model version | predicted nightly price, residual, quality flags | Same currency/night units as target; record model/split |

Document every implemented derived feature with formula, availability at inference, units and missing-value policy. Avoid mixing currencies and snapshots silently. No occupancy, realized revenue, acquisition value or ROI column is established.

