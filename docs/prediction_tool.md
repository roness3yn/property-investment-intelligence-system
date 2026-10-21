# Prediction tool: hand-off to the agent

**Owner:** models & evaluation (Idan) · **Consumer:** agent (Adir) · **Code:** `src/models/predict.py` · **Design:** §19.1

**Before first use:** run `04_modeling` for each city (it saves the models the tool loads), then `05_model_evaluation` (it selects the model per city). Both need `data/processed/listings_features.parquet` from `03_Feature_Engineering`.

The agent gets price predictions through **one read-only function**. It never loads model files or data itself.

## One model per city

Each city has its own model, trained on that city's listings, with prices in that city's currency (README: one city, one currency per model). `PredictionService` loads every city that has a trained model and routes each call to the right one. The agent uses one tool for all cities.

## Load once, call many times

```python
from src.models.predict import PredictionService, TOOL_SCHEMA

service = PredictionService.load(".")    # run from the repository root; all trained cities
print(service.cities)                    # e.g. ['Amsterdam', 'Barcelona']
result = service.handle_tool_call({"listing_id": 28871})
```

`load()` does four things for each city:
- **Picks the model.** It uses the model that `05_model_evaluation` selected for that city (`reports/<city>/model_card_metrics.json`). Until 05 has run for a city, it uses the development pick of `04_modeling` and warns that predictions are provisional.
- **Checks the artifact.** It refuses to load a model file whose hash differs from the one recorded by `04_modeling`.
- **Loads the interval model** that matches the city, split and features.
- **Checks the data.** If that city's listings changed since training, it warns. Changes to *other* cities don't trigger the warning.

The listings file is read once and shared by all cities. For a single city, `PredictionTool.load(".", city="Amsterdam")` returns that city's tool alone.

Command line, for a quick check:
```
python -m src.models.predict --listing-id 28871
python -m src.models.predict --city Amsterdam --features "{\"neighbourhood_cleansed\": \"Zuid\", \"room_type\": \"Entire home/apt\", \"accommodates\": 4}"
```

## Input

Exactly one of `listing_id` or `features`, plus `city` when needed:

| Field | Meaning |
|---|---|
| `listing_id` (+ optional `snapshot_date`) | An existing listing. Uses the latest snapshot if no date is given. The city is found from the id; pass `city` only if the id exists in two cities (the tool says so). |
| `features` + `city` | A hypothetical listing. `city` is required when several cities are loaded. Required features: `neighbourhood_cleansed`, `room_type`, `accommodates`. Optional: `bedrooms`, `beds`, `bathrooms_text`, `property_type`, `latitude`/`longitude`, `amenities` (a list such as `["Wifi", "Kitchen"]`), … |

For a hypothetical listing the tool fills in what a caller cannot know, and lists each step in `assumptions`:
- **missing coordinates:** the neighbourhood's typical location;
- **competitor counts** (`competitors_500m/1km/2km`, `comparable_competitors_1km`): computed from the coordinates over the city's current listings, exactly as `03_Feature_Engineering` defines them (`src/models/engineered.py`);
- **neighbourhood counts:** taken from the neighbourhood;
- **amenity flags and count:** computed from `amenities` if given; otherwise imputed by the model.

## Output (always the same keys)

| Field | Meaning |
|---|---|
| `city` | Which city's model answered. |
| `status` | One of the four statuses in the next section. |
| `status_reason` | Why, when `status` is not `ok`. |
| `currency` | The city's local currency (EUR, GBP, CHF, CZK, DKK, NOK). **Never compare or average prices across currencies.** |
| `predicted_median_price`, `interval_80` | The typical **listing** price per night, and its 80% range. Both are `null` unless `status` is `ok`. |
| `observed_price`, `position` | The listing's own asking price, and whether it is `below_interval`, `within_interval` or `above_interval`. Both are `null` if the price is unknown. |
| `in_training_data` | `true` means the model saw this listing in training. Its error there is optimistic, and the 80% guarantee does not apply. |
| `top_drivers` | Up to 3 features that raise the prediction and 3 that lower it, each with a ready-made `statement`, e.g. `"+22% for accommodates = 6 (typical: 5)"`. |
| `assumptions` | Values the tool filled in for a hypothetical listing. |
| `model_version`, `split_version`, `data_snapshot_date` | Exactly which model and data produced the answer. |
| `evidence_id` | **Cite this for every claim**, e.g. `pred:hist_gbm_amsterdam_20261018_1406:28871:2026-06-15`. |

## Statuses

| `status` | When | What the agent should do |
|---|---|---|
| `ok` | The listing is within the model's experience. | Use the numbers; cite `evidence_id`. |
| `insufficient_evidence` | Any of the cases listed below. | **State no price.** Report `status_reason` instead. |
| `not_found` | Unknown listing id (in any modelled city), or not in that snapshot. | Say so. |
| `invalid_input` | The call itself was malformed, a city has no model, or a hypothetical listing lacks `city`. | Fix the call; `status_reason` lists the available cities. |

`insufficient_evidence` is returned in any of these cases:
- the neighbourhood has fewer than 20 training listings;
- the room type or property type was never seen in training;
- the coordinates are outside the training area;
- it's a long-term rental (minimum nights above the target policy's limit, 30), a market the model wasn't trained on;
- a core feature is missing;
- the interval is wider than 2× the prediction.

The limits are in `SupportLimits`. Agree them as a team and change them in one place.

The tool never raises an exception at the agent. Errors come back as data.

## Guardrails (put these in the agent's prompt as well; the tool already enforces the first)

1. No price is ever shown unless `status` is `ok`.
2. The prediction is a nightly **listing** price in the city's currency. It is not a property value, a purchase price, revenue or occupancy, and prices in different currencies are not comparable.
3. `below_interval` means the asking price is low compared with similar listings. It does **not** mean the property is a good investment.
4. Drivers describe the **model**, not causes in the market. They are computed by occlusion: the change in the prediction when a feature is set to its typical training value.

## Using it as an LLM tool (Anthropic SDK example)

```python
import json, anthropic
client = anthropic.Anthropic()
service = PredictionService.load(".")

messages = [{"role": "user", "content": "Is listing 28871 priced low for its area?"}]
resp = client.messages.create(model="<model id>", max_tokens=1024,
                              tools=[TOOL_SCHEMA], messages=messages)
for block in resp.content:
    if block.type == "tool_use" and block.name == TOOL_SCHEMA["name"]:
        result = service.handle_tool_call(block.input)
        messages += [{"role": "assistant", "content": resp.content},
                     {"role": "user", "content": [{"type": "tool_result",
                                                   "tool_use_id": block.id,
                                                   "content": json.dumps(result)}]}]
# ...call client.messages.create again with the updated messages
```

## For the groundedness harness (design §17.3)

```python
service = PredictionService.load(".", evidence_log="evaluation/agent_evidence.jsonl")
```

Every response is appended to that file. A claim in the agent's report is **grounded** if the `evidence_id` it cites exists in the log and the number it quotes matches the logged value.

## Performance and tests

- **Latency:** about 45 ms per call (median) on the stand-in data, of which about 25 ms is the model and interval. The README target is 200 ms.
- **Tests:** `python -m pytest tests/test_predict.py -q`, or `python -m tests.test_predict` without pytest. They build a small synthetic two-city repository (Amsterdam in EUR, London in GBP), so no real data is needed.
