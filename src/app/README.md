# src/app

`streamlit_app.py`: the investor workspace, "Investment Intelligence", in Hebrew (right to left).

```bash
streamlit run src/app/streamlit_app.py      # or ./run_app.sh / run_app.bat
```

**סוכן המחקר** (main tab): the conversation fills the screen height, with the input pinned under it.
- A welcome screen with suggested questions. While the agents work, a live status shows the pipeline
  stage (input, router, agent, tools, verification) and each tool call with its evidence reference.
- Under every answer: the path it took, how many numbers were checked, and the evidence behind each [E#].
- A context panel beside the chat: the ranked opportunities (each with an "investigate" button that
  hands it to the agents), a map of Amsterdam (Plotly dot-density; click a neighbourhood to filter),
  and the price model's test results and known weak segments.

**טבלת הזדמנויות**: the full deterministic table with filters; works without the agents.
**יומן ראיות**: every tool result of the session and the running groundedness (criterion 7).

Settings (API key, models) and data status are in the sidebar. The theme is in `.streamlit/config.toml`.
`requirements.txt` here is the dependency file Streamlit Community Cloud reads for this app.
The API key comes from the sidebar, a `.env` file, or the app's Secrets (`ANTHROPIC_API_KEY`).
