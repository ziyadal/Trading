 News Agent = OpenAI's `gpt-4.1-mini` reads that research and produces a structured assessment via `.with_structured_output(NewsOutput)`. --> pretty sure this can be enfored using eprelxity API

 Bear opening argument (sees bull's opening) this should be changed.

 ---

 ## Harness cutoff uses string replacement on SQL — fragile

 In `ta_agent.py:make_query_tool`, backtest mode "hides" future data from the
 TA agent by running `sql.replace("btc_ohlcv", "(SELECT * FROM btc_ohlcv WHERE timestamp <= '<cutoff>')")`
 on every query before executing it.

 **Why this is brittle:** `str.replace` matches the substring anywhere, so the
 moment anything else in the database contains `btc_ohlcv` (another table, a
 column, a comment, a string literal), the rewrite corrupts the query silently.

 **Example:** suppose we later add an hourly table `btc_ohlcv_1h`. The agent
 writes:

 ```sql
 SELECT timestamp, close FROM btc_ohlcv_1h ORDER BY timestamp DESC LIMIT 10
 ```

 After replacement this becomes:

 ```sql
 SELECT timestamp, close FROM (SELECT * FROM btc_ohlcv WHERE timestamp <= '2026-02-01 12:00:00')_1h ORDER BY timestamp DESC LIMIT 10
 ```

 — a syntax error at best, and at worst it silently reads from the wrong table.

 **Safer alternatives to think about:** a SQLite VIEW (`btc_ohlcv_view`) that
 the agent is told to query, swapped at runtime; or having the tool reject any
 row with `timestamp > cutoff` after execution rather than rewriting the SQL.
 Both remove the "table name must be unique substring" assumption.

 No code change yet — just flagging before we add more tables.
