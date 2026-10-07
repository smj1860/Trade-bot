# Paper trader on the droplet

Runs the same `python-strategy/scripts/paper_trade.py` as the GitHub workflow,
hourly at :07 from cron, so no hours are skipped. State stays in Supabase
(`paper_models`, `paper_trades`, `paper_runs`); inserts are idempotent
(`on conflict do nothing`), so running this and the GitHub workflow together
cannot create duplicate trades.

Setup (on the droplet, as root; the Supabase URL is a secret and lives only in /root/paper.env):

    cd /opt/trade-bot && git pull
    docker build -f deploy/paper-trader/Dockerfile -t kraken-paper .
    read -s -p "SUPABASE_DB_URL: " U; echo "SUPABASE_DB_URL=$U" > /root/paper.env; chmod 600 /root/paper.env; unset U
    deploy/paper-trader/run-paper.sh dry        # nothing is written
    deploy/paper-trader/run-paper.sh trade      # one real run

Schedule (/etc/cron.d/paper-trade): hourly trade at :07, daily report at 12:20 UTC.
The recorder's hourly xz compression finishes a few minutes after :00, so the
two jobs do not overlap. Each run prints its peak memory.

## Weekly challengers

`.github/workflows/weekly-challenger.yml` (Sundays 03:17 UTC, or run it by hand) trains the
champion's recipe on the latest candles and registers it as a new arm `ch-YYYYMMDD-m4`. This job
picks it up automatically because it runs every registered arm. Only the newest 4 challengers stay
active; older ones are marked `retired` in `paper_models.meta`, stop opening trades and just finish
their open ones. The workflow summary compares each challenger with `ext-fib-m4` on identical
(symbol, entry bar) trades with a bootstrap CI. After the first Sunday run, `git pull` here and
rebuild the image once (`docker build -f deploy/paper-trader/Dockerfile -t kraken-paper .`) so the
retirement handling is in the container.
