"""Book-aware execution simulator for the recorder's Kraken L2 + trade files.

replay.py    rebuilds the order book from recorded files and yields book / trade / gap events
engine.py    resting-order queue and fill model, market crossing, episodes, cost accounting
policies.py  baseline execution policies (market, post-then-cross, post-and-chase)
report.py    paired comparison of policies under one or more fee tiers
run.py       command line entry point
fetch.py     downloads recorder files from an S3-compatible bucket (DigitalOcean Spaces)

Nothing here places orders or talks to an exchange.
"""
