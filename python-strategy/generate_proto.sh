#!/usr/bin/env bash
# Regenerates Python gRPC stubs from proto/trading.proto into
# python-strategy/strategy/pb/. Run this any time trading.proto changes.
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p strategy/pb
touch strategy/pb/__init__.py

python -m grpc_tools.protoc \
  -I ../proto \
  --python_out=strategy/pb \
  --grpc_python_out=strategy/pb \
  --pyi_out=strategy/pb \
  ../proto/trading.proto

# grpc_tools generates absolute imports (`import trading_pb2`) that only
# work if pb/ is on sys.path directly. Rewrite to package-relative imports
# so `strategy.pb.trading_pb2_grpc` works from anywhere.
sed -i 's/^import trading_pb2/from . import trading_pb2/' strategy/pb/trading_pb2_grpc.py

echo "Generated strategy/pb/trading_pb2.py, trading_pb2_grpc.py, trading_pb2.pyi"
