#!/usr/bin/env bash

set -euo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPOSITORY_ROOT"

export PATH="/home/clannad/.nvm/versions/node/v22.19.0/bin:$PATH"

npm --prefix portfolio_viewer run portfolio:update
exec npm --prefix portfolio_viewer run start
