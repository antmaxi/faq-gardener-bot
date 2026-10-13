#!/bin/sh
# Docker creates missing bind-mount dirs as root. The job runs as uid 1000
# and must be able to write its own SQLite and FAQ cache under /data.
# The answer-bot database is a separate read-only mount and is left alone.
set -e
mkdir -p /data
if [ "$(id -u)" = "0" ]; then
  chown -R gardener:gardener /data
  exec gosu gardener "$@"
fi
exec "$@"
