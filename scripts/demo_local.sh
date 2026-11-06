#!/usr/bin/env bash
# Local demo without Docker: needs Redis on localhost:6379 OR use MemoryRedis via pytest.
# With Redis: start api + worker + scaler in three terminals, then run this script.
set -euo pipefail
API="${API:-http://127.0.0.1:8080}"

echo "== health =="
curl -s "$API/healthz" | tee /dev/stderr
echo

echo "== enqueue burst =="
IDS=()
for i in $(seq 1 12); do
  ID=$(curl -s -X POST "$API/v1/jobs" \
    -H 'content-type: application/json' \
    -d "{\"prompt\":\"distributed systems lab job $i\",\"max_tokens\":16}" \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["job_id"])')
  IDS+=("$ID")
  echo "enqueued $ID"
done

echo "== scale snapshot =="
curl -s "$API/v1/scale"; echo

echo "== poll first job =="
JOB="${IDS[0]}"
for _ in $(seq 1 40); do
  STATUS=$(curl -s "$API/v1/jobs/$JOB" | python3 -c 'import sys,json; print(json.load(sys.stdin)["status"])')
  echo "status=$STATUS"
  if [[ "$STATUS" == "succeeded" || "$STATUS" == "failed" ]]; then
    curl -s "$API/v1/jobs/$JOB"; echo
    break
  fi
  sleep 0.25
done
