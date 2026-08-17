#!/usr/bin/env bash
# =============================================================================
# Sample request/response test calls  --  Deliverable 3
# =============================================================================
# Usage:
#   1. Start the service:  uvicorn src.serving.app:app --port 8000
#   2. Run this script:    bash api_tests/curl_collection.sh
#
# On Windows, run from Git Bash (curl is bundled with Git for Windows).
#
# Each call below is annotated with the expected status code and what the call
# demonstrates. The set is ordered to tell a story: healthy service, correct
# predictions, then progressively more hostile input.
# =============================================================================

BASE="${BASE:-http://127.0.0.1:8000}"
PASS=0; FAIL=0

call() {
  local desc="$1" expected="$2" method="$3" path="$4" body="$5"
  echo ""
  echo "-----------------------------------------------------------------------"
  echo "  $desc"
  echo "  expect HTTP $expected"
  echo "-----------------------------------------------------------------------"

  if [ -z "$body" ]; then
    resp=$(curl -s -w '\n%{http_code}' -X "$method" "$BASE$path" \
      -H 'accept: application/json')
  else
    resp=$(curl -s -w '\n%{http_code}' -X "$method" "$BASE$path" \
      -H 'accept: application/json' -H 'Content-Type: application/json' \
      -d "$body")
    echo "  request: $body"
  fi

  code=$(echo "$resp" | tail -1)
  echo "  response ($code):"
  echo "$resp" | sed '$d' | sed 's/^/    /'

  if [ "$code" = "$expected" ]; then
    echo "  [PASS]"; PASS=$((PASS+1))
  else
    echo "  [FAIL] got $code, expected $expected"; FAIL=$((FAIL+1))
  fi
}

echo "======================================================================="
echo "  API TEST COLLECTION  --  $BASE"
echo "======================================================================="

# --- 1. Service health -------------------------------------------------------
call "Health check: both artefacts loaded?" \
     200 GET /health

# --- 2. The happy path -------------------------------------------------------
call "Strongly negative input" \
     200 POST /predict/sentiment \
     '{"text":"terrible awful worst movie ever"}'

call "Strongly positive input" \
     200 POST /predict/sentiment \
     '{"text":"i loved it absolutely brilliant"}'

call "Genuinely neutral input" \
     200 POST /predict/sentiment \
     '{"text":"the film was ok i guess"}'

# --- 3. A known weakness, shown rather than hidden ---------------------------
call "Idiomatic negativity -- model returns NEUTRAL (documented in DEPLOYMENT.md sec 5)" \
     200 POST /predict/sentiment \
     '{"text":"this film was a complete waste of time"}'

# --- 4. Batch ----------------------------------------------------------------
call "Batch of three: per-item latency should be well below single-call" \
     200 POST /predict/batch \
     '{"texts":["great film","awful film","a film"]}'

call "Batch with empty entries -- skipped and COUNTED, never silently dropped" \
     200 POST /predict/batch \
     '{"texts":["good","","   ","bad"]}'

# --- 5. Edge cases that must SUCCEED: unusual is not invalid ------------------
call "Emoji only" \
     200 POST /predict/sentiment \
     '{"text":"\ud83d\ude00 \ud83d\ude02"}'

call "Entirely out-of-vocabulary -- must return LOW confidence, not a 500" \
     200 POST /predict/sentiment \
     '{"text":"qwertyuiop asdfghjkl zxcvbnm"}'

call "Non-Latin script" \
     200 POST /predict/sentiment \
     '{"text":"\u092f\u0939 \u0905\u0915\u094d\u0937\u0930"}'

call "Single character" \
     200 POST /predict/sentiment \
     '{"text":"a"}'

# --- 6. Malformed input: 422, never 500 --------------------------------------
call "Empty string" \
     422 POST /predict/sentiment \
     '{"text":""}'

call "Whitespace only -- min_length would accept it, the custom validator does not" \
     422 POST /predict/sentiment \
     '{"text":"     "}'

call "Missing required field" \
     422 POST /predict/sentiment \
     '{}'

call "Wrong field name" \
     422 POST /predict/sentiment \
     '{"txt":"hello"}'

call "Wrong type: integer instead of string" \
     422 POST /predict/sentiment \
     '{"text":12345}'

call "Null value" \
     422 POST /predict/sentiment \
     '{"text":null}'

call "Empty batch" \
     422 POST /predict/batch \
     '{"texts":[]}'

# --- 7. Observability --------------------------------------------------------
call "Operational metrics from the prediction log" \
     200 GET /metrics

echo ""
echo "======================================================================="
echo "  RESULT: $PASS passed, $FAIL failed"
echo "======================================================================="
[ "$FAIL" -eq 0 ] || exit 1
