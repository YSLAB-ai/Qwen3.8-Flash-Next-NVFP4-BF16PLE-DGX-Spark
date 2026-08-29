#!/usr/bin/env bash
# Quick check that the selected guarded server is up, coherent, and measure throughput.
#   MODEL=qwen3.8-flash-next-orca-uncensored scripts/smoke-test.sh [host:port]
set -euo pipefail
EP="${1:-localhost:18300}"
BASE="http://$EP"
MODEL="${MODEL:-qwen3.8-flash-next-orca-uncensored}"
SAMPLER='"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0,"presence_penalty":0,"repetition_penalty":1,"reasoning_effort":"medium","chat_template_kwargs":{"enable_thinking":true}'

echo ">> health"
curl -sf -m 5 "$BASE/health" >/dev/null && echo "   OK" || { echo "   not ready"; exit 1; }

echo ">> coherence"
curl -s -m 120 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d \
  "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"The capital of France is\"}],\"max_tokens\":12,$SAMPLER}" \
  | python3 -c 'import json,sys;print("  ",repr(json.load(sys.stdin)["choices"][0]["message"]["content"]))'

echo ">> prefill (TTFT on a ~8k-token prompt)"
python3 - "$BASE" "$MODEL" <<'PY'
import json,sys,time,urllib.request
base,model=sys.argv[1:]; prompt="word "*8000
t=time.time()
req=urllib.request.Request(base+"/v1/chat/completions",
    data=json.dumps({"model":model,"messages":[{"role":"user","content":prompt}],"max_tokens":1,
                     "temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0,
                     "presence_penalty":0,"repetition_penalty":1,"reasoning_effort":"medium",
                     "chat_template_kwargs":{"enable_thinking":True}}).encode(),
    headers={"Content-Type":"application/json"})
u=json.load(urllib.request.urlopen(req,timeout=300))["usage"]; dt=time.time()-t
print(f"   {u['prompt_tokens']} tok in {dt:.2f}s  =>  {u['prompt_tokens']/dt:.0f} tok/s prefill")
PY

echo ">> decode (256 tokens, short prompt)"
python3 - "$BASE" "$MODEL" <<'PY'
import json,sys,time,urllib.request
base,model=sys.argv[1:]
t=time.time()
req=urllib.request.Request(base+"/v1/chat/completions",
    data=json.dumps({"model":model,"messages":[{"role":"user","content":"Hello"}],"max_tokens":256,
                     "temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0,
                     "presence_penalty":0,"repetition_penalty":1,"reasoning_effort":"medium",
                     "chat_template_kwargs":{"enable_thinking":True},"ignore_eos":True}).encode(),
    headers={"Content-Type":"application/json"})
n=json.load(urllib.request.urlopen(req,timeout=300))["usage"]["completion_tokens"]; dt=time.time()-t
print(f"   {n} tok in {dt:.2f}s  =>  {n/dt:.1f} tok/s decode")
PY
