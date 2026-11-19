# Common tasks. Docker targets mount data from ../datasets and keep outputs/cache/experiments on the host.
IMG ?= winelab
build:        ; docker build --target core -t $(IMG):core . && docker build --target causal -t $(IMG):causal .
build-embed:  ; docker build --target embed -t $(IMG):embed .
smoke:        ; docker run --rm -v "$(PWD)/outputs:/app/outputs:ro" $(IMG):core
matcher:      ; docker compose run --rm lab python src/run_matcher.py all --nocolour
guess-dev:    ; docker compose run --rm lab python src/guess_model.py dev
blind-round:  ; docker compose run --rm lab python src/blind.py run --n 500
blind-status: ; docker compose run --rm lab python src/blind.py status
causal:       ; docker compose run --rm lab python src/causal.py export && docker compose run --rm lab python src/causal_estimate.py
serve:        ; cd dashboard && python3 -m http.server 8765
parity:       ; node dashboard/parity.js && node dashboard/guess_parity.js && node dashboard/causal_parity.js
