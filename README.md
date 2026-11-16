# ztkik_p — Bezpieczna detekcja spamu (Paillier + Naive Bayes)

Privacy-preserving SMS spam classification: the client encrypts a TF-IDF feature vector with Paillier; the server (honest-but-curious) computes a linear Naive Bayes score homomorphically without seeing the message.

## Wymagania

- Python **3.11+** (testowane na 3.14)
- Linux / WSL2 (pomiary wydajności planowane na WSL2)
- `libgmp` (dla `gmpy2`; na Ubuntu: `sudo apt install libgmp-dev`)

## Szybki start

```bash
cd ztkik_p
make install          # tworzy .venv i instaluje zależności
make test             # pytest (smoke)
make verify-gmpy2     # porównanie szyfrowania z/bez gmpy2
```

Jeśli `python3 -m venv` kończy się błędem `ensurepip` (typowe na WSL bez `python3-venv`):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv .venv --python 3.14
uv pip install -r requirements.txt --python .venv/bin/python
uv pip freeze --python .venv/bin/python > requirements.lock.txt
```

Ręcznie (gdy `python3-venv` jest zainstalowany):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```



## Struktura

Pakiet źródłowy: `src/ppspam/` (`config.py` — `SEED=42`, ścieżki, parametry).  
Dane: `data/raw/` (pobierane skryptem), `data/processed/` (train/test).  
Artefakty modelu: `artifacts/`, wyniki pomiarów: `results/`.

## Demo klient–serwer (B5)
Najpierw przygotuj dane, model i kwantyzację (`make data && make train && make quantize` — artefakty trafią do `artifacts/`). Potem lokalnie w jednym procesie:

```bash
make demo
# albo jedna wiadomość:
make demo ARGS='--text "WINNER!! claim your prize now"'
```

Albo dwa terminale (API + klient sieciowy):

```bash
make serve                                          # http://127.0.0.1:8000
make demo ARGS='--url http://127.0.0.1:8000'
```

Endpointy: `GET /public-model`, `POST /score`, `GET /health`. Domyślny klucz w demo to 1024 bit (szybko); wyniki do raportu liczy `make eval-encrypted` na 2048.

## Ewaluacja zaszyfrowana (A7)

```bash
make eval-encrypted                 # pełny test, klucz 2048, wiele procesów
make eval-encrypted ARGS='--limit 10 --key-size 1024'   # szybki smoke
```

Wyniki: `results/encrypted_eval.csv`, `results/encrypted_metrics.json` (K2–K4).

## Optymalizacje i pomiary (B6, W2)

```bash
make bench                              # E2–E7, 10 powtórzeń, ~20 min
make bench ARGS='--only E6,E7'          # wybrane eksperymenty
make bench ARGS='--quick --repeats 2'   # smoke, nie do raportu
make serve ARGS='--workers 6'           # API z równoległym iloczynem na serwerze
```

Wyniki: `results/bench.md` (tabele + wykresy), `results/bench_e*.csv`, `results/bench_env.json`. Przy `d=1000` i kluczu 2048 jeden SMS end-to-end to ok. 8,2 s w wersji bazowej, ok. 1,85 s z równoległym szyfrowaniem i serwerem oraz ok. 88 ms online z prekomputowaną pulą `r^n`. Wersja jawna to ok. 0,36 ms.

