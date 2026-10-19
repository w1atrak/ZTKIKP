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

Najpierw `make train` (artefakty w `artifacts/`). Potem lokalnie w jednym procesie:

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

