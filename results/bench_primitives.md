# B1 — operacje elementarne Paillier

Mediana z 100 powtórzeń (keygen: 5). `encrypt` używa domyślnej obfuskacji `r^n mod n²`. `add` i `mul` to surowe operacje homomorficzne (bez ponownej obfuskacji) — w `phe` właśnie obfuskacja, a nie `(1+n)^m`, dominuje koszt szyfrowania.

## Poprawność (klucz 1024)

- Dec(Enc(42)) = 42
- Dec(Enc(-7)) = -7
- Dec(Enc(0)) = 0
- Dec(Enc(1.5)) = 1.5
- Dec(Enc(12)+Enc(-5)) = 7
- Dec(Enc(12)*9) = 108
- Dec(Enc(12)*-9) = -108
- klucz 1024: mediana rozmiaru szyfrogramu 256 B (oczekiwane ~256 B = 2·n_length bitów)
- klucz 2048: mediana rozmiaru szyfrogramu 512 B (oczekiwane ~512 B = 2·n_length bitów)
- klucz 3072: mediana rozmiaru szyfrogramu 768 B (oczekiwane ~768 B = 2·n_length bitów)

## Mediana encrypt / decrypt / add (sekundy)

| n | gmpy2 | keygen | encrypt | decrypt | add |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1024 | True | 0.005478 | 0.001763 | 0.000541 | 0.000006 |
| 1024 | False | 0.153932 | 0.016695 | 0.003262 | 0.000011 |
| 2048 | True | 0.026513 | 0.007537 | 0.002107 | 0.000008 |
| 2048 | False | 0.541929 | 0.086850 | 0.024053 | 0.000036 |
| 3072 | True | 0.079155 | 0.026056 | 0.006575 | 0.000013 |
| 3072 | False | 4.265885 | 0.274199 | 0.078360 | 0.000073 |

## Mediana `enc * k` (sekundy)

| n | gmpy2 | 8 bit | 16 bit | 24 bit | 32 bit |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1024 | True | 0.000026 | 0.000043 | 0.000061 | 0.000072 |
| 1024 | False | 0.000131 | 0.000282 | 0.000431 | 0.000585 |
| 2048 | True | 0.000046 | 0.000084 | 0.000122 | 0.000147 |
| 2048 | False | 0.000480 | 0.001024 | 0.001590 | 0.002129 |
| 3072 | True | 0.000090 | 0.000172 | 0.000257 | 0.000315 |
| 3072 | False | 0.001003 | 0.002150 | 0.003340 | 0.004443 |

Przy 2048 bitach gmpy2 przyspiesza `encrypt` około 11.5×. Mediana z gmpy2 (0.0075 s) jest kosztem jednego szyfrogramu w E1.

Surowe liczby: `bench_primitives.csv`.
