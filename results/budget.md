# E1 — budżet obliczeniowy (klucz 2048, gmpy2)

Koszt jednego szyfrogramu: mediana `encrypt` = 0.007537 s (B1, obfuskacja włączona). Rozmiar szyfrogramu: 512 B. Zbiór testowy: 1035 SMS-ów.

Czas szyfrowania jednego SMS-a to `d × t_enc` (każda współrzędna, także zera). Po stronie serwera dominuje `enc * w_q`; wagi po A6 mają ≤ 10 bitów, więc bierzemy medianę mnożenia przez 16-bitową skalę (0.000084 s).

| d | 1 SMS (encrypt) | cały test (encrypt) | 1 SMS (serwer) | rozmiar / SMS |
| ---: | ---: | ---: | ---: | ---: |
| 500 | 3.8 s | 65.0 min | 42 ms | 0.26 MB |
| 1000 | 7.5 s | 2.2 h | 84 ms | 0.51 MB |
| 2000 | 15.1 s | 4.3 h | 167 ms | 1.02 MB |
| 5000 | 37.7 s | 10.8 h | 418 ms | 2.56 MB |
| 9000 | 67.8 s | 19.5 h | 752 ms | 4.61 MB |

## Decyzja D5

`d = 1000` (już wybrane w A5, w zakresie D5 {500, 1000, 2000}). Zaszyfrowanie całego testu trwa około 2.2 h, a jeden SMS to 0.51 MB. Przy `d = 5000` i `9000` sam klient spędza na szyfrowaniu wiele godzin, a A5 pokazało, że zejście do 1000 cech nie psuje accuracy modelu jawnego. Zostawiamy `FEATURE_DIM = 1000`.
