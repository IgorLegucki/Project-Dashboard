Dokumentacja Projektu Analiza ogłoszeń samochodowych na portalu Otomoto

Opis Struktury Projektu:

1. Projekt ma na celu pobrać dane ze strony Otomoto poprzez scrapping. Ma przeanalizować szereg zmiennych
i wyciągnąć w ten sposób informacje dotyczące samochodów. Dodatkowo program wykonuje predykcje na cenie
samochodów względem danych tecznicznych.

2. Projekt składa się z 9 plików. W tych plikach umieszczona jest logika programu, gdzie każdy plik służy do innej czynności.
Są dwa pliki w których jest umieszczona lista pojęć potrzebna do filtrowania danych.
Dwa pliki również służą do dokumentacji programu.
- main.py - główny program scrapujący dane z serwisu Otomoto.
- EDA.py - jest to plik, zawierający proste obliczenia i wstępna analiza danych.
- Analiza danych.py - zawiera rozkład najpoplarniejszych marek i innych danych.
 Dodatkowo plik tworzy wykresy testujące różne zależności pomięzy danymi samochodu i jego sprzedaży.
- Streamlit - tworzy dashboard, który pokazuje parę istotnych wykresów z analizy danych.
- ML - Machine Learning, czyli program, który wykorzystuje modele uczące do przewidywania danych.
- brands.yaml - zawiera przygotowaną listę przez autora z około 80 nazw marek samochodów.
- voivodes - lista województw w Polsce.
- ReadMe.md - plik zawiera dokumentację projektową
- Analiza rynku samochodów i predykcja cen.docx - Plik zawiera interpretację analizy danych i predykcji.

3. Strona techniczna projektu:
Program został napisany w języku Python. Wykorzystuje dodatkowo składnię yaml. Kod został napisany w
środowisku programistycznym używany do pythona, czyli PyChram firmy JetBrains w wersji Professional. Dodatkowo dokumentacja 
techiczna została napisana w zwykłym pliku tekstowym.

4. Biblioteki użyte w pythonie:
- pandas
- numpy
- plotly
- matplotlib
- seaborn
- statsmodels
- streamlit
- yaml
- chi2_contingency
- sklearn