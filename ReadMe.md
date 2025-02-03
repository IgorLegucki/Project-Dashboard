Dokumentacja Projektu Analiza ogłoszeń samochodowych na portalu Otomoto

1. Cel projektu
Projekt ma na celu pobrać dane ze strony Otomoto poprzez scrapping. Ma przeanalizować szereg zmiennych
i wyciągnąć w ten sposób informacje dotyczące samochodów. Dodatkowo program wykonuje predykcje na cenie
samochodów względem danych tecznicznych.

2. Opis Struktury Projektu
Projekt składa się z 6 plików. W tych plikach umieszczona jest logika programu, gdzie każdy plik służy do innej czynności.
Dwa pliki programowe w których jest logika scrappingu, obliczania wartości, tworzenia wykresów oraz ML.
Dwa pliki w których jest umieszczona lista pojęć potrzebna do filtrowania danych.
Dwa pliki również służą do dokumentacji programu.
- main.py - Program scrapujący dane z serwisu Otomoto.
- Streamlit - tworzy dashboard, który pokazuje obliczenia, wykresy i machine learning na danych.
- brands.yaml - zawiera przygotowaną listę przez autora z około 80 nazw marek samochodów.
- voivodes - lista województw w Polsce.
- ReadMe.md - plik zawiera dokumentację projektową
- Analiza rynku samochodów i predykcja cen.docx - Plik zawiera interpretację analizy danych i predykcji.

3. Strona techniczna projektu
Program został napisany w języku Python. Wykorzystuje dodatkowo składnię yaml. Kod został napisany w
środowisku programistycznym używany do pythona, czyli PyChram firmy JetBrains w wersji Professional. Dodatkowo dokumentacja 
techiczna została napisana w zwykłym pliku tekstowym.

4. Biblioteki użyte w pythonie
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

5. Budowa Dashboardu
Dashboard składa się z 5 zakładek:
Informacje - podstawowe informacje na temat dashboardu.
Obliczenia - wykaz podstawowych obliczeń na danych, takie jak, obliczenie średniej, maksymalnej i minimalnej ceny, 
przedstawienie popularnych marek oraz wykaz popularności rozwiązań technicznych w samochodach
Wykresy - przedstawiają graficznie dane w różnych zależnościach między sobą. Pokazują w różny sposób rozkład cen, danych technicznych
pod każdym kątem. 
Tabele - przedstawienie danych w formie tabel. Obliczenie regresji liniowej dla ceny i przebiegu. Obliczenie chi-kwadratu, itp
Machine Learning - przedstawienie dwóch modeli: regresji liniowej i random forest. Model ma wykonać predykcję ceny samochodu
na portalu ogłoszeniowym Otomoto wykorzystując pobrane dane. Wynik jest podawany w procentach oznaczający jak dobrze model przewiduje cenę.