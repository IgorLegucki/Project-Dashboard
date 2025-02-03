import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import chi2_contingency
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


tabs = st.tabs(["ℹ️ Informacje", "📐 Obliczenia", "📈 Wykresy", "📊 Tabela", "🤖 Machine learning"])

# Załadowanie danych
@st.cache_data
def load_data():
    df = pd.read_csv('Scraping_result_otomoto.csv')
    return df

df = load_data()

#oczyszczanie danych
def load_voivodes(file_path="voivodes.yaml"):
    with open(file_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data.get("voivodes", [])

polish_voivodes = load_voivodes()

df['price_clean'] = pd.to_numeric(df['price'], errors='coerce').fillna(0).astype(int)
df['engine_clean'] = pd.to_numeric(df['engine'], errors='coerce').fillna(0)
df['KM_clean'] = pd.to_numeric(df['KM'], errors='coerce').fillna(0).astype(int)
df['mileage_clean'] = pd.to_numeric(df['mileage'], errors='coerce').fillna(0).astype(int)
df['year_clean'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
df['voivode'] = df['voivode'].apply(lambda x: x if x in polish_voivodes else 'Inny kraj')

df_cleaned = df.dropna(subset=['price_clean', 'engine_clean', 'KM_clean', 'mileage_clean', 'year_clean', 'brand', 'model', 'gearbox', 'voivode'])

brands = np.array(df['brand'])
model = np.array(df['model'])
prices = np.array(df['price_clean'])
engine = np.array(df['engine_clean'])
KM = np.array(df['KM_clean'])
mileage = np.array(df['mileage_clean'])
gearbox = np.array(df['gearbox'])
year = np.array(df['year_clean'])
voivode = np.array(df['voivode'])

with tabs[0]:
    st.title('Analiza danych rynku samochodowego')
    st.write("""
        Instrukcja dashboardu z analizy danych o ogłoszeniach samochodowych z serwisu Otomoto.
        \n
        \nOpis zakładek:
        \nℹ️ Informacje - Zakładka w której jesteś. Podstawowe informacje na temat tego dashboardu
        \n📐 Obliczenia - Podstawowe obliczenia na danych
        \n📈 Wykresy - Wykresy na danych do analizy
        \n📊 Tabela - Tabele na danych do analizy
        \n🤖 Machine learning - Predyckja ceny w czasie na modelu ML
        \n
        \nCechy danych:
        \n- Dane pochodzą ze strony polskiego serwisu ogłoszeniowego samochodowego Otomoto.
        \n- Dane są pobrane tylko z samochodów osobowych.
        \n- Dane jakie zostały wzięte pod uwagę to: 
        \n[Cena], [Marka], [Model], [Pojemność silnika], [Moc silnika], [Zastosowane zasilanie], [Przebieg], 
        \n[Skrzynia biegów], [Rok produkcji], [Województwo].
        \n
        \nOprócz analizy danych, w dashboardzie został umieszczony model uczenia maszynowego, który przetestował jak dobrze
        przewidzi cenę aut po uwzględnieniu wszystkich czynników.  
    """)

with tabs[1]:
    st.title('Podstawowe obliczenia na danych')
    st.write("""
    Tutaj znajdują się podstawowe obliczenia na cenie, województwach i danych technicznych pojazdów.
    Obliczenia sprowadzają się do średniej, maksimum i minimum. 
    Oprócz tego w tej sekcji będzie pokazane aktualna popularność różnych rozwiazań technicznych czy marek samochodów.
    """)

    # wykonanie obliczeń na cenach
    mean_price = np.nanmean(prices)
    max_price = np.nanmax(prices)
    min_price = np.nanmin(prices)

    max_price_index = np.argmax(prices)
    min_price_index = np.argmin(prices)

    max_price_car = f"{brands[max_price_index]} {model[max_price_index]}"
    min_price_car = f"{brands[min_price_index]} {model[min_price_index]}"

    st.subheader("Statystyki obliczeń na cenie:")
    st.write("Jaka jest średnia, maskymalna i minimalna cena?")
    st.write(f"Średnia cena: {mean_price:.2f} PLN")
    st.write(f"Maksymalna cena: {max_price} PLN modelu: {max_price_car}")
    st.write(f"Mininimalna cena: {min_price} PLN modelu: {min_price_car}")

    # wykonanie obliczeń na cenach i silnikach
    average_price_manual = df_cleaned[df_cleaned['gearbox'] == 'Manualna']['price'].mean()
    average_price_auto = df_cleaned[df_cleaned['gearbox'] == 'Automatyczna']['price'].mean()

    st.subheader("Statystyki obliczeń na danych:")
    st.write("Które rozwiązanie skrzyni biegów jest średnio droższe?")
    st.write(f"Średnia cena dla skrzyni manualnej: {average_price_manual} PLN")
    st.write(f"Średnia cena dla skrzyni automatycznej: {average_price_auto} PLN")

    # 1. wykonanie obliczeń na województwach:
    voivode_counts = df_cleaned[df_cleaned['voivode'] != 'Inny kraj']['voivode'].value_counts()
    voivode_avg_price = df_cleaned[df_cleaned['voivode'] != 'Inny kraj'].groupby('voivode')['price_clean'].mean()
    grouped_by_voivode = df_cleaned.groupby('voivode')

    # 2. Najwięcej samochodów w województwie (liczba samochodów)
    most_cars_voivode = voivode_counts.idxmax()
    most_cars_count = voivode_counts.max()

    # 3. Najmniej samochodów w województwie
    least_cars_voivode = voivode_counts.idxmin()
    least_cars_count = voivode_counts.min()

    # 4. Średnia cena dla województwa z największą liczbą samochodów
    average_price_most_cars = voivode_avg_price[most_cars_voivode]

    # 5. Średnia cena dla województwa z najmniejszą liczbą samochodów
    average_price_least_cars = voivode_avg_price[least_cars_voivode]

    # 6. Największa średnia cena w województwie
    most_expensive_voivode = grouped_by_voivode['price'].mean().idxmax()
    most_expensive_avg_price = grouped_by_voivode['price'].mean().max()

    # 7. Najmniejsza średnia cena w województwie
    least_expensive_voivode = grouped_by_voivode['price'].mean().idxmin()
    least_expensive_avg_price = grouped_by_voivode['price'].mean().min()

    # 8. Zlcizenie aut zza granicy i ich średnią
    foreign_cars_count = (df_cleaned['voivode'] == 'Inny kraj').sum()
    average_price_foreign = df_cleaned[df_cleaned['voivode'] == 'Inny kraj']['price_clean'].mean()

    st.header("Najwięcej i najmniej samochodów na sprzedaż pod względem województw")
    st.write(f"\n**Które województwo ma na sprzedaż najwięcej i najmniej samochodów oraz jaka jest ich średnia?**")
    st.write(f"Województwo **{most_cars_voivode}** sprzedające najwięcej samochodów: **{most_cars_count}** ze średnią ceną: **{average_price_most_cars:.2f} PLN**")
    st.write(f"Województwo **{least_cars_voivode}** sprzedające najmniej samochodów: **{least_cars_count}** ze średnią ceną: **{average_price_least_cars:.2f} PLN**")

    st.header("Najwyższa i najniższa średnia cena w województwach")
    st.write(f"\n**Które województwo ma najwyższą średnią cenę a które najmniejszą?**")
    st.write(f"Województwo **{most_expensive_voivode}** ze średnią najwyższą ceną: **{most_expensive_avg_price:.2f} PLN**")
    st.write(f"Województwo **{least_expensive_voivode}** ze średnią najniższą ceną: **{least_expensive_avg_price:.2f} PLN**")

    st.header("Oferty samochodów zza granicy")
    st.write(f"\n**Ile jest ofert sprzedaży samochodów za granicą i jaka jest ich średnia cena?**")
    st.write(f"Liczba samochodów sprzedanych za granicą: **{foreign_cars_count}** ze średnią ceną: **{average_price_foreign:.2f} PLN**")

    st.header("Najpopularniejsze marki samochodów:")
    st.write(df["brand"].value_counts().head(20))

    st.header("Najpopularniejsze typy paliwa")
    st.write(df["fuel"].value_counts())

    st.header("Najpopularniejsze skrzynie biegów")
    st.write(df["gearbox"].value_counts())

with tabs[2]:
    st.title('Wykresy przedstawiające graficznie dane sprzedaży')
    st.write("""
     W tej części przedstawione zostają różne wykresy testujące różne dane i zależności miedzy nimi
     """)

    # Wykres najwięcej sprzedawanych marek
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(x=df["brand"].value_counts().head(20).index,
                y=df["brand"].value_counts().head(20).values,
                palette="viridis", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title("Top 20 najpopularniejszych marek")
    ax.set_ylabel("Liczba ogłoszeń")
    st.pyplot(fig)

    # Wykres rozkładu cen
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df[df["price_clean"] <= 300_000]["price_clean"], bins=150, kde=True, color="blue", ax=ax)
    ax.set_title("Rozkład cen samochodów (do 300 000 zł)")
    ax.set_xlabel("Cena (PLN)")
    ax.set_ylabel("Liczba samochodów")
    st.pyplot(fig)

    # Wykres zależności: rok produkcji vs. cena
    fig, ax = plt.subplots(figsize=(16, 8))
    df_filtered = df[df["price"] <= 1_000_000]
    sns.scatterplot(data=df_filtered, x="price", y="year", alpha=0.5, ax=ax)
    ax.set_title("Zależność cena vs rok produkcji (do 1 000 000 zł)")
    ax.set_xlabel("Cena (PLN)")
    ax.set_ylabel("Rok produkcji")
    ax.set_xlim(0, df_filtered["price_clean"].max())
    st.pyplot(fig)

    # Wykres korelacji między zmiennymi liczbowymi
    fig, ax = plt.subplots(figsize=(10, 6))  # Tworzymy figurę i oś
    sns.heatmap(df[["price_clean", "KM_clean", "mileage_clean", "year_clean"]].corr(), annot=True, cmap="coolwarm",
                fmt=".2f", ax=ax)
    ax.set_title("Macierz korelacji zmiennych numerycznych")
    st.pyplot(fig)

    # Wykres: Cena pojazdów w zależności od roku
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Cena pojazdów w zależności od roku")
    sns.boxplot(x='year', y='price', data=df, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

with tabs[3]:
    st.title('Tabele przedstawiające dane ogłoszeń motoryzacyjnych')
    st.write("""
      Tabele pokazane tutaj są z różnego zakresu analizy
      """)

    # Wyświetlanie ogólnych statystyk
    st.subheader('Podstawowe statystyki danych')
    st.write(df.describe())

    # Filtry: wybór województwa
    wojewodztwa = df['voivode'].unique()
    wojewodztwo = st.selectbox('Wybierz województwo:', wojewodztwa)

    df_woj = df[df['voivode'] == wojewodztwo]
    st.subheader(f'Dane z województwa {wojewodztwo}')
    st.write(df_woj)

    # chi2 i tabela krzyżowa dla marki i województwa
    brand_counts = df['brand'].value_counts()
    filtered_brands = brand_counts[brand_counts >= 100]
    df_filtered_brand = df[df['brand'].isin(filtered_brands.index)]

    cross_tab = pd.crosstab(df_filtered['brand'], df_filtered['voivode'])
    st.write(f"\nTabela krzyżowa dla marki i województwa")
    st.write(f"\n{cross_tab}")
    chi2, p, dof, expected = chi2_contingency(cross_tab)
    st.write(f"\nChi2: {chi2}, p-value: {p}")

    # Regresja liniowa ceny i przebiegu
    st.write(f"\nRegresja liniowa ceny i przebiegu")
    df['log_price'] = np.log(df['price_clean'])
    df['log_mileage'] = np.log(df['mileage_clean'] + 1)

    X = df[["mileage_clean"]]
    y = df["price_clean"]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    st.write(model.summary())

with tabs[4]:
    st.title('Machine learning - predykcja cen')
    st.write("""
        Ostatnia zakładka, w której model uczenia maszynowego na podstawie podanych mu zmiennych ocenia i próbuje z jakąś dokładnością przewidzieć cenę samochodów.
        Wykorzystano dwa modele:
        - Regresja liniowa
        - Random Forest
        \nOba sposoby zostały przetestowane, aby ustalić, która metoda machine learningu zwróci lepszy wynik uczenia i przewidywania ceny.
          """)

    df = df.dropna(subset=['price', 'year', 'mileage', 'KM', 'engine', 'fuel', 'gearbox'])

    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0).astype(int)
    df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
    df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce').fillna(0).astype(int)
    df['KM'] = pd.to_numeric(df['KM'], errors='coerce').fillna(0).astype(int)
    df['engine'] = pd.to_numeric(df['engine'], errors='coerce').fillna(0)

    # Wybór cech (features) i zmiennej docelowej (target)
    features = ['year', 'mileage', 'KM', 'engine', 'fuel', 'gearbox']
    target = 'price'

    # Konwersja zmiennych kategorycznych (OneHotEncoding)
    df = pd.get_dummies(df, columns=['fuel', 'gearbox'], drop_first=True)

    X = df.drop(columns=['price', 'price_clean', 'log_price', 'brand', 'model', 'place', 'voivode'])
    y = df['price']

    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standaryzacja danych
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model 1: Regresja liniowa
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lin = lin_reg.predict(X_test)

    # Model 2: Random Forest
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_train, y_train)
    y_pred_rf = rf_reg.predict(X_test)

    # Ocena modeli
    r2_lin = r2_score(y_test, y_pred_lin)
    r2_rf = r2_score(y_test, y_pred_rf)

    st.write(f"Regresja liniowa: R² = {r2_lin:.3f}")
    st.write(f"Random Forest: R² = {r2_rf:.3f}")

    # Wizualizacja wyników
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(x=y_test, y=y_pred_rf, alpha=0.5, ax=ax)
    ax.set_xlabel("Rzeczywista cena")
    ax.set_ylabel("Przewidziana cena")
    ax.set_title("Predykcja cen samochodów - Random Forest")
    st.pyplot(fig)