import re
import json
import time
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error


class SmartEstateAI:
    BASE_URL = "https://www.cian.ru"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    CITY_MAP = {
        "москва": "kupit-kvartiru-moskva/",
        "санкт-петербург": "kupit-kvartiru-sankt-peterburg/",
        "новосибирск": "kupit-kvartiru-novosibirsk/",
        "екатеринбург": "kupit-kvartiru-ekaterinburg/",
    }

    def __init__(self, city: str = "москва"):
        self.city = city.lower()
        self.data = None
        self.model = None
        self.scaler = None
        self.feature_columns = ["total_area", "rooms", "floor", "floors_total"]

    # ---------- Парсинг (прежний, без изменений) ----------
    def _build_url(self, page: int = 1) -> str:
        path = self.CITY_MAP.get(self.city, f"kupit-kvartiru-{self.city}/")
        return f"{self.BASE_URL}/{path}?page={page}"

    def _extract_initial_state(self, html: str) -> Optional[dict]:
        soup = BeautifulSoup(html, "html.parser")
        scripts = soup.find_all("script")
        for script in scripts:
            if script.string and ("window.__NUXT__" in script.string or "window.__INITIAL_STATE__" in script.string):
                match = re.search(r'window\.__(?:NUXT|INITIAL_STATE)__\s*=\s*({.*?});', script.string, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group(1))
                    except json.JSONDecodeError:
                        continue
        return None

    def _parse_offer(self, offer: dict) -> Dict:
        geo = offer.get("geo", {})
        address_info = geo.get("address", [])
        address = ", ".join([item.get("name", "") for item in address_info if item.get("name")])

        metro_list = []
        for metro in geo.get("undergrounds", []):
            metro_list.append(f"{metro.get('name', '')} ({metro.get('time', '')} мин.)")
        metro = "; ".join(metro_list) if metro_list else None

        total_area = offer.get("totalArea")
        rooms = offer.get("roomsCount")
        floor = offer.get("floorNumber")
        floors_total = offer.get("floorsCount")
        price = offer.get("bargainTerms", {}).get("price")
        offer_type = offer.get("offerType")

        return {
            "address": address,
            "metro": metro,
            "price": price,
            "total_area": total_area,
            "rooms": rooms,
            "floor": floor,
            "floors_total": floors_total,
            "offer_type": offer_type,
            "url": f"{self.BASE_URL}{offer.get('fullUrl', '')}"
        }

    def parse_listings(self, max_pages: int = 1) -> pd.DataFrame:
        all_offers = []
        for page in range(1, max_pages + 1):
            url = self._build_url(page)
            print(f"Парсинг страницы {page}: {url}")
            try:
                resp = requests.get(url, headers=self.HEADERS, timeout=15)
                resp.raise_for_status()
                state = self._extract_initial_state(resp.text)
                if not state:
                    print(f"  Не удалось извлечь JSON со страницы {page}")
                    continue

                offers = []
                if "page-data" in state:
                    offers = state["page-data"].get("search", {}).get("offers", [])
                elif "offers" in state:
                    offers = state["offers"]

                if not offers:
                    print(f"  Нет объявлений на странице {page}")
                    break

                for offer in offers:
                    try:
                        parsed = self._parse_offer(offer)
                        all_offers.append(parsed)
                    except Exception as e:
                        print(f"  Ошибка разбора объявления: {e}")
                print(f"  Собрано {len(offers)} предложений")
                time.sleep(1.5)
            except requests.RequestException as e:
                print(f"  Ошибка при запросе: {e}")
                break

        self.data = pd.DataFrame(all_offers)
        return self.data

    # ---------- Очистка данных ----------
    def clean_data(self) -> pd.DataFrame:
        if self.data is None or self.data.empty:
            print("Нет данных для очистки. Запустите parse_listings()")
            return pd.DataFrame()

        df = self.data.copy()

        if "url" in df.columns:
            df.drop_duplicates(subset="url", inplace=True)

        df["price"] = df["price"].astype(str).str.replace(r"\D", "", regex=True)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["total_area"] = pd.to_numeric(df["total_area"], errors="coerce")
        df["floor"] = pd.to_numeric(df["floor"], errors="coerce")
        df["floors_total"] = pd.to_numeric(df["floors_total"], errors="coerce")
        df["rooms"] = pd.to_numeric(df["rooms"], errors="coerce")

        df.dropna(subset=["price", "total_area"], inplace=True)
        df = df[(df["price"] > 1_000_000) & (df["total_area"].between(10, 500))]
        df["price_per_sqm"] = (df["price"] / df["total_area"]).round(2)
        df = df[df["floor"] <= df["floors_total"]]

        df["metro"] = df["metro"].fillna("Не указано")
        df["rooms"] = df["rooms"].fillna(df["rooms"].median()).astype(int)
        df["address"] = df["address"].fillna("Не указан")

        df.sort_values("price", inplace=True)
        df.reset_index(drop=True, inplace=True)
        self.data = df
        return self.data

    # ---------- Новый метод: baseline-модель и метрики ----------
    def train_baseline(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[float, float]:
        """
        Обучает baseline-модель (Ridge-регрессия) на числовых признаках.
        Возвращает RMSE и MAE на тестовой выборке.
        """
        if self.data is None or len(self.data) < 10:
            print("Недостаточно данных для обучения (нужно хотя бы 10 записей). Сначала запустите parse_listings() и clean_data()")
            return None, None

        df = self.data.copy()

        # Проверим, что все нужные колонки есть
        missing_cols = set(self.feature_columns) - set(df.columns)
        if missing_cols:
            print(f"Отсутствуют признаки: {missing_cols}. Обучение невозможно.")
            return None, None

        # Удаляем строки с пропусками в признаках
        df_model = df.dropna(subset=self.feature_columns + ["price"])
        if len(df_model) < 10:
            print(f"После удаления пропусков осталось {len(df_model)} записей — недостаточно.")
            return None, None

        X = df_model[self.feature_columns].values
        y = df_model["price"].values

        # Разделение
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Масштабирование
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Модель
        self.model = Ridge(alpha=1.0)
        self.model.fit(X_train_scaled, y_train)

        # Предсказание
        y_pred = self.model.predict(X_test_scaled)

        # Метрики
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        # Вывод результатов
        print("\n" + "=" * 50)
        print("РЕЗУЛЬТАТЫ BASELINE-МОДЕЛИ")
        print("=" * 50)
        print(f"Признаки: {self.feature_columns}")
        print(f"Количество объектов: обучение {len(X_train)}, тест {len(X_test)}")
        print(f"Средняя цена в данных: {y.mean():,.0f} руб.")
        print(f"RMSE (корень из среднеквадратичной ошибки): {rmse:,.0f} руб.")
        print(f"MAE  (средняя абсолютная ошибка):         {mae:,.0f} руб.")
        print(f"Средняя цена за кв.м. (по всем данным): {df['price_per_sqm'].mean():,.0f} руб.")
        # Дополнительно: сравним с наивным прогнозом (среднее)
        y_mean = np.mean(y_train)
        rmse_mean = np.sqrt(mean_squared_error(y_test, [y_mean] * len(y_test)))
        mae_mean = mean_absolute_error(y_test, [y_mean] * len(y_test))
        print(f"\nНаивный baseline (средняя цена обучения):")
        print(f"  RMSE: {rmse_mean:,.0f} руб.")
        print(f"  MAE:  {mae_mean:,.0f} руб.")
        print("=" * 50)

        return rmse, mae


if __name__ == "__main__":
    # Инициализация для Москвы
    bot = SmartEstateAI(city="москва")

    # Попытка загрузить ранее сохранённые данные, чтобы не парсить каждый раз
    try:
        existing_data = pd.read_csv("smart_estate_data.csv", encoding="utf-8-sig")
        if len(existing_data) >= 10:
            print("Загружены сохранённые данные (smart_estate_data.csv)")
            bot.data = existing_data
        else:
            raise ValueError("Мало данных в CSV")
    except (FileNotFoundError, ValueError):
        print("CSV не найден или недостаточно данных, выполняем парсинг...")
        bot.parse_listings(max_pages=2)   # осторожно, не более 2 страниц в тестовом режиме
        bot.clean_data()
        bot.data.to_csv("smart_estate_data.csv", index=False, encoding="utf-8-sig")
        print("Данные сохранены в smart_estate_data.csv")

    # Запуск baseline
    bot.train_baseline()
