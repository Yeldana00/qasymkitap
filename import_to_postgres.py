# import_to_postgres.py
import pandas as pd
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CSV_FILE = 'qasymkitap_data.csv'

def clean_data(df):
    """Применяет всю очистку данных"""
    logger.info("Очистка данных...")
    if 'Бағасы' in df.columns:
        df['Бағасы'] = df['Бағасы'].str.replace(' тг', '', regex=False)
        df['Бағасы'] = pd.to_numeric(df['Бағасы'], errors='coerce')
    
    if 'Категория' in df.columns:
        df['Категория'] = df['Категория'].fillna('Белгісіз')
        df['Категория'] = df['Категория'].str.replace(' әдебиеті', '', regex=False)
        df['Категория'] = df['Категория'].str.strip()

    df['Аты'] = df['Аты'].fillna('')
    df['Толығырақ'] = df['Толығырақ'].fillna('')
    df['Автор'] = df['Автор'].fillna('')
    logger.info("Очистка завершена.")
    return df

def import_data():
    """Читает CSV, очищает и загружает в PostgreSQL."""
    
    # Загружаем секреты из .env файла
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    
    if not db_url:
        logger.error("Ошибка: DATABASE_URL не найден. Проверьте .env файл.")
        return
        
    if not os.path.exists(CSV_FILE):
        logger.error(f"Ошибка: Файл {CSV_FILE} не найден!")
        return

    try:
        logger.info(f"Чтение {CSV_FILE}...")
        df = pd.read_csv(CSV_FILE)
        df_cleaned = clean_data(df)
        
        # Создаем "движок" для подключения к Postgres
        # 'postgresql' в URL — это указание для SQLAlchemy использовать psycopg2
        engine = create_engine(db_url)
        
        logger.info("Подключение к PostgreSQL и загрузка данных...")
        # Загружаем DataFrame в таблицу 'books'
        # if_exists='replace' — удалит старую таблицу, если она есть, и создаст новую
        df_cleaned.to_sql('books', engine, index=False, if_exists='replace')
        
        logger.info("Успех! Данные загружены в таблицу 'books' в PostgreSQL.")
        
        # Проверка
        with engine.connect() as conn:
            result = conn.execute("SELECT COUNT(*) FROM books")
            count = result.scalar()
            logger.info(f"В таблице 'books' теперь {count} записей.")

    except Exception as e:
        logger.error(f"Произошла ошибка при импорте: {e}")

if __name__ == "__main__":
    # Установите библиотеки перед запуском:
    # pip install pandas SQLAlchemy psycopg2-binary python-dotenv
    import_data()