# telegram_bot.py
import pandas as pd
import psycopg2 # Используем psycopg2 для подключения
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    filters, ContextTypes, CallbackQueryHandler
)
import logging
import os
from dotenv import load_dotenv

# --- Настройка ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Загрузка секретов ---
load_dotenv() 
TOKEN = os.getenv("TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")

# --- (1) ЗАГРУЗКА ДАННЫХ ИЗ POSTGRES И ПОДГОТОВКА МОДЕЛИ ---

def load_data_from_db():
    """Загружает данные из PostgreSQL, создает TF-IDF Векторизатор и Матрицу."""
    
    if not DATABASE_URL:
        logger.error("Критическая ошибка: DATABASE_URL не найден!")
        exit()
        
    try:
        logger.info("Подключение к PostgreSQL...")
        with psycopg2.connect(DATABASE_URL) as conn:
            sql = "SELECT * FROM books"
            df = pd.read_sql_query(sql, conn)
        
        logger.info(f"Загружено {len(df)} книг из БД.")
        
        # --- Подготовка данных ---
        df['Аты'] = df['Аты'].fillna('')
        df['Категория'] = df['Категория'].fillna('Белгісіз')
        df['Толығырақ'] = df['Толығырақ'].fillna('')
        df['Автор'] = df['Автор'].fillna('')

        category_mapping = {cat: i for i, cat in enumerate(df['Категория'].unique())}
        df['Category_Numeric'] = df['Категория'].map(category_mapping).fillna(-1)

        kazakh_stop_words = [
            'және', 'немесе', 'бірақ', 'үшін', 'дейін', 'бастап', 'жоқ', 'бар', 
            'барлық', 'әр', 'көп', 'аз', 'болады'
        ]
        
        # --- ИЗМЕНЕНИЕ: Создаем и сохраняем Векторизатор (tfidf) ---
        tfidf_vectorizer = TfidfVectorizer(stop_words=kazakh_stop_words)
        
        df['combined'] = (
            df['Аты'].str.lower() + ' ' + 
            df['Толығырақ'].str.lower() + ' ' + 
            df['Автор'].str.lower() + ' ' + 
            df['Category_Numeric'].astype(str)
        )
        
        # Обучаем векторизатор и создаем матрицу для ВСЕХ книг
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined'])
        
        logger.info("Модель TF-IDF (Векторизатор и Матрица) готова!")
        
        # Возвращаем DF, Векторизатор и Матрицу
        return df, tfidf_vectorizer, tfidf_matrix

    except Exception as e:
        logger.error(f"Критическая ошибка при загрузке данных из БД: {e}")
        exit()

# --- Глобальные переменные: загружаем данные 1 раз при старте ---
# ИЗМЕНЕНИЕ: сохраняем tfidf (векторизатор) и tfidf_matrix
books_df, tfidf, tfidf_matrix = load_data_from_db()

# --- (2) НОВАЯ ЛОГИКА ПОИСКА ПО КЛЮЧЕВЫМ СЛОВАМ ---

def find_books_by_keywords(query: str) -> pd.DataFrame:
    """
    Ищет книги по ключевым словам из запроса, сравнивая
    запрос со всеми книгами в tfidf_matrix.
    """
    try:
        query = query.lower()
        
        # 1. Векторизуем текстовый запрос пользователя
        #    Используем .transform() (не .fit_transform()), 
        #    так как модель уже обучена
        query_vector = tfidf.transform([query])
        
        # 2. Считаем схожесть (cosine similarity) запроса со ВСЕМИ книгами
        #    (query_vector: 1xN, tfidf_matrix: 177xN) -> (scores: 1x177)
        scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        # 3. Находим 5 лучших (non-zero) результатов
        if scores.max() == 0:
            return pd.DataFrame() # Ничего не найдено

        # argsort() сортирует от меньшего к большему
        # [:-6:-1] берет 5 последних (самых больших) в обратном порядке
        top_indices = scores.argsort()[:-6:-1]
        
        # 4. Фильтруем те, у которых 0 схожесть
        top_scores = scores[top_indices]
        valid_indices = top_indices[top_scores > 0]
        
        if len(valid_indices) == 0:
            return pd.DataFrame()
            
        return books_df.iloc[valid_indices]
    except Exception as e:
        logger.error(f"Ошибка в find_books_by_keywords: {e}")
        return pd.DataFrame()


def format_book_message(book: pd.Series) -> str:
    """Форматирует вывод книги (без изменений)"""
    if pd.notna(book['Бағасы']):
        price = f"{book['Бағасы']:,.0f} тг".replace(',', ' ')
    else:
        price = "Бағасы белгісіз"
    
    message = (
        f"📚 <b>{book['Аты']}</b>\n"
        f"👤 Авторы: {book['Автор']}\n"
        f"🗂 Категория: {book['Категория']}\n"
        f"💰 Бағасы: {price}\n"
    )
    if pd.notna(book['URL']):
         message += f"🔗 <a href=\"{book['URL']}\">Сайттан қарау</a>\n"
    return message + "\n"

# --- (3) ОБРАБОТЧИКИ TELEGRAM ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик /start (без изменений)"""
    user = update.effective_user
    categories = [cat for cat in books_df['Категория'].unique() if cat != 'Белгісіз']
    top_categories = categories[:9] 
    
    keyboard = []
    row = []
    for category in top_categories:
        row.append(InlineKeyboardButton(category, callback_data=f"cat_{category}"))
        if len(row) == 3:
            keyboard.append(row)
            row = []
    if row: keyboard.append(row)
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_html(
        f"Сәлем, {user.mention_html()}! 👋\n\n"
        f"Төмендегі санатты (категорияны) таңдаңыз, немесе кітап атауын/сипаттамасын жазыңыз:", # "или напишите название/описание книги"
        reply_markup=reply_markup
    )

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обрабатывает текстовый запрос пользователя, 
    ИСПОЛЬЗУЯ НОВУЮ ФУНКЦИЮ ПОИСКА
    """
    query_text = update.message.text
    logger.info(f"Пользователь ищет по ключам: '{query_text}'")
    try:
        # ИЗМЕНЕНИЕ: Вызываем новую функцию
        found_books = find_books_by_keywords(query_text)
        
        if found_books.empty:
            await update.message.reply_text("Кешіріңіз, осы сөздер бойынша ештеңе табылмады. 😕") # "Sorry, nothing found for these words"
            return
            
        # ИЗМЕНЕНИЕ: Меняем заголовок ответа
        response_message = f"<b>'{query_text}'</b> сөздері бойынша іздеу нәтижелері:\n\n" # "Search results for the words:"
        for _, book in found_books.iterrows():
            response_message += format_book_message(book)
        await update.message.reply_html(response_message, disable_web_page_preview=True)
        
    except Exception as e:
        logger.error(f"Ошибка при поиске по ключам: {e}")
        await update.message.reply_text("Ой, бір қателік орын алды.")

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик кнопок (без изменений)"""
    query = update.callback_query
    await query.answer()
    callback_data = query.data
    
    if callback_data.startswith("cat_"):
        category_name = callback_data.split("_", 1)[1]
        logger.info(f"Нажата категория: {category_name}")
        category_books = books_df[books_df['Категория'] == category_name]
        
        if category_books.empty:
            await query.message.reply_text(f"<b>{category_name}</b> санатында кітаптар табылмады.", parse_mode='HTML')
            return
            
        sample_books = category_books.sample(min(5, len(category_books)))
        response_message = f"<b>{category_name}</b> санатындағы кездейсоқ кітаптар:\n\n"
        for _, book in sample_books.iterrows():
            response_message += format_book_message(book)
        await query.message.reply_html(response_message, disable_web_page_preview=True)

def main():
    """Главная функция (без изменений)"""
    if not TOKEN:
        logger.error("!!! ТОКЕН НЕ УСТАНОВЛЕН !!!")
        logger.error("Проверьте .env файл или переменные окружения хостинга.")
        return

    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))

    logger.info("Бот запускается...")
    application.run_polling()

if __name__ == "__main__":
    main()
