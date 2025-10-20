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
# Загружаем переменные из .env (для локального запуска)
# На хостинге (Railway) переменные будут заданы в интерфейсе
load_dotenv() 

TOKEN = os.getenv("TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")

# --- (1) ЗАГРУЗКА ДАННЫХ ИЗ POSTGRES И ПОДГОТОВКА МОДЕЛИ ---

def load_data_from_db():
    """Загружает данные из PostgreSQL в DataFrame и готовит модель."""
    
    if not DATABASE_URL:
        logger.error("Критическая ошибка: DATABASE_URL не найден!")
        exit()
        
    try:
        logger.info("Подключение к PostgreSQL...")
        # Подключаемся к базе
        with psycopg2.connect(DATABASE_URL) as conn:
            # Загружаем все данные из таблицы 'books' в pandas DataFrame
            sql = "SELECT * FROM books"
            df = pd.read_sql_query(sql, conn)
        
        logger.info(f"Загружено {len(df)} книг из БД.")
        
        # --- Дальнейшая логика идентична SQLite ---
        
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
        
        tfidf = TfidfVectorizer(stop_words=kazakh_stop_words)
        
        df['combined'] = (
            df['Аты'].str.lower() + ' ' + 
            df['Толығырақ'].str.lower() + ' ' + 
            df['Автор'].str.lower() + ' ' + 
            df['Category_Numeric'].astype(str)
        )
        
        tfidf_matrix = tfidf.fit_transform(df['combined'])
        cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        logger.info("Модель рекомендаций (TF-IDF) готова!")
        
        return df, cosine_sim_matrix

    except Exception as e:
        logger.error(f"Критическая ошибка при загрузке данных из БД: {e}")
        exit()

# --- Глобальные переменные: загружаем данные 1 раз при старте ---
books_df, cosine_sim = load_data_from_db()

# --- (2) ЛОГИКА РЕКОМЕНДАЦИЙ (остается без изменений) ---

def get_recommendations(book_title: str) -> pd.DataFrame:
    book_title = book_title.lower()
    idx_list = books_df.index[
        (books_df['Аты'].str.lower().str.contains(book_title, na=False)) | 
        (books_df['Толығырақ'].str.lower().str.contains(book_title, na=False))
    ].tolist()
    
    if not idx_list: return pd.DataFrame()
    idx = idx_list[0]
    top_indices = [i[0] for i in sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:6]]
    return books_df.iloc[top_indices]

def format_book_message(book: pd.Series) -> str:
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

# --- (3) ОБРАБОТЧИКИ TELEGRAM (остаются без изменений) ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
        f"Төмендегі санатты (категорияны) таңдаңыз, немесе кітап атауын жазыңыз:",
        reply_markup=reply_markup
    )

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    book_title_query = update.message.text
    logger.info(f"Пользователь ищет: '{book_title_query}'")
    try:
        recommendations = get_recommendations(book_title_query)
        if recommendations.empty:
            await update.message.reply_text("Кешіріңіз, ұсыныстар табылмады. 😕")
            return
        response_message = f"<b>'{book_title_query}'</b> кітабына ұқсас ұсыныстар:\n\n"
        for _, book in recommendations.iterrows():
            response_message += format_book_message(book)
        await update.message.reply_html(response_message, disable_web_page_preview=True)
    except Exception as e:
        logger.error(f"Ошибка при поиске: {e}")
        await update.message.reply_text("Ой, бір қателік орын алды.")

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
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