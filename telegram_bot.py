# telegram_bot.py
import pandas as pd
import psycopg2 # Используем psycopg2 для подключения
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    filters, ContextTypes
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

# --- (1) ЖАҢАРТЫЛҒАН ЛОГИКА: Деректерді жүктеу және модельді дайындау ---

def load_data_from_db():
    """
    PostgreSQL-ден деректерді жүктейді, дубликаттарды алып тастайды,
    тақырыптарға салмақ қосады (x3) және TF-IDF моделін дайындайды.
    Сондай-ақ батырмалар үшін санаттар тізімін қайтарады.
    """
    if not DATABASE_URL:
        logger.error("Критическая ошибка: DATABASE_URL не найден!")
        exit()
        
    try:
        logger.info("Подключение к PostgreSQL...")
        with psycopg2.connect(DATABASE_URL) as conn:
            sql = "SELECT * FROM books"
            df = pd.read_sql_query(sql, conn)
        
        logger.info(f"Загружено {len(df)} книг из БД.")
        
        # --- Деректерді тазалау (test_model.py алынған) ---
        df = df.drop_duplicates(subset=['Аты'])
        logger.info(f"Дубликаттар алынып тасталды. Қалғаны: {len(df)} кітап.")

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
        
        tfidf_vectorizer = TfidfVectorizer(stop_words=kazakh_stop_words)
        
        # --- Жақсарту (test_model.py алынған) ---
        logger.info("Улучшение: Применяем веса (Название x3)...")
        df['combined'] = (
            (df['Аты'].str.lower() * 3) + ' ' + # <--- УЛУЧШЕНИЕ
            df['Толығырақ'].str.lower() + ' ' + 
            df['Автор'].str.lower() + ' ' + 
            df['Category_Numeric'].astype(str)
        )
        
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined'])
        
        logger.info("Модель TF-IDF (Векторизатор и Матрица) готова!")
        
        # --- Батырмалар үшін санаттарды алу ---
        categories = [cat for cat in df['Категория'].unique() if cat != 'Белгісіз']
        top_categories = sorted(categories)[:9] # Алғашқы 9-ын аламыз (немесе сұрыптаймыз)
        
        return df, tfidf_vectorizer, tfidf_matrix, top_categories

    except Exception as e:
        logger.error(f"Критическая ошибка при загрузке данных из БД: {e}")
        exit()

# --- Глобалды айнымалылар: 1 рет жүктеу ---
books_df, tfidf, tfidf_matrix, TOP_CATEGORIES = load_data_from_db()

# --- Тұрақты пернетақтаны құру ---
# 3x3 тор (grid) жасаймыз
keyboard_layout = [TOP_CATEGORIES[i:i + 3] for i in range(0, len(TOP_CATEGORIES) - (len(TOP_CATEGORIES) % 3), 3)]
# Қалған батырмаларды қосамыз
remaining = TOP_CATEGORIES[len(TOP_CATEGORIES) - (len(TOP_CATEGORIES) % 3):]
if remaining:
    keyboard_layout.append(remaining)

PERSISTENT_KEYBOARD = ReplyKeyboardMarkup(keyboard_layout, resize_keyboard=True)

# --- (2) ІЗДЕУ ЖӘНЕ ФОРМАТТАУ ЛОГИКАСЫ ---

def find_books_by_keywords(query: str) -> pd.DataFrame:
    """
    Кілттік сөздер бойынша кітаптарды іздейді, TF-IDF моделін қолданады.
    (Бұл функция енді жаһандық 'tfidf' және 'tfidf_matrix' айнымалыларын пайдаланады)
    """
    try:
        query = query.lower()
        query_vector = tfidf.transform([query])
        scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        if scores.max() == 0:
            return pd.DataFrame() # Ештеңе табылмады

        top_indices = scores.argsort()[:-6:-1]
        top_scores = scores[top_indices]
        valid_indices = top_indices[top_scores > 0]
        
        if len(valid_indices) == 0:
            return pd.DataFrame()
            
        return books_df.iloc[valid_indices]
    except Exception as e:
        logger.error(f"Ошибка в find_books_by_keywords: {e}")
        return pd.DataFrame()

def format_book_message(book: pd.Series) -> str:
    """Кітап туралы ақпаратты форматтайды (өзгеріссіз)"""
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

# --- (3) TELEGRAM ОБРАБОТЧИКТЕРІ ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /start командасын өңдеуші.
    Сәлемдесу хабарламасын ЖӘНЕ тұрақты пернетақтаны жібереді.
    """
    user = update.effective_user
    await update.message.reply_html(
        f"Сәлем, {user.mention_html()}! 👋\n\n"
        f"Төмендегі санатты (категорияны) таңдаңыз, немесе кітап атауын/сипаттамасын жазыңыз:",
        reply_markup=PERSISTENT_KEYBOARD # <--- ЖАҢА ТҰРАҚТЫ ПЕРНЕТАҚТА
    )

async def handle_category_click(category_name: str, update: Update):
    """
    Санат батырмасы басылғанда іске қосылады.
    Осы санаттан кездейсоқ кітаптарды көрсетеді.
    """
    logger.info(f"Пользователь выбрал категорию: {category_name}")
    category_books = books_df[books_df['Категория'] == category_name]
    
    if category_books.empty:
        await update.message.reply_text(f"<b>{category_name}</b> санатында кітаптар табылмады.", parse_mode='HTML')
        return
        
    sample_books = category_books.sample(min(5, len(category_books)))
    response_message = f"<b>{category_name}</b> санатындағы кездейсоқ кітаптар:\n\n"
    for _, book in sample_books.iterrows():
        response_message += format_book_message(book)
    # Тұрақты пернетақтаны қайта жіберудің қажеті жоқ, ол орнында қалады.
    await update.message.reply_html(response_message, disable_web_page_preview=True)

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    БАРЛЫҚ кіріс мәтінді өңдейді.
    Мәтіннің батырма немесе іздеу сөзі екенін тексереді.
    """
    query_text = update.message.text
    
    # ТЕКСЕРУ: Бұл батырма ма, әлде іздеу ме?
    if query_text in TOP_CATEGORIES:
        # Егер бұл біздің батырмалардың бірі болса:
        await handle_category_click(query_text, update)
    else:
        # Егер бұл кәдімгі іздеу сөзі болса:
        logger.info(f"Пользователь ищет по ключам: '{query_text}'")
        try:
            found_books = find_books_by_keywords(query_text)
            
            if found_books.empty:
                await update.message.reply_text("Кешіріңіз, осы сөздер бойынша ештеңе табылмады. 😕")
                return
                
            response_message = f"<b>'{query_text}'</b> сөздері бойынша іздеу нәтижелері:\n\n"
            for _, book in found_books.iterrows():
                response_message += format_book_message(book)
            await update.message.reply_html(response_message, disable_web_page_preview=True)
            
        except Exception as e:
            logger.error(f"Ошибка при поиске по ключам: {e}")
            await update.message.reply_text("Ой, бір қателік орын алды.")

def main():
    """Болты іске қосу (басты функция)"""
    if not TOKEN:
        logger.error("!!! ТОКЕН НЕ УСТАНОВЛЕН !!!")
        logger.error("Проверьте .env файл или переменные окружения хостинга.")
        return

    application = Application.builder().token(TOKEN).build()
    
    # Командалар
    application.add_handler(CommandHandler("start", start))
    
    # 'button_callback' енді қажет емес.
    # 'handle_text_message' енді бәрін өңдейді (батырмаларды да, іздеуді де).
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))

    logger.info("Бот запускается...")
    application.run_polling()

if __name__ == "__main__":
    main()

