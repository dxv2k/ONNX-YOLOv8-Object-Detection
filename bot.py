import os 
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes


TELE_BOT_TOKEN = os.environ.get("TELE_BOT_TOKEN")
USER_CHAT_ID = "1799254419"  
print(TELE_BOT_TOKEN) 

async def hello(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print(update.message.chat_id)
    await update.message.reply_text(f'Hello {update.effective_user.first_name}')


app = ApplicationBuilder().token(TELE_BOT_TOKEN).build()

app.add_handler(CommandHandler("hello", hello))

app.run_polling()
