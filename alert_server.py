import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import telegram

app = FastAPI()

# ----------------------------
# Telegram Bot Credentials
# ----------------------------
TELE_BOT_TOKEN = os.environ.get("TELE_BOT_TOKEN", "YOUR_TELE_BOT_TOKEN")
USER_CHAT_ID = os.environ.get("TELE_CHAT_ID", "YOUR_TELE_CHAT_ID")

if not TELE_BOT_TOKEN or not USER_CHAT_ID:
    raise ValueError(
        "TELE_BOT_TOKEN and TELE_CHAT_ID must be set as environment variables."
    )

bot = telegram.Bot(token=TELE_BOT_TOKEN)


# ----------------------------
# Pydantic Model for Alert
# ----------------------------
class Alert(BaseModel):
    message: str


# ----------------------------
# API Endpoint to Receive Alerts
# ----------------------------
@app.post("/send_alert")
async def send_alert(alert: Alert):
    try:
        await bot.send_message(chat_id=USER_CHAT_ID, text=alert.message)
        print(f"Alert sent to Telegram: {alert.message}")
        return {"status": "success", "message": "Alert sent successfully."}
    except Exception as e:
        print(f"Error sending alert to Telegram: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
