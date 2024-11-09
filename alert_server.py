import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import telegram
import logging

app = FastAPI()

# ----------------------------
# Configure Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

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
# Pydantic Models
# ----------------------------
class Alert(BaseModel):
    message: str


# ----------------------------
# API Endpoint to Receive Text Alerts
# ----------------------------
@app.post("/send_alert")
async def send_alert(alert: Alert, api_key: str = Form(...)):
    # Optional: Implement API key authentication here if needed
    try:
        await bot.send_message(chat_id=USER_CHAT_ID, text=alert.message)
        logger.info(f"Alert sent to Telegram: {alert.message}")
        return {"status": "success", "message": "Alert sent successfully."}
    except Exception as e:
        logger.error(f"Error sending alert to Telegram: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
# API Endpoint to Receive and Send Images with Captions
# ----------------------------
@app.post("/send_image")
async def send_image(
    file: UploadFile = File(...),
    caption: str = Form(...),
    # api_key: str = Form(...)
):
    """
    Endpoint to receive an image file and a caption, then send them to Telegram.

    **Parameters:**
    - `file`: The image file to be uploaded.
    - `caption`: The caption text accompanying the image.
    - `api_key`: API key for authentication (optional based on your security setup).

    **Returns:**
    - JSON response indicating success or failure.
    """
    # Optional: Implement API key authentication
    # if api_key != os.environ.get("API_KEY"):
    #     logger.warning("Unauthorized access attempt.")
    #     raise HTTPException(status_code=403, detail="Forbidden")

    # Validate the uploaded file is an image
    if not file.content_type.startswith("image/"):
        logger.error(f"Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="Invalid image file.")

    try:
        # Read the file content
        image_bytes = await file.read()

        # Send the image to Telegram with the caption
        await bot.send_photo(chat_id=USER_CHAT_ID, photo=image_bytes, caption=caption)
        logger.info(f"Image sent to Telegram: {file.filename} with caption: {caption}")

        return {"status": "success", "message": "Image sent successfully."}
    except Exception as e:
        logger.error(f"Error sending image to Telegram: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
# Run the Server
# ----------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
