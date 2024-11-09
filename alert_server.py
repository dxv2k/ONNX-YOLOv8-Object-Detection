import os
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
import telegram
import logging
from PIL import Image
from io import BytesIO

app = FastAPI()

# ----------------------------
# Configure Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,  # Set to INFO to capture general events
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
# Helper Functions
# ----------------------------


def compress_image(image_bytes: bytes, max_size_kb: int = 100) -> bytes:
    """
    Compresses the image to ensure its size is below `max_size_kb` kilobytes.

    Args:
        image_bytes (bytes): Original image in bytes.
        max_size_kb (int): Maximum allowed size in kilobytes.

    Returns:
        bytes: Compressed image in bytes.
    """
    image = Image.open(BytesIO(image_bytes))
    # Convert to JPEG if not already
    if image.format != "JPEG":
        image = image.convert("RGB")
    # Initialize buffer
    buffer = BytesIO()
    # Start with high quality
    quality = 85
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    compressed_image = buffer.getvalue()
    buffer.close()

    # Reduce quality until the image is below max_size_kb
    while len(compressed_image) > max_size_kb * 1024 and quality > 10:
        buffer = BytesIO()
        quality -= 5
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        compressed_image = buffer.getvalue()
        buffer.close()

    return compressed_image


async def send_image_to_telegram(compressed_image: bytes, caption: str, filename: str):
    """
    Sends the compressed image to Telegram with the specified caption.

    Args:
        compressed_image (bytes): Compressed image in bytes.
        caption (str): Caption for the image.
        filename (str): Original filename of the image.
    """
    try:
        await bot.send_photo(
            chat_id=USER_CHAT_ID, photo=compressed_image, caption=caption
        )
        logger.info(f"Image sent to Telegram: {filename} with caption: '{caption}'")
    except Exception as e:
        logger.error(f"Error sending image to Telegram: {e}", exc_info=True)


async def process_and_send_image_bytes(image_bytes: bytes, caption: str, filename: str):
    """
    Processes the image by compressing it and sending it to Telegram.

    Args:
        image_bytes (bytes): Original image in bytes.
        caption (str): Caption for the image.
        filename (str): Original filename of the image.
    """
    try:
        # Compress image
        compressed_image = compress_image(image_bytes)
        original_size_kb = len(image_bytes) / 1024
        compressed_size_kb = len(compressed_image) / 1024
        logger.info(
            f"Image compressed: Original Size = {original_size_kb:.2f} KB, "
            f"Compressed Size = {compressed_size_kb:.2f} KB"
        )
        # Send image to Telegram
        await send_image_to_telegram(compressed_image, caption, filename)
    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)


# ----------------------------
# API Endpoint to Receive Text Alerts
# ----------------------------
@app.post("/send_alert")
async def send_alert(alert: Alert):
    """
    Endpoint to send text alerts to Telegram.

    Args:
        alert (Alert): Pydantic model containing the alert message.

    Returns:
        dict: Status message.
    """
    try:
        await bot.send_message(chat_id=USER_CHAT_ID, text=alert.message)
        logger.info(f"Alert sent to Telegram: {alert.message}")
        return {"status": "success", "message": "Alert sent successfully."}
    except Exception as e:
        logger.error(f"Error sending alert to Telegram: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
# API Endpoint to Receive and Send Images
# ----------------------------
@app.post("/send_image")
async def send_image(
    file: UploadFile = File(...),
    caption: str = "Detected Image",
    background_tasks: BackgroundTasks = None,
):
    """
    Endpoint to receive an image, compress it, and send it to Telegram with a caption.

    Args:
        file (UploadFile): The image file to be uploaded.
        caption (str, optional): Caption for the image. Defaults to "Detected Image".
        background_tasks (BackgroundTasks): FastAPI's BackgroundTasks for asynchronous processing.

    Returns:
        dict: Status message indicating that the image is being processed.
    """
    # Validate the uploaded file is an image
    if not file.content_type.startswith("image/"):
        logger.error(f"Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Read the image bytes
    try:
        image_bytes = await file.read()
    except Exception as e:
        logger.error(f"Error reading file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error reading file.")

    # Add background task to process and send image
    background_tasks.add_task(
        process_and_send_image_bytes, image_bytes, caption, file.filename
    )

    logger.info(
        f"Received image for sending: {file.filename} with caption: '{caption}'"
    )
    return {
        "status": "accepted",
        "message": "Image is being processed and will be sent shortly.",
    }


# ----------------------------
# Health Check Endpoint
# ----------------------------
@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify that the server is running.

    Returns:
        dict: Health status.
    """
    return {"status": "healthy"}


# ----------------------------
# Run the Server
# ----------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
