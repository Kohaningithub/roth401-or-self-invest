import requests
import time
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('keep_alive.log'),
        logging.StreamHandler()
    ]
)

def ping_website(url, interval_minutes=10):
    """
    Continuously ping a website at specified intervals to keep it alive.
    
    Args:
        url (str): The URL of the website to ping
        interval_minutes (int): Time interval between pings in minutes
    """
    while True:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                logging.info(f"Successfully pinged {url} - Status: {response.status_code}")
            else:
                logging.warning(f"Ping failed - Status: {response.status_code}")
        except Exception as e:
            logging.error(f"Error pinging website: {str(e)}")
        
        # Wait for the specified interval
        time.sleep(interval_minutes * 60)

if __name__ == "__main__":
    # Replace this URL with your Streamlit app URL
    STREAMLIT_URL = "https://roth401-or-self-invest.streamlit.app/?user_id=9be41e85-1720-43d4-bae1-ce1d3522fc24"
    PING_INTERVAL = 1440  # minutes
    
    logging.info(f"Starting keep-alive script for {STREAMLIT_URL}")
    logging.info(f"Ping interval: {PING_INTERVAL} minutes")
    
    try:
        ping_website(STREAMLIT_URL, PING_INTERVAL)
    except KeyboardInterrupt:
        logging.info("Script stopped by user")
    except Exception as e:
        logging.error(f"Script crashed: {str(e)}") 