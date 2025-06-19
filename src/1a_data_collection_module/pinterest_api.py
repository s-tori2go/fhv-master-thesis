import os
import requests
import config

# API credentials

API_URL = f"https://api.pinterest.com/v5/boards/{config.BOARD_ID}/pins"
HEADERS = {
    "Authorization": f"Bearer {config.BEARER_TOKEN}",
    "Content-Type": "application/json",
    "Accept": "application/json"
}

# Output folder
OUTPUT_FOLDER = "./data/outfits/"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def fetch_pins():
    pins = []
    bookmark = None
    while True:
        params = {"bookmark": bookmark} if bookmark else {}
        response = requests.get(API_URL, headers=HEADERS, params=params)
        if response.status_code == 200:
            data = response.json()
            pins.extend(data.get("items", []))
            bookmark = data.get("bookmark")
            if not bookmark:
                break  # No more pages
        else:
            print(f"Error: {response.status_code}, {response.text}")
            break
    return pins

def download_image(image_url, filename):
    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print(f"Saved: {file_path}")
    else:
        print(f"Failed to download {image_url}")


def main():
    pins_data = fetch_pins()
    if not pins_data:
        return

    for pin in pins_data:
        image_url = pin.get("media", {}).get("images", {}).get("1200x", {}).get("url")
        if image_url:
            filename = f"{pin['id']}.jpg"
            download_image(image_url, filename)
        else:
            print("No 1200x image found for pin", pin.get("id"))


if __name__ == "__main__":
    main()
