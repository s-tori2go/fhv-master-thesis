import time
import os
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# Configure WebDriver
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode (no browser window)
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")

# Set up ChromeDriver with Service
driver_service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=driver_service, options=chrome_options)


# Create a function to ensure we get high-resolution images
def get_high_res_url(img_url):
    """Convert low-res Pinterest image URL to high-res format."""
    if "pinimg.com" in img_url:
        return img_url.replace("/60x60/", "/originals/").replace("/236x/", "/originals/")
    return img_url


# Main function to scrape and save images
def get_images_from_pinterest(query, num_images=10):
    search_url = f"https://www.pinterest.com/search/pins/?q={query}"
    driver.get(search_url)
    time.sleep(5)  # Let the page load

    # Scroll down multiple times to load more images
    # body = driver.find_element(By.TAG_NAME, "body")
    # for _ in range(3):  # Scroll 3 times
        # body.send_keys(Keys.END)
        # time.sleep(3)  # Increase sleep time to load more images

    # Get page source after scrolling
    soup = BeautifulSoup(driver.page_source, "html.parser")

    # Extract image URLs from <img> tags
    # Try to extract images using the outfits-src attribute
    image_elements = soup.find_all("img", {"src": True})

    # If no images found, try looking for srcset as a fallback
    if not image_elements:
        image_elements = soup.find_all("img", {"src": True})

    image_urls = [get_high_res_url(img.get("src")) for img in image_elements]

    # Extract full image links from <a> tags
    for a_tag in soup.find_all("a", href=True):
        if "/pin/" in a_tag["href"]:
            pin_url = "https://www.pinterest.com" + a_tag["href"]
            driver.get(pin_url)
            time.sleep(2)
            pin_soup = BeautifulSoup(driver.page_source, "html.parser")
            pin_img = pin_soup.find("img", {"src": True})
            if pin_img:
                high_res_url = get_high_res_url(pin_img.get("src"))
                image_urls.append(high_res_url)

    # Remove duplicates and limit to num_images
    image_urls = list(set(image_urls))[:num_images]

    # Ensure "outfits" directory exists
    os.makedirs("data/outfits", exist_ok=True)

    # Download and save images in "outfits" directory
    downloaded_images = []
    for idx, img_url in enumerate(image_urls):
        image_data = requests.get(img_url).content
        image_name = os.path.join("data/outfits", f"{query}_{idx + 1}.jpg")

        # Save the image
        with open(image_name, "wb") as f:
            f.write(image_data)
            downloaded_images.append(image_name)

        print(f"Downloaded {image_name}")

    return downloaded_images


# Example usage
if __name__ == "__main__":
    search_query = "casual outfit woman"
    images = get_images_from_pinterest(search_query, num_images=15)
    print(f"Downloaded images: {images}")

    driver.quit()