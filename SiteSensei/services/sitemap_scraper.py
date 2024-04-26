from playwright.sync_api import sync_playwright

def scrape_sitemap(url):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        loc_elements = page.query_selector_all("loc")
        urls = [element.inner_text() for element in loc_elements]
        browser.close()
    return ','.join(urls)
