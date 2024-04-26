from playwright.sync_api import sync_playwright
import tldextract

def scrape_sitemap(url):
    domain = get_domain_name(url)  # Automatically determine the domain
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        loc_elements = page.query_selector_all("loc")
        urls = [element.inner_text() for element in loc_elements]

        if not urls:
            links = page.query_selector_all(f"a[href*='{domain}']")
            urls = [link.get_attribute('href') for link in links if domain in link.get_attribute('href')]

        browser.close()
    return ','.join(urls)

def get_domain_name(url):
    extracted = tldextract.extract(url)
    return "{}.{}".format(extracted.domain, extracted.suffix)
