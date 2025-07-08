# Scraper Functionality Enhancement Plan

This document outlines a detailed, phased approach to upgrading the project's scraping functionality. The goal is to make the scraping framework more robust, performant, efficient, and maintainable by integrating modern libraries and best practices.

---

## Phase 1: Core Extraction & Data Quality Improvement

**Objective:** Immediately improve the quality and reliability of the data being extracted by using specialized libraries for content and metadata extraction.

### Task 1.1: Integrate `trafilatura` for Main Content Extraction
- **Why:** The current method of using CSS selectors (`<main>`, `<article>`) is brittle. `trafilatura` is a state-of-the-art library specifically designed to remove boilerplate (ads, navigation, footers) and extract the core article text from a webpage.
- **Action Steps:**
    1.  Add `trafilatura` to the `requirements.txt` file.
    2.  In `agents/scraper/scraper_agent.py`, modify the `scrape_html_content` method.
    3.  Inside the `try` block, after successfully fetching the `html_content`, replace the existing `BeautifulSoup` parsing logic for content extraction.
    4.  Use the following code to extract the main content:
        ```python
        import trafilatura
        # ... inside scrape_html_content ...
        main_content = trafilatura.extract(html_content, include_comments=False, include_tables=True)
        # The 'text' variable should now be assigned main_content
        text = main_content if main_content else ''
        ```
    5.  The rest of the function (metadata extraction with BeautifulSoup) can remain for now, but the primary text content should come from `trafilatura`.

### Task 1.2: Integrate `extruct` for Structured Metadata Extraction
- **Why:** Many websites embed high-quality, structured metadata using formats like JSON-LD (Schema.org). `extruct` can parse these formats, providing much more reliable metadata (author, title, date) than manual parsing.
- **Action Steps:**
    1.  Add `extruct` to the `requirements.txt` file.
    2.  In `agents/scraper/scraper_agent.py`, within the `scrape_html_content` method, after getting the `html_content`:
    3.  Use `extruct` to pull out structured data:
        ```python
        import extruct
        # ... inside scrape_html_content, you'll need the url passed in ...
        structured_data = extruct.extract(
            html_content,
            base_url=url, # You will need to pass the URL to this function
            syntaxes=['json-ld', 'microdata'],
            uniform=True
        )
        ```
    4.  Modify the metadata population logic. Before falling back to parsing the title tag or other meta tags, check the `structured_data` dictionary for richer information. Prioritize `json-ld` data if available.
        -   Example: `metadata['title'] = structured_data.get('json-ld', [{}])[0].get('headline', soup.title.string)`

### Task 1.3: Upgrade Language Detection
- **Why:** The current character-pattern-based language detection is basic. A dedicated library offers much higher accuracy.
- **Action Steps:**
    1.  Add `pycld3` to the `requirements.txt` file.
    2.  In `agents/scraper/scraper_agent.py`, find the `detect_language` method.
    3.  Replace the entire implementation with a call to `pycld3`.
        ```python
        import pycld3 as cld3
        # ... inside the WebScraper class ...
        def detect_language(self, text: str) -> str:
            """Detect language of text content using pycld3."""
            if not text or not text.strip():
                return 'unknown'
            try:
                # The model is most accurate for texts of 100+ characters.
                prediction = cld3.detect(text)
                return prediction.language
            except Exception:
                return 'unknown'
        ```

---

## Phase 2: Robustness & Anti-Blocking

**Objective:** Make the scraper significantly more difficult for websites to detect and block.

### Task 2.1: Implement Proxy and User-Agent Rotation
- **Why:** Using a single IP and User-Agent is a clear sign of a bot. Rotation is essential for any serious scraping.
- **Action Steps:**
    1.  In `agents/scraper/scraper_config.py`, modify the `SCRAPER_CONFIG` dictionary:
        -   Rename `'user_agent'` to `'user_agents'` and make it a list of at least 5 real-world browser user agents.
        -   Add a new key `'proxies'` which is a list of proxy URLs (e.g., `['http://user:pass@host:port']`). Initially, this can be an empty list.
    2.  In `agents/scraper/scraper_agent.py`, modify the `create_session` method:
        -   Randomly select a user agent from the config list for the `headers`.
    3.  Modify the `scrape_html_content` (and other request-making methods):
        -   Before making a request with `self.session.get()`, randomly select a proxy from the config list.
        -   Pass the selected proxy to the request: `async with self.session.get(url, proxy=random_proxy) as response:`. Handle the case where the proxy list is empty.

### Task 2.2: Integrate `selenium-stealth`
- **Why:** When using Selenium for JavaScript-heavy sites, standard Selenium is easily detectable. `selenium-stealth` patches it to appear like a normal browser.
- **Action Steps:**
    1.  Add `selenium-stealth` to `requirements.txt`.
    2.  In `agents/scraper/scraper_agent.py`, update the `setup_webdriver` method.
    3.  Import the `stealth` function: `from selenium_stealth import stealth`.
    4.  After initializing `chrome_options` and before creating the `webdriver.Chrome` instance, apply the stealth patches:
        ```python
        # ... inside setup_webdriver ...
        driver = webdriver.Chrome(options=chrome_options)
        
        stealth(driver,
              languages=["en-US", "en"],
              vendor="Google Inc.",
              platform="Win32",
              webgl_vendor="Intel Inc.",
              renderer="Intel Iris OpenGL Engine",
              fix_hairline=True,
              )
        self.driver = driver
        ```

---

## Phase 3: Architectural Improvements

**Objective:** Refactor the codebase for better long-term maintainability and scalability.

### Task 3.1: Unify `scraper_agent.py` and `high_performance_scraper.py`
- **Why:** The two files have overlapping functionality. A single, configurable scraper class reduces code duplication and simplifies the architecture.
- **Action Steps:**
    1.  The primary class should be `WebScraper` in `agents/scraper/scraper_agent.py`.
    2.  Modify its `__init__` to accept a `profile` argument (e.g., `profile: str = 'comprehensive'`).
    3.  Merge the high-performance logic into `WebScraper`:
        -   Integrate the `FastHTMLParser` from `high_performance_scraper.py` into `scraper_agent.py`.
        -   Integrate the `HighPerformanceCache` (Redis + memory cache) into `scraper_agent.py`.
    4.  In the `scrape_url` method (or equivalent), use conditional logic based on the `profile`:
        -   If `profile == 'fast'`: Use `selectolax` for parsing, enable the cache, and do not run Selenium.
        -   If `profile == 'comprehensive'`: Use `trafilatura` and `extruct`, run Selenium if needed, perform OCR, etc.
    5.  Once all functionality is merged, deprecate and delete `high_performance_scraper.py`.

### Task 3.2: Implement a Plugin Architecture for Site-Specific Parsers
- **Why:** A generic scraper can only go so far. For key target sites, custom logic is needed. A plugin system keeps this organized.
- **Action Steps:**
    1.  Create a new directory: `agents/scraper/parsers/`.
    2.  In this directory, create an `__init__.py` and a base parser template.
    3.  Create example parser files named after the domain, e.g., `plato_stanford_edu.py`.
    4.  Each parser file should contain a `parse(html: str) -> dict:` function that returns a dictionary of extracted data.
    5.  In the main `WebScraper` class, before performing generic parsing, check if a custom parser exists for the target domain.
        -   Dynamically import the module from the `parsers` directory.
        -   If it exists, call its `parse` function.
        -   If not, fall back to the default `trafilatura`/`extruct` logic.
