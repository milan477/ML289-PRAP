from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import requests
import os
import time

class PoliceRecordsDownloader:
    def __init__(self, output_dir="police_records", headless=True):
        self.base_url = "https://policerecords.kqed.org"
        self.output_dir = output_dir
        self.total_doc_counter = self._get_highest_id()

        print('starting id at',self.total_doc_counter)
        # Setup Chrome options
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

        # Initialize the driver
        print("Initializing Chrome driver...")
        self.driver = webdriver.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, 20)

        # Session for downloading files
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def _get_highest_id(self):
        """Find the highest ID number from existing files"""
        if not os.path.exists(self.output_dir):
            return 0

        max_id = 0

        # Walk through all subdirectories
        for root, dirs, files in os.walk(self.output_dir):
            for filename in files:
                # Look for pattern _idXXXX.pdf
                if '_id' in filename and filename.endswith('.pdf'):
                    try:
                        # Extract the ID number
                        id_part = filename.split('_id')[1].replace('.pdf', '')
                        file_id = int(id_part)
                        max_id = max(max_id, file_id)
                    except (ValueError, IndexError):
                        continue

        return max_id +1

    def get_case_links_from_page(self, page_num):
        """Get all case links from a specific page number"""
        url = f"{self.base_url}/?page={page_num}"
        print(f"\nFetching page {page_num}: {url}")

        try:
            self.driver.get(url)

            # Wait for the page to load - wait for case links to appear
            print("Waiting for content to load...")
            time.sleep(3)  # Give extra time for JavaScript to render

            # Try to wait for specific elements
            try:
                self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "a")))
            except:
                print("Timeout waiting for links, but continuing anyway...")

            # Find all links with href starting with /cases/
            case_links = []
            all_links = self.driver.find_elements(By.TAG_NAME, "a")

            print(f"Total links found on page: {len(all_links)}")

            for link in all_links:
                try:
                    href = link.get_attribute('href')
                    if href and '/cases/' in href:
                        if href not in case_links:  # Avoid duplicates
                            case_links.append(href)
                            # Debug: show what we're finding
                            case_id = href.split('/')[-1]
                            print(f"  Found case: {case_id}")
                except:
                    continue

            print(f"Found {len(case_links)} unique cases on page {page_num}")
            return case_links

        except Exception as e:
            print(f"Error fetching page {page_num}: {e}")
            return []

    def get_documents_from_case(self, case_url):
        """Get all document URLs from a specific case page"""
        print(f"\nFetching case: {case_url}")

        try:
            self.driver.get(case_url)

            # Wait for documents to load
            print("Waiting for documents to load...")
            time.sleep(3)

            # Find all links to cleanpdfs.blob.core.windows.net
            doc_links = []
            all_links = self.driver.find_elements(By.TAG_NAME, "a")

            for link in all_links:
                try:
                    href = link.get_attribute('href')
                    if href and 'cleanpdfs.blob.core.windows.net' in href:
                        if href not in doc_links:  # Avoid duplicates
                            doc_links.append(href)
                            filename = href.split('/')[-1][:20]  # Show first 20 chars
                            print(f"  Found document: {filename}...")
                except:
                    continue

            print(f"Found {len(doc_links)} documents in case")
            return doc_links

        except Exception as e:
            print(f"Error fetching case {case_url}: {e}")
            return []

    def download_document(self, doc_url, case_id, page_num, case_num, doc_num):
        """Download a single document (only if 10MB or less)"""
        try:
            case_dir = self.output_dir
            if not os.path.exists(case_dir):
                os.makedirs(case_dir)

            # Create descriptive filename with total ID

            root = f"page{page_num:02d}_case{case_num:02d}_document{doc_num:02d}"

            # Look for existing files with this root in the case directory
            existing_files = [f for f in os.listdir(case_dir) if f.startswith(root)]

            if existing_files:
                print(f"  Already exists: {existing_files[0]}")
                return True

            filename = f"page{page_num:02d}_case{case_num:02d}_document{doc_num:02d}_id{self.total_doc_counter:04d}.pdf"

            # Create case-specific subdirectory
            filepath = os.path.join(case_dir, filename)
            # Skip if already downloaded
            if os.path.exists(filepath):
                print(f"  Already exists: {filename}")
                return True

            # Increment total counter
            self.total_doc_counter += 1

            # First, check the file size without downloading the whole file
            print(f"  Checking size: {filename}")
            response = self.session.head(doc_url, timeout=30)

            # Get file size from headers (in bytes)
            file_size_bytes = int(response.headers.get('content-length', 0))
            file_size_mb = file_size_bytes / (1024 * 1024)

            # Skip if larger than 10MB
            if file_size_mb > 10:
                print(f"  ⊘ Skipping (too large): {filename} ({file_size_mb:.1f} MB)")
                return False

            print(f"  Downloading: {filename} ({file_size_mb:.2f} MB)")
            response = self.session.get(doc_url, timeout=60, stream=True)
            response.raise_for_status()

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"  ✓ Saved: {filename} ({file_size_mb:.2f} MB)")
            return True

        except Exception as e:
            print(f"  ✗ Error downloading {doc_url}: {e}")
            return False

    def download_page(self, page_num):
        """Download all documents from a specific page"""
        print(f"\n{'='*60}")
        print(f"Processing Page {page_num}")
        print(f"{'='*60}")

        # Get all case links from the page
        case_links = self.get_case_links_from_page(page_num)

        if not case_links:
            print(f"\n⚠ No cases found on page {page_num}")
            print("This could mean:")
            print("  - The page doesn't exist")
            print("  - JavaScript didn't load properly")
            print("  - The page structure has changed")
            return 0

        total_downloaded = 0

        # Process each case
        for case_num, case_url in enumerate(case_links, 1):
            print(f"\n{'─'*60}")
            print(f"[Case {case_num}/{len(case_links)}]")

            # Extract case ID from URL
            case_id = case_url.split('/')[-1]

            # Get document links from case
            doc_links = self.get_documents_from_case(case_url)

            if not doc_links:
                print(f"  ⚠ No documents found in case {case_id}")
                continue

            # Download each document
            for doc_num, doc_url in enumerate(doc_links, 1):
                if self.download_document(doc_url, case_id, page_num, case_num, doc_num):
                    total_downloaded += 1
                time.sleep(0.5)  # Be polite to the server

            time.sleep(1)  # Delay between cases

        return total_downloaded

    def download_multiple_pages(self, start_page, end_page):
        """Download documents from multiple pages"""
        print(f"\nStarting download from pages {start_page} to {end_page}")
        print(f"Output directory: {os.path.abspath(self.output_dir)}\n")

        total_docs = 0

        try:
            for page_num in range(start_page, end_page + 1):
                docs_downloaded = self.download_page(page_num)
                total_docs += docs_downloaded

                print(f"\n{'='*60}")
                print(f"Page {page_num} complete. Downloaded {docs_downloaded} documents.")
                print(f"{'='*60}")

                if page_num < end_page:
                    time.sleep(2)  # Delay between pages

        finally:
            self.driver.quit()
            print("\nBrowser closed.")

        print(f"\n{'='*60}")
        print(f"COMPLETE: Downloaded {total_docs} total documents")
        print(f"Location: {os.path.abspath(self.output_dir)}")
        print(f"{'='*60}")

    def __del__(self):
        """Cleanup"""
        try:
            self.driver.quit()
        except:
            pass


def scrape_database(start_page = 1, end_page = 1):
    print(os.getcwd())
    print("KQED Police Records Downloader")
    print("="*60)

    # Configuration
    output_dir = "data/policerecords/pdfs"
    os.makedirs(output_dir, exist_ok=True)


    print(f"\nDownloading pages {start_page} to {end_page}")
    print(f"Output directory: {output_dir}")

    # Create downloader and start (headless mode)
    downloader = PoliceRecordsDownloader(output_dir, headless=True)
    downloader.download_multiple_pages(start_page, end_page)


if __name__ == "__main__":
    scrape_database()