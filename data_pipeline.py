import aiohttp
import asyncio
import aiofiles
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from config import FEDERAL_REGISTRY_API, PIPELINE_CONFIG
from database import db_manager

class DataDownloader:
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.base_url = FEDERAL_REGISTRY_API['base_url']
        self.documents_endpoint = FEDERAL_REGISTRY_API['documents_endpoint']

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30, connect=10)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            self.session = None

    async def fetch_documents(self, start_date: str, end_date: str, page: int = 1) -> Dict:
        """Fetch documents from Federal Registry API"""
        if not self.session:
            raise RuntimeError("DataDownloader must be used as async context manager")

        params = {
            'conditions[publication_date][gte]': start_date,
            'conditions[publication_date][lte]': end_date,
            'per_page': 100,
            'page': page,
            'fields[]': ['title', 'abstract', 'document_number', 'publication_date', 
                        'type', 'agencies', 'html_url', 'pdf_url', 'raw_text_url']
        }

        url = f"{self.base_url}{self.documents_endpoint}"
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"Error fetching data: HTTP {response.status}")
                    return {}
        except asyncio.TimeoutError:
            print(f"Timeout error for page {page}")
            return {}
        except Exception as e:
            print(f"Error in API request: {e}")
            return {}

    async def fetch_raw_text(self, raw_text_url: str) -> str:
        """Fetch raw text content from document URL"""
        if not self.session:
            raise RuntimeError("DataDownloader must be used as async context manager")
        
        try:
            if not raw_text_url:
                return ""
            
            async with self.session.get(raw_text_url) as response:
                if response.status == 200:
                    return await response.text()
                return ""
        except asyncio.TimeoutError:
            print(f"Timeout fetching raw text from: {raw_text_url}")
            return ""
        except Exception as e:
            print(f"Error fetching raw text: {e}")
            return ""

class DataProcessor:
    def __init__(self):
        pass
    
    @staticmethod
    def clean_document(document: Dict) -> Dict:
        """Clean and process document data"""
        # Extract agency names from the agencies array
        agencies = document.get('agencies', [])
        agency_names = []
        
        if agencies:
            for agency in agencies:
                if isinstance(agency, dict):
                    agency_names.append(agency.get('name', ''))
                elif isinstance(agency, str):
                    agency_names.append(agency)
        
        agency_str = ', '.join(filter(None, agency_names))

        processed_doc = {
            'id': str(document.get('document_number', '')),
            'title': document.get('title', '').strip(),
            'abstract': document.get('abstract', '').strip() if document.get('abstract') else '',
            'document_number': document.get('document_number', ''),
            'publication_date': document.get('publication_date', ''),
            'type': document.get('type', ''),
            'agency': agency_str,
            'html_url': document.get('html_url', ''),
            'pdf_url': document.get('pdf_url', ''),
            'raw_text': document.get('raw_text', '')
        }
        
        return processed_doc

    @staticmethod
    async def save_raw_data(data: Dict, filename: str):
        """Save raw data to file"""
        try:
            os.makedirs(PIPELINE_CONFIG['raw_data_dir'], exist_ok=True)
            filepath = os.path.join(PIPELINE_CONFIG['raw_data_dir'], filename)
            
            async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Error saving raw data to {filename}: {e}")

    @staticmethod
    async def save_processed_data(data: List[Dict], filename: str):
        """Save processed data to file"""
        try:
            os.makedirs(PIPELINE_CONFIG['processed_data_dir'], exist_ok=True)
            filepath = os.path.join(PIPELINE_CONFIG['processed_data_dir'], filename)
            
            async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Error saving processed data to {filename}: {e}")

class DataPipeline:
    def __init__(self):
        self.processor = DataProcessor()

    async def run_pipeline(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> bool:
        """Run the complete data pipeline"""
        # Default to last 7 days if no dates provided
        if not start_date or not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

        print(f"Starting pipeline for dates: {start_date} to {end_date}")
        
        # Ensure database is initialized
        try:
            await db_manager.initialize_database()
            print("✅ Database initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize database: {e}")
            return False

        # Log pipeline start
        log_id = await db_manager.log_pipeline_run(end_date, 'running')
        total_processed = 0

        try:
            # Use the DataDownloader as an async context manager
            async with DataDownloader() as downloader:
                all_documents = []
                page = 1

                while True:
                    print(f"Fetching page {page}...")
                    data = await downloader.fetch_documents(start_date, end_date, page)
                    
                    if not data or 'results' not in data:
                        print(f"No more data found at page {page}")
                        break

                    documents = data.get('results', [])
                    if not documents:
                        print(f"No documents found at page {page}")
                        break

                    # Save raw data
                    raw_filename = f"federal_docs_raw_{end_date}_page_{page}.json"
                    await self.processor.save_raw_data(data, raw_filename)

                    # Process documents
                    processed_docs = []
                    for i, doc in enumerate(documents):
                        try:
                            # Fetch raw text if available
                            if doc.get('raw_text_url'):
                                print(f"Fetching raw text for doc {i+1}/{len(documents)} on page {page}")
                                raw_text = await downloader.fetch_raw_text(doc['raw_text_url'])
                                doc['raw_text'] = raw_text

                            processed_doc = self.processor.clean_document(doc)
                            processed_docs.append(processed_doc)

                            # Insert into database
                            success = await db_manager.insert_document(processed_doc)
                            if success:
                                total_processed += 1
                            else:
                                print(f"Failed to insert document: {processed_doc.get('document_number', 'unknown')}")
                                
                        except Exception as e:
                            print(f"Error processing document {i}: {e}")
                            continue

                    all_documents.extend(processed_docs)

                    # Save processed data
                    processed_filename = f"federal_docs_processed_{end_date}_page_{page}.json"
                    await self.processor.save_processed_data(processed_docs, processed_filename)

                    print(f"Processed {len(processed_docs)} documents from page {page} (Total processed: {total_processed})")

                    # Check if we have more pages
                    if not data.get('next_page_url'):
                        print("No more pages available")
                        break

                    page += 1
                    
                    # Add small delay to be respectful to the API
                    await asyncio.sleep(1)
                    
                    # Safety break for testing
                    if page > 10:  # Limit to 10 pages for testing
                        print("Reached page limit for testing")
                        break

                # Update pipeline log with success
                await db_manager.update_pipeline_log(log_id, 'completed', total_processed)
                print(f"Pipeline completed successfully. Processed {total_processed} documents.")
                
                # Verify database has data
                doc_count = await db_manager.get_document_count()
                print(f"Total documents in database: {doc_count}")
                
                return True

        except Exception as e:
            error_msg = str(e)
            print(f"Pipeline failed: {error_msg}")
            try:
                await db_manager.update_pipeline_log(log_id, 'failed', total_processed, error_msg)
            except Exception as log_error:
                print(f"Error updating pipeline log: {log_error}")
            return False

async def main():
    """Main function to run the pipeline"""
    try:
        # Initialize database
        await db_manager.initialize_database()
        
        # Run pipeline
        pipeline = DataPipeline()
        success = await pipeline.run_pipeline()
        
        if success:
            # Print some stats
            doc_count = await db_manager.get_document_count()
            agency_count = await db_manager.get_agency_count()
            latest_date = await db_manager.get_latest_publication_date()
            
            print(f"\n📊 Pipeline Summary:")
            print(f"📄 Total Documents: {doc_count}")
            print(f"🏛️ Unique Agencies: {agency_count}")
            print(f"📅 Latest Publication: {latest_date}")
        
        return success
    except Exception as e:
        print(f"Error in main: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(main())
    print(f"Pipeline execution {'succeeded' if result else 'failed'}")