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
        
        # Debug: Print the actual API request details
        print(f"🔍 API Request Debug:")
        print(f"   URL: {url}")
        print(f"   Date Range: {start_date} to {end_date}")
        print(f"   Page: {page}")
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    # Debug: Print response summary
                    results_count = len(data.get('results', []))
                    total_pages = data.get('total_pages', 'unknown')
                    current_page = data.get('current_page', page)
                    print(f"   Response: {results_count} documents, page {current_page}/{total_pages}")
                    return data
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

        print(f"🚀 Starting pipeline for date range: {start_date} to {end_date}")
        
        # Validate date range
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            days_diff = (end_dt - start_dt).days
            print(f"📅 Processing {days_diff + 1} days of data")
        except ValueError as e:
            print(f"❌ Invalid date format: {e}")
            return False
        
        # Ensure database is initialized
        try:
            await db_manager.initialize_database()
            print("✅ Database initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize database: {e}")
            return False

        # Log pipeline start with proper date range info
        log_date = f"{start_date}_to_{end_date}"
        log_id = await db_manager.log_pipeline_run(log_date, 'running')
        total_processed = 0
        total_new_docs = 0
        total_existing_docs = 0

        try:
            # Use the DataDownloader as an async context manager
            async with DataDownloader() as downloader:
                all_documents = []
                page = 1
                max_pages = 50  # Reasonable limit for safety

                print(f"🔄 Starting to fetch documents...")

                while page <= max_pages:
                    print(f"\n📄 Fetching page {page}...")
                    data = await downloader.fetch_documents(start_date, end_date, page)
                    
                    if not data:
                        print(f"❌ No data returned from API for page {page}")
                        break
                    
                    if 'results' not in data:
                        print(f"❌ No 'results' key in API response for page {page}")
                        print(f"Available keys: {list(data.keys())}")
                        break

                    documents = data.get('results', [])
                    if not documents:
                        print(f"✅ No more documents found at page {page} - end of data")
                        break

                    print(f"📥 Found {len(documents)} documents on page {page}")

                    # Save raw data with better filename
                    raw_filename = f"federal_docs_raw_{start_date}_to_{end_date}_page_{page}.json"
                    await self.processor.save_raw_data(data, raw_filename)

                    # Process documents
                    processed_docs = []
                    page_new_docs = 0
                    page_existing_docs = 0
                    
                    for i, doc in enumerate(documents):
                        try:
                            # Show progress for raw text fetching
                            if doc.get('raw_text_url'):
                                if i % 10 == 0:  # Show progress every 10 docs
                                    print(f"   📝 Fetching raw text for doc {i+1}/{len(documents)} on page {page}")
                                raw_text = await downloader.fetch_raw_text(doc['raw_text_url'])
                                doc['raw_text'] = raw_text

                            processed_doc = self.processor.clean_document(doc)
                            processed_docs.append(processed_doc)

                            # Insert into database (only if new)
                            success, status = await db_manager.insert_document_if_new(processed_doc)
                            
                            if success:
                                if status == "inserted":
                                    total_processed += 1
                                    page_new_docs += 1
                                    total_new_docs += 1
                                    if page_new_docs <= 5:  # Only show first 5 per page to avoid spam
                                        print(f"   ✅ New: {processed_doc.get('document_number', 'unknown')} ({processed_doc.get('publication_date', 'no date')})")
                                elif status == "exists":
                                    page_existing_docs += 1
                                    total_existing_docs += 1
                                    if page_existing_docs <= 3:  # Show fewer existing docs
                                        print(f"   📋 Exists: {processed_doc.get('document_number', 'unknown')}")
                            else:
                                print(f"   ❌ Failed to insert: {processed_doc.get('document_number', 'unknown')}")
                                
                        except Exception as e:
                            print(f"   ❌ Error processing document {i}: {e}")
                            continue

                    all_documents.extend(processed_docs)

                    # Enhanced page summary
                    print(f"\n📊 Page {page} Summary:")
                    print(f"   • Total documents on page: {len(processed_docs)}")
                    print(f"   • New documents: {page_new_docs}")
                    print(f"   • Existing documents: {page_existing_docs}")
                    print(f"   • Running totals - New: {total_new_docs}, Existing: {total_existing_docs}")

                    # Save processed data with better filename
                    processed_filename = f"federal_docs_processed_{start_date}_to_{end_date}_page_{page}.json"
                    await self.processor.save_processed_data(processed_docs, processed_filename)

                    # Check if we have more pages
                    has_next_page = data.get('next_page_url') is not None
                    total_pages = data.get('total_pages', 'unknown')
                    current_page = data.get('current_page', page)
                    
                    print(f"   📋 Page info: {current_page}/{total_pages}, Has next: {has_next_page}")
                    
                    if not has_next_page:
                        print("✅ Reached end of available pages")
                        break

                    page += 1
                    
                    # Add small delay to be respectful to the API
                    await asyncio.sleep(1)

                # Final summary
                print(f"\n🎉 Pipeline Processing Complete!")
                print(f"   📄 Total pages processed: {page}")
                print(f"   📊 Total new documents: {total_new_docs}")
                print(f"   📋 Total existing documents: {total_existing_docs}")
                print(f"   📈 Total documents processed: {total_processed}")

                # Update pipeline log with success
                await db_manager.update_pipeline_log(log_id, 'completed', total_processed)
                
                # Verify database has data
                doc_count = await db_manager.get_document_count()
                print(f"   🗄️ Total documents in database: {doc_count}")
                
                return True

        except Exception as e:
            error_msg = str(e)
            print(f"❌ Pipeline failed: {error_msg}")
            try:
                await db_manager.update_pipeline_log(log_id, 'failed', total_processed, error_msg)
            except Exception as log_error:
                print(f"❌ Error updating pipeline log: {log_error}")
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