import aiosqlite
import asyncio
import os
from typing import List, Dict, Optional
import json
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        # Ensure the database is in a writable location for HF Spaces
        self.db_path = os.path.join("/tmp", "federal_registry.db")
        self._initialized = False
        self._lock = asyncio.Lock()
    
    async def _ensure_initialized(self):
        """Ensure the database is initialized before operations"""
        if not self._initialized:
            await self.initialize_database()
    
    async def _ensure_db_directory(self):
        """Ensure the database directory exists and is writable"""
        db_dir = os.path.dirname(self.db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        
        # Test write permissions
        test_file = os.path.join(db_dir, "test_write.tmp")
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            # Fallback to current directory if /tmp is not writable
            self.db_path = "federal_registry.db"
            logger.warning(f"Using current directory for database: {e}")
    
    async def initialize_database(self):
        """Initialize SQLite database and create tables"""
        async with self._lock:
            if self._initialized:
                return
            
            try:
                # Ensure directory exists and is writable
                await self._ensure_db_directory()
                
                # Create tables
                await self.create_tables()
                self._initialized = True
                logger.info(f"✅ SQLite database initialized successfully at {self.db_path}")
            except Exception as e:
                logger.error(f"❌ Error initializing database: {e}")
                raise
    
    async def create_tables(self):
        """Create necessary tables if they don't exist"""
        create_documents_table = """
        CREATE TABLE IF NOT EXISTS federal_documents (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            abstract TEXT,
            document_number TEXT UNIQUE,
            publication_date TEXT,
            type TEXT,
            agency TEXT,
            raw_text TEXT,
            html_url TEXT,
            pdf_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        create_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_publication_date ON federal_documents(publication_date);",
            "CREATE INDEX IF NOT EXISTS idx_agency ON federal_documents(agency);",
            "CREATE INDEX IF NOT EXISTS idx_type ON federal_documents(type);",
            "CREATE INDEX IF NOT EXISTS idx_title ON federal_documents(title);",
            "CREATE INDEX IF NOT EXISTS idx_abstract ON federal_documents(abstract);"
        ]
        
        create_pipeline_logs_table = """
        CREATE TABLE IF NOT EXISTS pipeline_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date TEXT,
            status TEXT CHECK(status IN ('running', 'completed', 'failed')),
            records_processed INTEGER DEFAULT 0,
            start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            end_time TIMESTAMP,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(create_documents_table)
                for index_sql in create_indexes:
                    await db.execute(index_sql)
                await db.execute(create_pipeline_logs_table)
                await db.commit()
                logger.info("✅ Database tables created successfully")
        except Exception as e:
            logger.error(f"❌ Error creating tables: {e}")
            raise
    
    async def insert_document(self, document: Dict) -> bool:
        """Insert or update a federal document"""
        await self._ensure_initialized()
        
        query = """
        INSERT OR REPLACE INTO federal_documents 
        (id, title, abstract, document_number, publication_date, type, agency, raw_text, html_url, pdf_url, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(query, (
                    document.get('id'),
                    document.get('title'),
                    document.get('abstract'),
                    document.get('document_number'),
                    document.get('publication_date'),
                    document.get('type'),
                    document.get('agency'),
                    document.get('raw_text'),
                    document.get('html_url'),
                    document.get('pdf_url')
                ))
                await db.commit()
            return True
        except Exception as e:
            logger.error(f"Error inserting document: {e}")
            return False
    
    async def search_documents(self, query: str, limit: int = 10) -> List[Dict]:
        """Search documents using SQLite LIKE with better search logic"""
        await self._ensure_initialized()
        
        # Split query into words for better matching
        words = query.lower().split()
        
        # Build dynamic WHERE clause for multiple word search
        where_conditions = []
        params = []
        
        for word in words:
            word_pattern = f"%{word}%"
            where_conditions.append("(LOWER(title) LIKE ? OR LOWER(abstract) LIKE ? OR LOWER(raw_text) LIKE ?)")
            params.extend([word_pattern, word_pattern, word_pattern])
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        search_query = f"""
        SELECT id, title, abstract, document_number, publication_date, type, agency, html_url
        FROM federal_documents 
        WHERE {where_clause}
        ORDER BY publication_date DESC
        LIMIT ?
        """
        
        params.append(limit)
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(search_query, params) as cursor:
                    rows = await cursor.fetchall()
                    return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    async def get_documents_by_date_range(self, start_date: str, end_date: str, limit: int = 50) -> List[Dict]:
        """Get documents within a date range"""
        await self._ensure_initialized()
        
        query = """
        SELECT id, title, abstract, document_number, publication_date, type, agency, html_url
        FROM federal_documents 
        WHERE publication_date BETWEEN ? AND ?
        ORDER BY publication_date DESC
        LIMIT ?
        """
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(query, (start_date, end_date, limit)) as cursor:
                    rows = await cursor.fetchall()
                    return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting documents by date range: {e}")
            return []
    
    async def get_documents_by_agency(self, agency: str, limit: int = 20) -> List[Dict]:
        """Get documents by specific agency"""
        await self._ensure_initialized()
        
        query = """
        SELECT id, title, abstract, document_number, publication_date, type, agency, html_url
        FROM federal_documents 
        WHERE LOWER(agency) LIKE LOWER(?)
        ORDER BY publication_date DESC
        LIMIT ?
        """
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(query, (f"%{agency}%", limit)) as cursor:
                    rows = await cursor.fetchall()
                    return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting documents by agency: {e}")
            return []
    
    async def get_recent_documents(self, days: int = 7, limit: int = 20) -> List[Dict]:
        """Get recent documents from last N days"""
        await self._ensure_initialized()
        
        query = """
        SELECT id, title, abstract, document_number, publication_date, type, agency, html_url
        FROM federal_documents 
        WHERE publication_date >= date('now', '-' || ? || ' days')
        ORDER BY publication_date DESC
        LIMIT ?
        """
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(query, (days, limit)) as cursor:
                    rows = await cursor.fetchall()
                    return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting recent documents: {e}")
            return []
    
    async def log_pipeline_run(self, run_date: str, status: str, records_processed: int = 0, 
                              error_message: Optional[str] = None) -> int:
        """Log pipeline execution"""
        await self._ensure_initialized()
        
        query = """
        INSERT INTO pipeline_logs (run_date, status, records_processed, error_message, start_time)
        VALUES (?, ?, ?, ?, datetime('now'))
        """
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(query, (run_date, status, records_processed, error_message))
                await db.commit()
                return cursor.lastrowid or 0
        except Exception as e:
            logger.error(f"Error logging pipeline run: {e}")
            return 0
    
    async def update_pipeline_log(self, log_id: int, status: str, records_processed: int = 0, 
                                 error_message: Optional[str] = None):
        """Update pipeline log with completion status"""
        await self._ensure_initialized()
        
        query = """
        UPDATE pipeline_logs 
        SET status = ?, records_processed = ?, end_time = datetime('now'), error_message = ?
        WHERE id = ?
        """
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(query, (status, records_processed, error_message, log_id))
                await db.commit()
        except Exception as e:
            logger.error(f"Error updating pipeline log: {e}")
    
    async def get_document_count(self) -> int:
        """Get total document count"""
        await self._ensure_initialized()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("SELECT COUNT(*) FROM federal_documents") as cursor:
                    result = await cursor.fetchone()
                    return result[0] if result else 0
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0
    
    async def get_agency_count(self) -> int:
        """Get count of unique agencies"""
        await self._ensure_initialized()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                query = """
                SELECT COUNT(DISTINCT agency) 
                FROM federal_documents 
                WHERE agency IS NOT NULL AND agency != ''
                """
                async with db.execute(query) as cursor:
                    result = await cursor.fetchone()
                    return result[0] if result else 0
        except Exception as e:
            logger.error(f"Error getting agency count: {e}")
            return 0
    
    async def get_latest_documents(self, limit: int = 5) -> List[Dict]:
        """Get latest documents for status display"""
        await self._ensure_initialized()
        
        query = """
        SELECT title, publication_date, agency, type
        FROM federal_documents 
        WHERE publication_date IS NOT NULL
        ORDER BY publication_date DESC
        LIMIT ?
        """
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(query, (limit,)) as cursor:
                    rows = await cursor.fetchall()
                    return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting latest documents: {e}")
            return []
    
    async def get_latest_publication_date(self) -> Optional[str]:
        """Get the most recent publication date"""
        await self._ensure_initialized()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                query = """
                SELECT publication_date 
                FROM federal_documents 
                WHERE publication_date IS NOT NULL
                ORDER BY publication_date DESC 
                LIMIT 1
                """
                async with db.execute(query) as cursor:
                    result = await cursor.fetchone()
                    return result[0] if result else None
        except Exception as e:
            logger.error(f"Error getting latest publication date: {e}")
            return None
    
    async def health_check(self) -> bool:
        """Check if database is accessible"""
        try:
            await self._ensure_initialized()
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    # ========================================
    # DEBUG AND MAINTENANCE METHODS
    # ========================================
    
    async def debug_data_quality(self) -> Dict:
        """Debug data quality issues and return detailed report"""
        await self._ensure_initialized()
        
        report = {
            'total_documents': 0,
            'agency_analysis': {},
            'date_analysis': {},
            'field_analysis': {},
            'sample_records': []
        }
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                # Total document count - FIXED
                async with db.execute("SELECT COUNT(*) FROM federal_documents") as cursor:
                    result = await cursor.fetchone()
                    report['total_documents'] = result[0] if result else 0
                
                # Agency analysis - FIXED
                async with db.execute("SELECT COUNT(*) FROM federal_documents WHERE agency IS NULL") as cursor:
                    result = await cursor.fetchone()
                    report['agency_analysis']['null_count'] = result[0] if result else 0
                
                async with db.execute("SELECT COUNT(*) FROM federal_documents WHERE agency = ''") as cursor:
                    result = await cursor.fetchone()
                    report['agency_analysis']['empty_count'] = result[0] if result else 0
                
                async with db.execute("SELECT COUNT(*) FROM federal_documents WHERE agency IS NOT NULL AND agency != ''") as cursor:
                    result = await cursor.fetchone()
                    report['agency_analysis']['valid_count'] = result[0] if result else 0
                
                async with db.execute("SELECT DISTINCT agency FROM federal_documents WHERE agency IS NOT NULL AND agency != '' LIMIT 10") as cursor:
                    agencies = await cursor.fetchall()
                    report['agency_analysis']['sample_agencies'] = [row[0] for row in agencies] if agencies else []
                
                # Date analysis - FIXED
                async with db.execute("SELECT COUNT(*) FROM federal_documents WHERE publication_date IS NULL") as cursor:
                    result = await cursor.fetchone()
                    report['date_analysis']['null_count'] = result[0] if result else 0
                
                async with db.execute("SELECT COUNT(*) FROM federal_documents WHERE publication_date = ''") as cursor:
                    result = await cursor.fetchone()
                    report['date_analysis']['empty_count'] = result[0] if result else 0
                
                async with db.execute("SELECT COUNT(*) FROM federal_documents WHERE publication_date IS NOT NULL AND publication_date != ''") as cursor:
                    result = await cursor.fetchone()
                    report['date_analysis']['valid_count'] = result[0] if result else 0
                
                async with db.execute("SELECT DISTINCT publication_date FROM federal_documents WHERE publication_date IS NOT NULL AND publication_date != '' ORDER BY publication_date DESC LIMIT 5") as cursor:
                    dates = await cursor.fetchall()
                    report['date_analysis']['sample_dates'] = [row[0] for row in dates] if dates else []
                
                # Field analysis - FIXED
                async with db.execute("SELECT COUNT(*) FROM federal_documents WHERE title IS NULL OR title = ''") as cursor:
                    result = await cursor.fetchone()
                    report['field_analysis']['invalid_titles'] = result[0] if result else 0
                
                async with db.execute("SELECT type, COUNT(*) FROM federal_documents WHERE type IS NOT NULL AND type != '' GROUP BY type") as cursor:
                    types = await cursor.fetchall()
                    report['field_analysis']['document_types'] = {row[0]: row[1] for row in types} if types else {}
                
                # Sample records - FIXED
                async with db.execute("SELECT id, title, agency, publication_date, type FROM federal_documents LIMIT 5") as cursor:
                    samples = await cursor.fetchall()
                    report['sample_records'] = [dict(row) for row in samples] if samples else []
                
        except Exception as e:
            logger.error(f"Debug analysis failed: {e}")
            report['error'] = str(e)
        
        return report
    
    async def fix_data_issues(self) -> Dict:
        """Attempt to fix common data issues"""
        await self._ensure_initialized()
        
        fixes_applied = {
            'empty_strings_fixed': False,
            'agency_count_after': 0,
            'latest_date_after': None,
            'errors': []
        }
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Update empty strings to NULL for proper counting
                cursor1 = await db.execute("UPDATE federal_documents SET agency = NULL WHERE agency = ''")
                cursor2 = await db.execute("UPDATE federal_documents SET publication_date = NULL WHERE publication_date = ''")
                await db.commit()
                
                fixes_applied['empty_strings_fixed'] = True
                
                # Re-run counts to verify fixes - FIXED
                async with db.execute("SELECT COUNT(DISTINCT agency) FROM federal_documents WHERE agency IS NOT NULL") as cursor:
                    result = await cursor.fetchone()
                    fixes_applied['agency_count_after'] = result[0] if result else 0
                
                async with db.execute("SELECT publication_date FROM federal_documents WHERE publication_date IS NOT NULL ORDER BY publication_date DESC LIMIT 1") as cursor:
                    result = await cursor.fetchone()
                    fixes_applied['latest_date_after'] = result[0] if result else None
                
        except Exception as e:
            fixes_applied['errors'].append(str(e))
            logger.error(f"Data fix failed: {e}")
        
        return fixes_applied
    
    async def get_database_stats(self) -> Dict:
        """Get comprehensive database statistics"""
        await self._ensure_initialized()
        
        stats = {
            'document_count': 0,
            'agency_count': 0,
            'latest_publication_date': None,
            'date_range': {'earliest': None, 'latest': None},
            'document_types': {},
            'top_agencies': {},
            'health_status': False
        }
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                # Basic counts
                stats['document_count'] = await self.get_document_count()
                stats['agency_count'] = await self.get_agency_count()
                stats['latest_publication_date'] = await self.get_latest_publication_date()
                stats['health_status'] = await self.health_check()
                
                # Date range - FIXED
                async with db.execute("SELECT MIN(publication_date), MAX(publication_date) FROM federal_documents WHERE publication_date IS NOT NULL") as cursor:
                    result = await cursor.fetchone()
                    if result and result[0]:
                        stats['date_range'] = {'earliest': result[0], 'latest': result[1]}
                
                # Document types - FIXED
                async with db.execute("SELECT type, COUNT(*) as count FROM federal_documents WHERE type IS NOT NULL GROUP BY type ORDER BY count DESC") as cursor:
                    types = await cursor.fetchall()
                    stats['document_types'] = {row['type']: row['count'] for row in types} if types else {}
                
                # Top agencies - FIXED
                async with db.execute("SELECT agency, COUNT(*) as count FROM federal_documents WHERE agency IS NOT NULL GROUP BY agency ORDER BY count DESC LIMIT 10") as cursor:
                    agencies = await cursor.fetchall()
                    stats['top_agencies'] = {row['agency']: row['count'] for row in agencies} if agencies else {}
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            stats['error'] = str(e)
        
        return stats
    
    async def cleanup_database(self) -> Dict:
        """Clean up database by removing duplicates and invalid records"""
        await self._ensure_initialized()
        
        cleanup_results = {
            'duplicates_removed': 0,
            'invalid_records_removed': 0,
            'empty_fields_cleaned': 0,
            'errors': []
        }
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Remove records with NULL or empty titles (invalid)
                cursor = await db.execute("DELETE FROM federal_documents WHERE title IS NULL OR title = ''")
                cleanup_results['invalid_records_removed'] = cursor.rowcount or 0
                
                # Clean empty string fields
                await db.execute("UPDATE federal_documents SET agency = NULL WHERE agency = ''")
                await db.execute("UPDATE federal_documents SET publication_date = NULL WHERE publication_date = ''")
                await db.execute("UPDATE federal_documents SET type = NULL WHERE type = ''")
                cursor = await db.execute("UPDATE federal_documents SET abstract = NULL WHERE abstract = ''")
                cleanup_results['empty_fields_cleaned'] = cursor.rowcount or 0
                
                await db.commit()
                
        except Exception as e:
            cleanup_results['errors'].append(str(e))
            logger.error(f"Database cleanup failed: {e}")
        
        return cleanup_results

# Global database manager instance
db_manager = DatabaseManager()

async def test_database():
    """Test function for the database with debug capabilities"""
    try:
        await db_manager.initialize_database()
        
        print("🧪 RUNNING DATABASE TESTS")
        print("=" * 50)
        
        # Test insert
        test_doc = {
            'id': 'test-001',
            'title': 'Test Document',
            'abstract': 'This is a test document for validation',
            'document_number': 'TEST-001',
            'publication_date': '2025-01-01',
            'type': 'Rule',
            'agency': 'Test Agency',
            'raw_text': 'Full text of test document for search testing',
            'html_url': 'https://federalregister.gov/test',
            'pdf_url': 'https://federalregister.gov/test.pdf'
        }
        
        success = await db_manager.insert_document(test_doc)
        print(f"Insert test: {'✅ Success' if success else '❌ Failed'}")
        
        # Test search
        results = await db_manager.search_documents('test')
        print(f"Search test: Found {len(results)} documents")
        
        # Get comprehensive stats
        stats = await db_manager.get_database_stats()
        print(f"\n📊 DATABASE STATISTICS:")
        print(f"   • Documents: {stats['document_count']}")
        print(f"   • Agencies: {stats['agency_count']}")
        print(f"   • Latest date: {stats['latest_publication_date']}")
        print(f"   • Health: {'✅ Healthy' if stats['health_status'] else '❌ Unhealthy'}")
        
        if stats['document_types']:
            print(f"   • Document types: {stats['document_types']}")
        
        # Run debug analysis if there are issues
        if stats['agency_count'] == 0 or not stats['latest_publication_date']:
            print(f"\n🔍 RUNNING DEBUG ANALYSIS...")
            debug_report = await db_manager.debug_data_quality()
            
            print(f"   • Total documents: {debug_report['total_documents']}")
            print(f"   • Valid agencies: {debug_report['agency_analysis'].get('valid_count', 0)}")
            print(f"   • Valid dates: {debug_report['date_analysis'].get('valid_count', 0)}")
            
            if debug_report['sample_records']:
                print(f"   • Sample record: {debug_report['sample_records'][0]['title'][:50]}...")
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")

async def debug_main():
    """Main debug function"""
    try:
        print("🔍 DATABASE DEBUG MODE")
        print("=" * 50)
        
        # Run debug analysis
        debug_report = await db_manager.debug_data_quality()
        
        print(f"📊 Total documents: {debug_report['total_documents']}")
        print(f"\n🏛️ AGENCY ANALYSIS:")
        print(f"   • NULL: {debug_report['agency_analysis'].get('null_count', 0)}")
        print(f"   • Empty: {debug_report['agency_analysis'].get('empty_count', 0)}")
        print(f"   • Valid: {debug_report['agency_analysis'].get('valid_count', 0)}")
        
        print(f"\n📅 DATE ANALYSIS:")
        print(f"   • NULL: {debug_report['date_analysis'].get('null_count', 0)}")
        print(f"   • Empty: {debug_report['date_analysis'].get('empty_count', 0)}")
        print(f"   • Valid: {debug_report['date_analysis'].get('valid_count', 0)}")
        
        if debug_report['sample_records']:
            print(f"\n📝 SAMPLE RECORDS:")
            for i, record in enumerate(debug_report['sample_records'][:3], 1):
                print(f"   Record {i}: {record['title'][:50] if record['title'] else 'No title'}...")
                print(f"     Agency: {record['agency'] or 'None'}")
                print(f"     Date: {record['publication_date'] or 'None'}")
        
        # Offer to fix issues
        if (debug_report['agency_analysis'].get('empty_count', 0) > 0 or 
            debug_report['date_analysis'].get('empty_count', 0) > 0):
            
            print(f"\n🔧 APPLYING AUTOMATIC FIXES...")
            fixes = await db_manager.fix_data_issues()
            
            if fixes['empty_strings_fixed']:
                print(f"✅ Fixed empty string issues")
                print(f"   • Agency count after fix: {fixes['agency_count_after']}")
                print(f"   • Latest date after fix: {fixes['latest_date_after']}")
            
            if fixes['errors']:
                print(f"❌ Errors during fix: {fixes['errors']}")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        asyncio.run(debug_main())
    else:
        asyncio.run(test_database())