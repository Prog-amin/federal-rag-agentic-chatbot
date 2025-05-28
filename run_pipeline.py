#!/usr/bin/env python3
"""
Script to run the data pipeline manually
This simulates the daily data update process
"""
import asyncio
import sys
from datetime import datetime
from data_pipeline import DataPipeline
from database import db_manager

async def run_pipeline():
    """Run the complete data pipeline"""
    print("🔄 Starting Data Pipeline...")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Initialize database using the proper method
        await db_manager.initialize_database()
        print("✅ Database connection established")
        
        # Initialize and run pipeline
        pipeline = DataPipeline()
        
        print("📥 Downloading and processing latest data...")
        success = await pipeline.run_pipeline()
        
        if not success:
            print("❌ Pipeline failed. Check logs for details.")
            return False
        
        print("✅ Pipeline completed successfully!")
        print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline failed with error: {str(e)}")
        return False
    finally:
        # Always close the database connection
        await db_manager.close_pool()

async def check_data_status():
    """Check current data status in database"""
    print("\n📊 Checking current data status...")
    
    try:
        # Initialize database properly
        await db_manager.initialize_database()
        
        # Get the pool after ensuring it's initialized
        pool = db_manager.pool
        assert pool is not None, "Pool should be initialized at this point"
        
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                # Count total documents
                await cursor.execute("SELECT COUNT(*) FROM federal_documents")
                result = await cursor.fetchone()
                total_docs = result[0] if result else 0
                
                # Get latest document date
                await cursor.execute(
                    "SELECT MAX(publication_date) FROM federal_documents"
                )
                result = await cursor.fetchone()
                latest_date = result[0] if result and result[0] else None
                
                # Get oldest document date
                await cursor.execute(
                    "SELECT MIN(publication_date) FROM federal_documents"
                )
                result = await cursor.fetchone()
                oldest_date = result[0] if result and result[0] else None
                
                # Get documents by agency count
                await cursor.execute("""
                    SELECT agency, COUNT(*) as count 
                    FROM federal_documents 
                    WHERE agency != '' 
                    GROUP BY agency 
                    ORDER BY count DESC 
                    LIMIT 5
                """)
                agency_stats = await cursor.fetchall()
                
                print(f"📈 Total documents in database: {total_docs}")
                if latest_date:
                    print(f"📅 Latest document date: {latest_date}")
                if oldest_date:
                    print(f"📅 Oldest document date: {oldest_date}")
                
                if agency_stats:
                    print("\n🏛️  Top 5 agencies by document count:")
                    for agency, count in agency_stats:
                        print(f"   • {agency}: {count} documents")
                        
    except Exception as e:
        print(f"❌ Error checking data status: {e}")
    finally:
        await db_manager.close_pool()

async def run_pipeline_with_dates(start_date: str, end_date: str):
    """Run pipeline for specific date range"""
    print(f"🔄 Starting Data Pipeline for date range: {start_date} to {end_date}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Initialize database using the proper method
        await db_manager.initialize_database()
        print("✅ Database connection established")
        
        # Initialize and run pipeline with specific dates
        pipeline = DataPipeline()
        
        print(f"📥 Downloading data from {start_date} to {end_date}...")
        success = await pipeline.run_pipeline(start_date, end_date)
        
        if not success:
            print("❌ Pipeline failed. Check logs for details.")
            return False
        
        print("✅ Pipeline completed successfully!")
        print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline failed with error: {str(e)}")
        return False
    finally:
        await db_manager.close_pool()

async def check_pipeline_logs():
    """Check recent pipeline execution logs"""
    print("\n📋 Checking recent pipeline logs...")
    
    try:
        # Initialize database properly
        await db_manager.initialize_database()
        
        # Get the pool after ensuring it's initialized
        pool = db_manager.pool
        assert pool is not None, "Pool should be initialized at this point"
        
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("""
                    SELECT id, run_date, status, records_processed, start_time, end_time, error_message
                    FROM pipeline_logs 
                    ORDER BY start_time DESC 
                    LIMIT 10
                """)
                logs = await cursor.fetchall()
                
                if logs:
                    print("Recent pipeline runs:")
                    for log in logs:
                        log_id, run_date, status, records, start_time, end_time, error = log
                        status_emoji = "✅" if status == "completed" else "❌" if status == "failed" else "🔄"
                        print(f"   {status_emoji} {run_date} - {status} ({records} records)")
                        if error:
                            print(f"      Error: {error}")
                else:
                    print("No pipeline logs found.")
                    
    except Exception as e:
        print(f"❌ Error checking pipeline logs: {e}")
    finally:
        await db_manager.close_pool()

# Alternative approach using DatabaseManager methods instead of direct SQL
async def check_data_status_safe():
    """Check current data status using DatabaseManager methods"""
    print("\n📊 Checking current data status...")
    
    try:
        # Get recent documents to check if data exists
        recent_docs = await db_manager.get_recent_documents(days=365, limit=1000)  # Get up to 1000 recent docs
        total_recent = len(recent_docs)
        
        if total_recent > 0:
            # Get date range from recent documents - filter out None values
            valid_dates = []
            for doc in recent_docs:
                pub_date = doc.get('publication_date')
                if pub_date is not None:
                    valid_dates.append(pub_date)
            
            if valid_dates:
                # Use a safer approach for finding min/max dates
                try:
                    latest_date = max(valid_dates)
                    oldest_date = min(valid_dates)
                    print(f"📈 Recent documents found: {total_recent}")
                    print(f"📅 Latest document date: {latest_date}")
                    print(f"📅 Oldest recent document date: {oldest_date}")
                except (ValueError, TypeError) as e:
                    print(f"📈 Recent documents found: {total_recent}")
                    print(f"⚠️  Could not determine date range: {e}")
                
                # Count by agency
                agencies = {}
                for doc in recent_docs:
                    agency = doc.get('agency', 'Unknown')
                    if agency and agency.strip():
                        agencies[agency] = agencies.get(agency, 0) + 1
                
                if agencies:
                    print("\n🏛️  Top agencies by document count:")
                    sorted_agencies = sorted(agencies.items(), key=lambda x: x[1], reverse=True)[:5]
                    for agency, count in sorted_agencies:
                        print(f"   • {agency}: {count} documents")
            else:
                print(f"📈 Found {total_recent} documents but no valid dates")
        else:
            print("📈 No recent documents found in database")
                        
    except Exception as e:
        print(f"❌ Error checking data status: {e}")

def print_usage():
    """Print usage information"""
    print("Usage:")
    print("  python run_pipeline.py                    # Run pipeline for last 7 days")
    print("  python run_pipeline.py --status           # Check current data status")
    print("  python run_pipeline.py --logs             # Check pipeline execution logs")
    print("  python run_pipeline.py --dates START END  # Run pipeline for specific date range")
    print("                                             # Dates should be in YYYY-MM-DD format")

def main():
    """Main function"""
    if len(sys.argv) == 1:
        # Run full pipeline (default - last 7 days)
        success = asyncio.run(run_pipeline())
        
        if success:
            # Check final status using the safe method
            asyncio.run(check_data_status_safe())
            print("\n🎉 Data pipeline completed successfully!")
            print("💡 You can now start the main application with: python main.py")
        else:
            print("\n💥 Data pipeline failed!")
            sys.exit(1)
            
    elif len(sys.argv) == 2:
        if sys.argv[1] == "--status":
            # Just check status using the safe method
            asyncio.run(check_data_status_safe())
        elif sys.argv[1] == "--logs":
            # Check pipeline logs
            asyncio.run(check_pipeline_logs())
        elif sys.argv[1] == "--help":
            print_usage()
        else:
            print("❌ Invalid argument. Use --help for usage information.")
            sys.exit(1)
            
    elif len(sys.argv) == 4 and sys.argv[1] == "--dates":
        # Run pipeline for specific date range
        start_date = sys.argv[2]
        end_date = sys.argv[3]
        
        # Basic date format validation
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            print("❌ Invalid date format. Use YYYY-MM-DD format.")
            sys.exit(1)
        
        success = asyncio.run(run_pipeline_with_dates(start_date, end_date))
        
        if success:
            asyncio.run(check_data_status_safe())
            print(f"\n🎉 Data pipeline completed successfully for {start_date} to {end_date}!")
        else:
            print("\n💥 Data pipeline failed!")
            sys.exit(1)
    else:
        print("❌ Invalid arguments. Use --help for usage information.")
        sys.exit(1)

if __name__ == "__main__":
    main()