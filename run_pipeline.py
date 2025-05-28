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

async def check_data_status():
    """Check current data status in database"""
    print("\n📊 Checking current data status...")
    
    try:
        # Initialize database properly
        await db_manager.initialize_database()
        
        # Get total documents
        total_docs = await db_manager.get_document_count()
        
        # Get recent documents to check dates and agencies
        recent_docs = await db_manager.get_recent_documents(days=365, limit=1000)
        
        print(f"📈 Total documents in database: {total_docs}")
        
        if recent_docs:
            # Get date range from recent documents
            valid_dates = []
            agencies = {}
            
            for doc in recent_docs:
                pub_date = doc.get('publication_date')
                if pub_date:
                    valid_dates.append(pub_date)
                
                agency = doc.get('agency', '').strip()
                if agency:
                    agencies[agency] = agencies.get(agency, 0) + 1
            
            if valid_dates:
                latest_date = max(valid_dates)
                oldest_date = min(valid_dates)
                print(f"📅 Latest document date: {latest_date}")
                print(f"📅 Oldest recent document date: {oldest_date}")
            
            if agencies:
                print("\n🏛️  Top 5 agencies by document count:")
                sorted_agencies = sorted(agencies.items(), key=lambda x: x[1], reverse=True)[:5]
                for agency, count in sorted_agencies:
                    print(f"   • {agency}: {count} documents")
        else:
            print("📈 No recent documents found in database")
                        
    except Exception as e:
        print(f"❌ Error checking data status: {e}")

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

async def check_pipeline_logs():
    """Check recent pipeline execution logs using DatabaseManager methods"""
    print("\n📋 Checking recent pipeline logs...")
    
    try:
        # Initialize database properly
        await db_manager.initialize_database()
        
        # Use direct SQL query since we don't have a dedicated method for logs
        import aiosqlite
        async with aiosqlite.connect(db_manager.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT id, run_date, status, records_processed, start_time, end_time, error_message
                FROM pipeline_logs 
                ORDER BY start_time DESC 
                LIMIT 10
            """) as cursor:
                logs = await cursor.fetchall()
                
                if logs:
                    print("Recent pipeline runs:")
                    for log in logs:
                        status_emoji = "✅" if log['status'] == "completed" else "❌" if log['status'] == "failed" else "🔄"
                        print(f"   {status_emoji} {log['run_date']} - {log['status']} ({log['records_processed']} records)")
                        if log['error_message']:
                            print(f"      Error: {log['error_message']}")
                else:
                    print("No pipeline logs found.")
                    
    except Exception as e:
        print(f"❌ Error checking pipeline logs: {e}")

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
            # Check final status
            asyncio.run(check_data_status())
            print("\n🎉 Data pipeline completed successfully!")
            print("💡 You can now start the main application with: python main.py")
        else:
            print("\n💥 Data pipeline failed!")
            sys.exit(1)
            
    elif len(sys.argv) == 2:
        if sys.argv[1] == "--status":
            # Just check status
            asyncio.run(check_data_status())
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
            asyncio.run(check_data_status())
            print(f"\n🎉 Data pipeline completed successfully for {start_date} to {end_date}!")
        else:
            print("\n💥 Data pipeline failed!")
            sys.exit(1)
    else:
        print("❌ Invalid arguments. Use --help for usage information.")
        sys.exit(1)

if __name__ == "__main__":
    main()