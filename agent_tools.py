from typing import Dict, List, Any
from database import db_manager
import json
from datetime import datetime, timedelta

class AgentTools:
    """Tools available to the LLM agent for querying federal documents"""
    
    @staticmethod
    async def search_federal_documents(query: str, limit: int = 10) -> str:
        """
        Search federal documents using full-text search.
        Args:
            query: Search query string
            limit: Maximum number of results (default 10, max 50)
        Returns:
            JSON string with search results
        """
        try:
            limit = min(limit, 50)  
            results = await db_manager.search_documents(query, limit)
            
            if not results:
                return json.dumps({
                    "status": "success",
                    "message": "No documents found matching your query.",
                    "results": []
                })
            
            # Format results for LLM
            formatted_results = []
            for doc in results:
                pub_date = doc.get('publication_date')
                if pub_date and isinstance(pub_date, str):
                    formatted_date = pub_date
                else:
                    formatted_date = None
                
                formatted_doc = {
                    "title": doc.get('title', ''),
                    "document_number": doc.get('document_number', ''),
                    "publication_date": formatted_date,
                    "type": doc.get('type', ''),
                    "agency": doc.get('agency', ''),
                    "abstract": doc.get('abstract', '')[:500] + "..." if doc.get('abstract') and len(doc.get('abstract', '')) > 500 else doc.get('abstract', ''),
                    "url": doc.get('html_url', '')
                }
                formatted_results.append(formatted_doc)
            
            return json.dumps({
                "status": "success",
                "message": f"Found {len(results)} documents matching your search.",
                "count": len(results),
                "results": formatted_results
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Error searching documents: {str(e)}",
                "results": []
            })
    
    @staticmethod
    async def get_documents_by_date_range(start_date: str, end_date: str, limit: int = 20) -> str:
        """
        Get federal documents within a specific date range.
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            limit: Maximum number of results (default 20, max 50)
        Returns:
            JSON string with documents in date range
        """
        try:
            limit = min(limit, 50)
            results = await db_manager.get_documents_by_date_range(start_date, end_date, limit)
            
            if not results:
                return json.dumps({
                    "status": "success",
                    "message": f"No documents found between {start_date} and {end_date}.",
                    "results": []
                })
            
            formatted_results = []
            for doc in results:
                # Handle publication_date properly
                pub_date = doc.get('publication_date')
                if pub_date and isinstance(pub_date, str):
                    formatted_date = pub_date
                else:
                    formatted_date = None
                
                formatted_doc = {
                    "title": doc.get('title', ''),
                    "document_number": doc.get('document_number', ''),
                    "publication_date": formatted_date,
                    "type": doc.get('type', ''),
                    "agency": doc.get('agency', ''),
                    "abstract": doc.get('abstract', '')[:500] + "..." if doc.get('abstract') and len(doc.get('abstract', '')) > 500 else doc.get('abstract', ''),
                    "url": doc.get('html_url', '')
                }
                formatted_results.append(formatted_doc)
            
            return json.dumps({
                "status": "success",
                "message": f"Found {len(results)} documents between {start_date} and {end_date}.",
                "count": len(results),
                "date_range": {"start": start_date, "end": end_date},
                "results": formatted_results
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Error getting documents by date range: {str(e)}",
                "results": []
            })
    
    @staticmethod
    async def get_documents_by_agency(agency: str, limit: int = 20) -> str:
        """
        Get federal documents by specific agency.
        Args:
            agency: Agency name or partial name
            limit: Maximum number of results (default 20, max 50)
        Returns:
            JSON string with agency documents
        """
        try:
            limit = min(limit, 50)
            results = await db_manager.get_documents_by_agency(agency, limit)
            
            if not results:
                return json.dumps({
                    "status": "success",
                    "message": f"No documents found for agency: {agency}",
                    "results": []
                })
            
            formatted_results = []
            for doc in results:
                # Handle publication_date properly
                pub_date = doc.get('publication_date')
                if pub_date and isinstance(pub_date, str):
                    formatted_date = pub_date
                else:
                    formatted_date = None
                
                formatted_doc = {
                    "title": doc.get('title', ''),
                    "document_number": doc.get('document_number', ''),
                    "publication_date": formatted_date,
                    "type": doc.get('type', ''),
                    "agency": doc.get('agency', ''),
                    "abstract": doc.get('abstract', '')[:500] + "..." if doc.get('abstract') and len(doc.get('abstract', '')) > 500 else doc.get('abstract', ''),
                    "url": doc.get('html_url', '')
                }
                formatted_results.append(formatted_doc)
            
            return json.dumps({
                "status": "success",
                "message": f"Found {len(results)} documents from agency: {agency}",
                "count": len(results),
                "agency": agency,
                "results": formatted_results
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Error getting documents by agency: {str(e)}",
                "results": []
            })
    
    @staticmethod
    async def get_recent_documents(days: int = 7, limit: int = 20) -> str:
        """
        Get recent federal documents from the last N days.
        Args:
            days: Number of days to look back (default 7, max 30)
            limit: Maximum number of results (default 20, max 50)
        Returns:
            JSON string with recent documents
        """
        try:
            days = min(days, 30)  # Cap at 30 days
            limit = min(limit, 50)
            results = await db_manager.get_recent_documents(days, limit)
            
            if not results:
                return json.dumps({
                    "status": "success",
                    "message": f"No documents found in the last {days} days.",
                    "results": []
                })
            
            formatted_results = []
            for doc in results:
                # Handle publication_date properly
                pub_date = doc.get('publication_date')
                if pub_date and isinstance(pub_date, str):
                    formatted_date = pub_date
                else:
                    formatted_date = None
                
                formatted_doc = {
                    "title": doc.get('title', ''),
                    "document_number": doc.get('document_number', ''),
                    "publication_date": formatted_date,
                    "type": doc.get('type', ''),
                    "agency": doc.get('agency', ''),
                    "abstract": doc.get('abstract', '')[:500] + "..." if doc.get('abstract') and len(doc.get('abstract', '')) > 500 else doc.get('abstract', ''),
                    "url": doc.get('html_url', '')
                }
                formatted_results.append(formatted_doc)
            
            return json.dumps({
                "status": "success",
                "message": f"Found {len(results)} documents from the last {days} days.",
                "count": len(results),
                "days_back": days,
                "results": formatted_results
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Error getting recent documents: {str(e)}",
                "results": []
            })

    @staticmethod
    async def get_database_stats() -> str:
        """
        Get database statistics including document count, agency count, and latest publication date.
        Returns:
            JSON string with database statistics
        """
        try:
            doc_count = await db_manager.get_document_count()
            agency_count = await db_manager.get_agency_count()
            latest_date = await db_manager.get_latest_publication_date()
            
            return json.dumps({
                "status": "success",
                "statistics": {
                    "total_documents": doc_count,
                    "unique_agencies": agency_count,
                    "latest_publication_date": latest_date
                }
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Error getting database statistics: {str(e)}",
                "statistics": {}
            })

# Tool definitions for LLM function calling
AGENT_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_federal_documents",
            "description": "Search federal documents using full-text search. Use this when the user asks about specific topics, keywords, or content within federal documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string. Use keywords related to what the user is looking for."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (1-50, defaults to 10)",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_documents_by_date_range",
            "description": "Get federal documents published within a specific date range. Use this when the user asks about documents from a particular time period.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (1-50, defaults to 20)",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 20
                    }
                },
                "required": ["start_date", "end_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_documents_by_agency",
            "description": "Get federal documents published by a specific agency. Use this when the user asks about documents from particular government agencies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "agency": {
                        "type": "string",
                        "description": "Name or partial name of the agency (e.g., 'EPA', 'Department of Defense', 'Treasury')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (1-50, defaults to 20)",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 20
                    }
                },
                "required": ["agency"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_documents",
            "description": "Get recent federal documents from the last N days. Use this when the user asks about recent or latest federal documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Number of days to look back (1-30, defaults to 7)",
                        "minimum": 1,
                        "maximum": 30,
                        "default": 7
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (1-50, defaults to 20)",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 20
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_database_stats",
            "description": "Get database statistics including total document count, unique agency count, and latest publication date.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

# Tool execution mapping
TOOL_FUNCTIONS = {
    "search_federal_documents": AgentTools.search_federal_documents,
    "get_documents_by_date_range": AgentTools.get_documents_by_date_range,
    "get_documents_by_agency": AgentTools.get_documents_by_agency,
    "get_recent_documents": AgentTools.get_recent_documents,
    "get_database_stats": AgentTools.get_database_stats
}
