"""
Conversation Memory Module for Multilingual RAG System
Manages short-term and long-term memory for conversational context
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque
import sqlite3
import os

from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Represents a single turn in conversation"""
    user_query: str
    system_response: str
    timestamp: datetime
    language: str
    retrieved_chunks: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class ConversationMemory:
    """
    Manages conversation history with both short-term and long-term memory
    Short-term: Recent conversation turns in memory
    Long-term: Historical conversations in database
    """
    
    def __init__(self, 
                 db_path: str = "data/conversation_memory.db",
                 max_short_term: int = Config.MAX_CONVERSATION_HISTORY):
        self.db_path = db_path
        self.max_short_term = max_short_term
        
        # Short-term memory (in RAM)
        self.short_term_memory: deque = deque(maxlen=max_short_term)
        
        # Initialize database for long-term memory
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for conversation history"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_query TEXT NOT NULL,
                    system_response TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    language TEXT NOT NULL,
                    retrieved_chunks TEXT,  -- JSON string
                    metadata TEXT,  -- JSON string
                    session_id TEXT
                )
            ''')
            
            self.conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations(timestamp)
            ''')
            
            self.conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_language ON conversations(language)
            ''')
            
            self.conn.commit()
            logger.info("Conversation memory database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize conversation database: {e}")
            raise
    
    def add_conversation_turn(self, 
                            user_query: str,
                            system_response: str,
                            language: str,
                            retrieved_chunks: List[Dict[str, Any]] = None,
                            metadata: Dict[str, Any] = None,
                            session_id: str = "default") -> bool:
        """Add a new conversation turn to memory"""
        try:
            conversation_turn = ConversationTurn(
                user_query=user_query,
                system_response=system_response,
                timestamp=datetime.now(),
                language=language,
                retrieved_chunks=retrieved_chunks or [],
                metadata=metadata or {}
            )
            
            # Add to short-term memory
            self.short_term_memory.append(conversation_turn)
            
            # Add to long-term memory (database)
            self._save_to_database(conversation_turn, session_id)
            
            logger.info(f"Added conversation turn for language: {language}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add conversation turn: {e}")
            return False
    
    def _save_to_database(self, turn: ConversationTurn, session_id: str):
        """Save conversation turn to database"""
        try:
            self.conn.execute('''
                INSERT INTO conversations 
                (user_query, system_response, timestamp, language, retrieved_chunks, metadata, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                turn.user_query,
                turn.system_response,
                turn.timestamp.isoformat(),
                turn.language,
                json.dumps(turn.retrieved_chunks, ensure_ascii=False),
                json.dumps(turn.metadata, ensure_ascii=False),
                session_id
            ))
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to save to database: {e}")
    
    def get_recent_context(self, num_turns: int = 3) -> List[ConversationTurn]:
        """Get recent conversation turns for context"""
        try:
            recent_turns = list(self.short_term_memory)[-num_turns:]
            logger.info(f"Retrieved {len(recent_turns)} recent conversation turns")
            return recent_turns
            
        except Exception as e:
            logger.error(f"Failed to get recent context: {e}")
            return []
    
    def get_relevant_history(self, 
                           query: str, 
                           language: str = None,
                           days_back: int = 7,
                           limit: int = 5) -> List[Dict[str, Any]]:
        """Get relevant conversation history based on query similarity"""
        try:
            # Get conversations from the last N days
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            query_conditions = "timestamp >= ?"
            params = [cutoff_date.isoformat()]
            
            if language:
                query_conditions += " AND language = ?"
                params.append(language)
            
            cursor = self.conn.execute(f'''
                SELECT user_query, system_response, timestamp, language, retrieved_chunks, metadata
                FROM conversations 
                WHERE {query_conditions}
                ORDER BY timestamp DESC
                LIMIT ?
            ''', params + [limit])
            
            results = []
            for row in cursor.fetchall():
                try:
                    retrieved_chunks = json.loads(row[4]) if row[4] else []
                    metadata = json.loads(row[5]) if row[5] else {}
                except json.JSONDecodeError:
                    retrieved_chunks = []
                    metadata = {}
                
                results.append({
                    'user_query': row[0],
                    'system_response': row[1],
                    'timestamp': row[2],
                    'language': row[3],
                    'retrieved_chunks': retrieved_chunks,
                    'metadata': metadata
                })
            
            logger.info(f"Retrieved {len(results)} relevant history items")
            return results
            
        except Exception as e:
            logger.error(f"Failed to get relevant history: {e}")
            return []
    
    def format_context_for_llm(self, num_recent: int = 3) -> str:
        """Format recent conversation context for LLM input"""
        try:
            recent_turns = self.get_recent_context(num_recent)
            
            if not recent_turns:
                return ""
            
            context_parts = []
            for i, turn in enumerate(recent_turns):
                context_parts.append(f"Previous conversation {i+1}:")
                context_parts.append(f"User: {turn.user_query}")
                context_parts.append(f"Assistant: {turn.system_response}")
                context_parts.append("")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Failed to format context: {e}")
            return ""
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about conversation history"""
        try:
            # Short-term stats
            short_term_count = len(self.short_term_memory)
            
            # Long-term stats
            cursor = self.conn.execute('''
                SELECT 
                    COUNT(*) as total_conversations,
                    COUNT(DISTINCT language) as languages_used,
                    MIN(timestamp) as first_conversation,
                    MAX(timestamp) as last_conversation
                FROM conversations
            ''')
            
            long_term_stats = cursor.fetchone()
            
            # Language distribution
            cursor = self.conn.execute('''
                SELECT language, COUNT(*) as count
                FROM conversations
                GROUP BY language
                ORDER BY count DESC
            ''')
            
            language_dist = dict(cursor.fetchall())
            
            return {
                "short_term_memory": {
                    "current_turns": short_term_count,
                    "max_capacity": self.max_short_term
                },
                "long_term_memory": {
                    "total_conversations": long_term_stats[0],
                    "languages_used": long_term_stats[1],
                    "first_conversation": long_term_stats[2],
                    "last_conversation": long_term_stats[3],
                    "language_distribution": language_dist
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get conversation stats: {e}")
            return {}
    
    def clear_short_term_memory(self):
        """Clear short-term memory"""
        try:
            self.short_term_memory.clear()
            logger.info("Short-term memory cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear short-term memory: {e}")
    
    def clear_all_memory(self):
        """Clear both short-term and long-term memory"""
        try:
            # Clear short-term
            self.short_term_memory.clear()
            
            # Clear long-term
            self.conn.execute("DELETE FROM conversations")
            self.conn.commit()
            
            logger.info("All conversation memory cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear all memory: {e}")
    
    def export_conversations(self, 
                           output_path: str,
                           language: str = None,
                           days_back: int = None) -> bool:
        """Export conversations to JSON file"""
        try:
            query_conditions = "1=1"
            params = []
            
            if language:
                query_conditions += " AND language = ?"
                params.append(language)
            
            if days_back:
                cutoff_date = datetime.now() - timedelta(days=days_back)
                query_conditions += " AND timestamp >= ?"
                params.append(cutoff_date.isoformat())
            
            cursor = self.conn.execute(f'''
                SELECT user_query, system_response, timestamp, language, retrieved_chunks, metadata
                FROM conversations 
                WHERE {query_conditions}
                ORDER BY timestamp DESC
            ''', params)
            
            conversations = []
            for row in cursor.fetchall():
                try:
                    retrieved_chunks = json.loads(row[4]) if row[4] else []
                    metadata = json.loads(row[5]) if row[5] else {}
                except json.JSONDecodeError:
                    retrieved_chunks = []
                    metadata = {}
                
                conversations.append({
                    'user_query': row[0],
                    'system_response': row[1],
                    'timestamp': row[2],
                    'language': row[3],
                    'retrieved_chunks': retrieved_chunks,
                    'metadata': metadata
                })
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(conversations, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Exported {len(conversations)} conversations to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export conversations: {e}")
            return False
    
    def __del__(self):
        """Close database connection"""
        try:
            if hasattr(self, 'conn'):
                self.conn.close()
        except:
            pass

# Example usage
if __name__ == "__main__":
    # Test conversation memory
    memory = ConversationMemory()
    
    # Add some test conversations
    memory.add_conversation_turn(
        user_query="অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
        system_response="অনুপমের ভাষায় সুপুরুষ বলতে শুম্ভুনাথকে বোঝানো হয়েছে।",
        language="bengali",
        metadata={"confidence": 0.95}
    )
    
    memory.add_conversation_turn(
        user_query="Who is mentioned as a good person?",
        system_response="According to Anupam, Shumbhunath is referred to as a good person.",
        language="english",
        metadata={"confidence": 0.90}
    )
    
    # Get context
    context = memory.format_context_for_llm(2)
    print("Formatted context:")
    print(context)
    
    # Get stats
    stats = memory.get_conversation_stats()
    print("\nConversation stats:")
    print(json.dumps(stats, indent=2, default=str))
