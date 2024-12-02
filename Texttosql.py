from typing import List, Dict, Any, Optional, Union
import logging
import json
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import sqlparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import snowflake.connector
from snowflake.connector.errors import ProgrammingError
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import yaml
import re
from prometheus_client import Counter, Histogram
import torch
import tiktoken

# Metrics for monitoring
SQL_GENERATION_TIME = Histogram('sql_generation_seconds', 'Time spent generating SQL')
TOOL_EXECUTION_COUNT = Counter('tool_executions_total', 'Total tool executions', ['tool_name'])
ERROR_COUNT = Counter('errors_total', 'Total errors encountered', ['error_type'])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToolName(Enum):
    SCHEMA_LOOKUP = "schema_lookup"
    SQL_VALIDATION = "sql_validation"
    SQL_EXECUTION = "sql_execution"
    DATA_SAMPLING = "data_sampling"
    QUERY_OPTIMIZATION = "query_optimization"

@dataclass
class SchemaInfo:
    """Represents database schema information"""
    table_name: str
    columns: List[Dict[str, str]]  # [{name: str, type: str, description: str}]
    sample_data: Optional[pd.DataFrame] = None
    relationships: List[Dict[str, str]] = None  # Foreign key relationships

class DatabaseConfig:
    """Database connection configuration"""
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.snowflake_config = config['snowflake']
        self.schema_cache_ttl = config.get('schema_cache_ttl', 3600)  # 1 hour default

class BaseTool(ABC):
    """Abstract base class for tools"""
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        pass

class SchemaLookupTool(BaseTool):
    """Tool for looking up and understanding database schema"""
    def __init__(self, db_config: DatabaseConfig):
        self.db_config = db_config
        self._schema_cache = {}
        self._last_cache_update = None

    def execute(self, table_pattern: str = None) -> Dict[str, Any]:
        """Lookup schema information for matching tables"""
        try:
            schemas = self._get_schemas(table_pattern)
            return {
                "status": "success",
                "schemas": schemas
            }
        except Exception as e:
            logger.error(f"Schema lookup failed: {str(e)}")
            ERROR_COUNT.labels(error_type='schema_lookup').inc()
            return {"status": "error", "error": str(e)}

    def _get_schemas(self, table_pattern: str = None) -> List[SchemaInfo]:
        """Get schema information, using cache if available"""
        if not self._is_cache_valid():
            self._refresh_schema_cache()
        
        if table_pattern:
            return [
                schema for schema in self._schema_cache.values()
                if re.search(table_pattern, schema.table_name, re.IGNORECASE)
            ]
        return list(self._schema_cache.values())

class SQLValidationTool(BaseTool):
    """Tool for validating SQL queries"""
    def __init__(self):
        self.forbidden_patterns = [
            r"DROP\s+",
            r"DELETE\s+",
            r"TRUNCATE\s+",
            r"ALTER\s+",
            r"CREATE\s+",
            r"INSERT\s+",
            r"UPDATE\s+"
        ]

    def execute(self, sql: str) -> Dict[str, Any]:
        """Validate SQL query for correctness and safety"""
        try:
            # Check for dangerous operations
            for pattern in self.forbidden_patterns:
                if re.search(pattern, sql, re.IGNORECASE):
                    return {
                        "status": "error",
                        "error": "Query contains forbidden operations"
                    }

            # Parse SQL for syntax
            parsed = sqlparse.parse(sql)
            if not parsed:
                return {
                    "status": "error",
                    "error": "Failed to parse SQL"
                }

            return {
                "status": "success",
                "is_valid": True,
                "normalized_sql": str(parsed[0])
            }
        except Exception as e:
            logger.error(f"SQL validation failed: {str(e)}")
            ERROR_COUNT.labels(error_type='sql_validation').inc()
            return {"status": "error", "error": str(e)}

class SQLExecutionTool(BaseTool):
    """Tool for executing SQL queries"""
    def __init__(self, db_config: DatabaseConfig):
        self.db_config = db_config

    def execute(self, sql: str, max_rows: int = 1000) -> Dict[str, Any]:
        """Execute SQL query and return results"""
        try:
            with snowflake.connector.connect(**self.db_config.snowflake_config) as conn:
                cursor = conn.cursor()
                cursor.execute(sql)
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchmany(max_rows)
                
                return {
                    "status": "success",
                    "results": {
                        "columns": columns,
                        "data": data,
                        "row_count": len(data),
                        "truncated": cursor.rowcount > len(data) if cursor.rowcount >= 0 else False
                    }
                }
        except Exception as e:
            logger.error(f"SQL execution failed: {str(e)}")
            ERROR_COUNT.labels(error_type='sql_execution').inc()
            return {"status": "error", "error": str(e)}

class TextToSQLAgent:
    """Main agent class for converting text to SQL and executing queries"""
    def __init__(self, model_name: str, db_config: DatabaseConfig):
        # Initialize LLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Initialize tools
        self.tools = {
            ToolName.SCHEMA_LOOKUP.value: SchemaLookupTool(db_config),
            ToolName.SQL_VALIDATION.value: SQLValidationTool(),
            ToolName.SQL_EXECUTION.value: SQLExecutionTool(db_config)
        }
        
        # Conversation state
        self.conversation_history = []
        self.execution_flow = []

    def _generate_schema_prompt(self, query: str) -> str:
        """Generate prompt for schema understanding"""
        return f"""
Given the user query: "{query}"

Determine which tables and columns might be relevant.
Consider:
1. Key entities mentioned in the query
2. Required relationships between tables
3. Potential filters or aggregations needed

Response format:
{{
    "relevant_tables": ["table1", "table2"],
    "explanation": "Reasoning for table selection"
}}
"""

    def _generate_sql_prompt(self, query: str, schema_info: List[SchemaInfo]) -> str:
        """Generate prompt for SQL generation"""
        schema_context = "\n".join(
            f"Table: {schema.table_name}\nColumns: {', '.join(col['name'] for col in schema.columns)}"
            for schema in schema_info
        )
        
        return f"""
User Query: {query}

Available Schema:
{schema_context}

Generate a SQL query that:
1. Uses only the available tables and columns
2. Implements proper joins if needed
3. Includes appropriate filters and aggregations
4. Is optimized for performance

Response format:
{{
    "sql": "YOUR SQL QUERY",
    "explanation": "Explanation of the query logic"
}}
"""

    def _call_llm(self, prompt: str) -> Dict:
        """Call local LLM with proper error handling and metrics"""
        try:
            with SQL_GENERATION_TIME.time():
                inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
                outputs = self.model.generate(
                    **inputs,
                    max_length=500,
                    num_return_sequences=1,
                    temperature=0.7
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return json.loads(response)
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            ERROR_COUNT.labels(error_type='llm_call').inc()
            raise

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process natural language query to SQL and execute"""
        try:
            # Record start of new query
            self.execution_flow.append({
                "timestamp": datetime.now().isoformat(),
                "step": "query_received",
                "details": {"query": query}
            })

            # 1. Schema Understanding
            schema_decision = self._call_llm(self._generate_schema_prompt(query))
            relevant_schemas = self.tools[ToolName.SCHEMA_LOOKUP.value].execute(
                table_pattern="|".join(schema_decision["relevant_tables"])
            )

            if relevant_schemas["status"] == "error":
                return {"status": "error", "error": "Failed to fetch schema information"}

            # 2. SQL Generation
            sql_response = self._call_llm(
                self._generate_sql_prompt(query, relevant_schemas["schemas"])
            )

            # 3. SQL Validation
            validation_result = self.tools[ToolName.SQL_VALIDATION.value].execute(
                sql=sql_response["sql"]
            )

            if validation_result["status"] == "error" or not validation_result.get("is_valid"):
                return {
                    "status": "error",
                    "error": "Generated SQL failed validation",
                    "details": validation_result
                }

            # 4. SQL Execution
            execution_result = self.tools[ToolName.SQL_EXECUTION.value].execute(
                sql=validation_result["normalized_sql"]
            )

            # Record complete flow
            self.execution_flow.append({
                "timestamp": datetime.now().isoformat(),
                "step": "query_completed",
                "details": {
                    "schema_decision": schema_decision,
                    "sql_generated": sql_response,
                    "validation_result": validation_result,
                    "execution_result": execution_result
                }
            })

            return {
                "status": "success",
                "query": query,
                "sql": validation_result["normalized_sql"],
                "results": execution_result["results"],
                "explanation": sql_response["explanation"]
            }

        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            ERROR_COUNT.labels(error_type='query_processing').inc()
            return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    # Example usage
    config = DatabaseConfig("config.yaml")
    agent = TextToSQLAgent("google/flan-t5-base", config)
    
    query = "What were the total sales by product category in the last month?"
    result = agent.process_query(query)
    
    print("\nQuery:", query)
    print("\nExecution Flow:")
    for step in agent.execution_flow:
        print(f"\n{step['step']}:")
        print(f"Timestamp: {step['timestamp']}")
        print(f"Details: {json.dumps(step['details'], indent=2)}")
