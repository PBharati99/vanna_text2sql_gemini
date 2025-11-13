"""Snowflake implementation of SqlRunner interface."""
from typing import Optional, Union
import os
import pandas as pd
from threading import Lock

from vanna.capabilities.sql_runner import SqlRunner, RunSqlToolArgs
from vanna.core.tool import ToolContext


class SnowflakeRunner(SqlRunner):
    """Snowflake implementation of the SqlRunner interface."""

    def __init__(
        self,
        account: str,
        username: str,
        password: Optional[str] = None,
        database: str = "",
        role: Optional[str] = None,
        warehouse: Optional[str] = None,
        private_key_path: Optional[str] = None,
        private_key_passphrase: Optional[str] = None,
        private_key_content: Optional[bytes] = None,
        authenticator: Optional[str] = None,
        **kwargs
    ):
        """Initialize with Snowflake connection parameters.

        Args:
            account: Snowflake account identifier
            username: Database user
            password: Database password (optional if using key-pair or external browser auth)
            database: Database name
            role: Snowflake role to use (optional)
            warehouse: Snowflake warehouse to use (optional)
            private_key_path: Path to private key file for RSA key-pair authentication (optional)
            private_key_passphrase: Passphrase for encrypted private key (optional)
            private_key_content: Private key content as bytes (optional, alternative to private_key_path)
            authenticator: Authentication method. Use 'externalbrowser' for SSO/external browser auth (optional)
            **kwargs: Additional snowflake.connector connection parameters

        Note:
            Authentication methods (choose one):
            - externalbrowser: SSO/external browser authentication (recommended for enterprise)
            - password: User/password authentication
            - private_key_path/private_key_content: RSA key-pair authentication
        """
        try:
            import snowflake.connector
            self.snowflake = snowflake.connector
        except ImportError as e:
            raise ImportError(
                "snowflake-connector-python package is required. "
                "Install with: pip install 'vanna[snowflake]'"
            ) from e

        # Validate that at least one authentication method is provided
        if not authenticator and not password and not private_key_path and not private_key_content:
            raise ValueError(
                "Either authenticator, password, or private_key_path/private_key_content must be provided for authentication"
            )

        # Validate private key path exists if provided
        if private_key_path and not os.path.isfile(private_key_path):
            raise FileNotFoundError(
                f"Private key file not found: {private_key_path}"
            )

        self.account = account
        self.username = username
        self.password = password
        self.database = database
        self.role = role
        self.warehouse = warehouse
        self.private_key_path = private_key_path
        self.private_key_passphrase = private_key_passphrase
        self.private_key_content = private_key_content
        self.authenticator = authenticator
        self.kwargs = kwargs
        
        # Connection management
        self._connection = None
        self._connection_lock = Lock()
        
        # Establish initial connection (browser auth happens here, ONCE)
        self._connect()
    
    def _connect(self):
        """Establish connection to Snowflake and configure session.
        
        This is where external browser authentication happens (only once during initialization).
        """
        # Build connection parameters
        conn_params = {
            "user": self.username,
            "account": self.account,
            "client_session_keep_alive": True,  # Keep connection alive with periodic pings
        }
        
        # Add database if specified
        if self.database:
            conn_params["database"] = self.database
        
        # Configure authentication method (priority: authenticator > private_key > password)
        if self.authenticator:
            # Use specified authenticator (e.g., 'externalbrowser' for SSO)
            # BROWSER AUTHENTICATION HAPPENS HERE
            conn_params["authenticator"] = self.authenticator
            # For external browser auth, password is not required
            if self.password:
                conn_params["password"] = self.password
        elif self.private_key_path or self.private_key_content:
            # Use RSA key-pair authentication
            if self.private_key_path:
                conn_params["private_key_path"] = self.private_key_path
            else:
                conn_params["private_key_content"] = self.private_key_content
            
            # Add passphrase if provided
            if self.private_key_passphrase:
                conn_params["private_key_passphrase"] = self.private_key_passphrase
        else:
            # Use password authentication (fallback)
            conn_params["password"] = self.password
        
        # Add any additional kwargs
        conn_params.update(self.kwargs)
        
        # Create connection (this is where authentication happens)
        print(f"[INFO] Connecting to Snowflake as {self.username}@{self.account}...")
        self._connection = self.snowflake.connect(**conn_params)
        print("[INFO] Successfully connected to Snowflake")
        
        # Configure session settings (role, warehouse, database)
        self._configure_session()
    
    def _configure_session(self):
        """Apply role, warehouse, and database settings to the session."""
        # Helper to quote identifiers with special characters or lowercase
        def _quote(identifier: Optional[str]) -> Optional[str]:
            if not identifier:
                return identifier
            if identifier.startswith("\"") and identifier.endswith("\""):
                return identifier
            return f'"{identifier}"'
        
        cursor = self._connection.cursor()
        try:
            # Set role if specified
            if self.role:
                cursor.execute(f"USE ROLE {_quote(self.role)}")
                print(f"[INFO] Using role: {self.role}")
            
            # Set warehouse if specified
            if self.warehouse:
                cursor.execute(f"USE WAREHOUSE {_quote(self.warehouse)}")
                print(f"[INFO] Using warehouse: {self.warehouse}")
            
            # Use the specified database if provided
            if self.database:
                cursor.execute(f"USE DATABASE {_quote(self.database)}")
                print(f"[INFO] Using database: {self.database}")
        finally:
            cursor.close()
    
    def _is_connected(self) -> bool:
        """Check if the connection is still alive.
        
        Returns:
            True if connection is active, False otherwise
        """
        if not self._connection:
            return False
        
        try:
            # Attempt a simple query to check connection health
            cursor = self._connection.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            return True
        except Exception:
            return False
    
    def close(self):
        """Close the Snowflake connection.
        
        This should be called when the runner is no longer needed to free resources.
        """
        if self._connection:
            try:
                print("[INFO] Closing Snowflake connection...")
                self._connection.close()
            except Exception as e:
                print(f"[WARNING] Error closing connection: {e}")
            finally:
                self._connection = None
    
    def __del__(self):
        """Cleanup connection when the object is destroyed."""
        self.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close connection."""
        self.close()
        return False

    async def run_sql(self, args: RunSqlToolArgs, context: ToolContext) -> pd.DataFrame:
        """Execute SQL query against Snowflake database and return results as DataFrame.
        
        This method reuses the persistent connection established in __init__.
        It includes connection health checks and automatic reconnection if needed.

        Args:
            args: SQL query arguments containing the SQL to execute
            context: Tool execution context

        Returns:
            DataFrame with query results

        Raises:
            snowflake.connector.errors.ProgrammingError: If SQL syntax is invalid
            snowflake.connector.Error: If query execution fails after retries
        """
        max_retries = 1  # Retry once if connection fails
        
        # Use lock to ensure thread-safe execution
        with self._connection_lock:
            for attempt in range(max_retries + 1):
                try:
                    # Check connection health and reconnect if needed
                    if not self._connection or not self._is_connected():
                        if attempt == 0:
                            print("[INFO] Connection lost or invalid, reconnecting to Snowflake...")
                        self._connect()
                    
                    # Execute query using the persistent connection
                    cursor = self._connection.cursor()
                    
                    try:
                        # Execute the query
                        cursor.execute(args.sql)
                        results = cursor.fetchall()
                        
                        # Create a pandas dataframe from the results
                        if results and cursor.description:
                            df = pd.DataFrame(results, columns=[desc[0] for desc in cursor.description])
                        else:
                            # Empty result set
                            df = pd.DataFrame()
                        
                        return df
                    
                    finally:
                        # Always close cursor (but NOT the connection - we reuse it)
                        cursor.close()
                
                except self.snowflake.errors.ProgrammingError as e:
                    # SQL syntax error or invalid query - don't retry
                    print(f"[ERROR] SQL error: {e}")
                    raise
                
                except Exception as e:
                    # Connection or other error
                    if attempt < max_retries:
                        print(f"[WARNING] Query failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                        print("[INFO] Will retry after reconnecting...")
                        # Force reconnection on next attempt
                        self._connection = None
                    else:
                        # Final attempt failed
                        print(f"[ERROR] Query failed after {max_retries + 1} attempts: {e}")
                        raise
