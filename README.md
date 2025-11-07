# Promo Analytics Assistant

A conversational AI assistant powered by Google Gemini and Vanna that answers questions about promotional campaigns, pricing, and store performance by querying Snowflake databases using natural language.

## 🎯 Features

- **Natural Language Queries**: Ask questions in plain English about your promotional data
- **Intelligent Schema Discovery**: Automatically discovers relevant tables and columns using Excel knowledge base
- **SQL Generation**: Generates and executes SQL queries against Snowflake
- **Chat Interface**: Beautiful, business-friendly chat UI built with Streamlit
- **Safety First**: Read-only database access with SQL validation
- **RAG-Powered**: Uses Retrieval-Augmented Generation for accurate schema understanding

## 📋 Prerequisites

- Python 3.10 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))
- Snowflake account and credentials (optional - can run in logging mode)
- Excel knowledge base file (`Promo Semantic Data_20250519.xlsx`)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd vanna
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install vanna[gemini,snowflake]>=0.7.0
pip install pandas>=2.0.0
pip install openpyxl>=3.1.0
pip install streamlit>=1.28.0
pip install snowflake-connector-python>=3.0.0
pip install google-generativeai>=0.3.0
pip install python-dotenv>=1.0.0
```

### 3. Set Environment Variables

#### Windows PowerShell

```powershell
# Required: Google Gemini API Key
$env:GOOGLE_API_KEY="your-gemini-api-key-here"

# Optional: Excel file path (defaults to "Promo Semantic Data_20250519.xlsx")
$env:EXCEL_PATH="Promo Semantic Data_20250519.xlsx"

# Optional: Snowflake Configuration (if you want live SQL execution)
$env:SNOWFLAKE_ACCOUNT="your-snowflake-account"
$env:SNOWFLAKE_USER="your-username@example.com"
$env:SNOWFLAKE_WAREHOUSE="your-warehouse-name"
$env:SNOWFLAKE_ROLE="your-role-name"
$env:SNOWFLAKE_AUTHENTICATOR="externalbrowser"
$env:SNOWFLAKE_PASSWORD="your-password"  # Only if not using external browser auth

# Optional: Chat user email
$env:CHAT_USER_EMAIL="user@example.com"
```

#### Windows Command Prompt (CMD)

```cmd
set GOOGLE_API_KEY=your-gemini-api-key-here
set EXCEL_PATH=Promo Semantic Data_20250519.xlsx
set SNOWFLAKE_ACCOUNT=your-snowflake-account
set SNOWFLAKE_USER=your-username@example.com
set SNOWFLAKE_WAREHOUSE=your-warehouse-name
```

#### Linux/Mac (Bash)

```bash
export GOOGLE_API_KEY="your-gemini-api-key-here"
export EXCEL_PATH="Promo Semantic Data_20250519.xlsx"
export SNOWFLAKE_ACCOUNT="your-snowflake-account"
export SNOWFLAKE_USER="your-username@example.com"
export SNOWFLAKE_WAREHOUSE="your-warehouse-name"
```

#### Using .env File (Recommended)

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your-gemini-api-key-here
EXCEL_PATH=Promo Semantic Data_20250519.xlsx
SNOWFLAKE_ACCOUNT=your-snowflake-account
SNOWFLAKE_USER=your-username@example.com
SNOWFLAKE_WAREHOUSE=your-warehouse-name
SNOWFLAKE_ROLE=your-role-name
SNOWFLAKE_AUTHENTICATOR=externalbrowser
CHAT_USER_EMAIL=user@example.com
```

The application will automatically load these variables if you have `python-dotenv` installed.

### 4. Run the Streamlit Application

```bash
streamlit run streamlit_app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📖 Usage

### Starting the Application

1. **Start Streamlit**:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Open Browser**: The app will automatically open, or navigate to `http://localhost:8501`

3. **Ask Questions**: Type your question in the chat interface, for example:
   - "What is the performance of the promotional campaign with AD_BLK_NBR 45737?"
   - "What is the average sale price during promotions for SKU 98765?"
   - "Show me promotional activity for stores in the Northeast region"

### Running Without Snowflake (Logging Mode)

If you don't set Snowflake environment variables, the application will run in **logging mode**:
- SQL queries will be generated but not executed
- Queries will be printed to console/logs
- Useful for testing and development

To enable logging mode, simply don't set the Snowflake environment variables.

### Running the Example Script

You can also run the example script directly:

```bash
# Set PYTHONPATH to include the vanna source code
$env:PYTHONPATH="vanna_repo/src"

# Run the example
python vanna_repo/src/vanna/examples/gemini_excel_schema_example.py "Promo Semantic Data_20250519.xlsx"
```

## ⚙️ Configuration

### Environment Variables Reference

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `GOOGLE_API_KEY` | ✅ Yes | Google Gemini API key | - |
| `EXCEL_PATH` | ❌ No | Path to Excel knowledge base file | `Promo Semantic Data_20250519.xlsx` |
| `SNOWFLAKE_ACCOUNT` | ❌ No | Snowflake account identifier | - |
| `SNOWFLAKE_USER` | ❌ No | Snowflake username | - |
| `SNOWFLAKE_WAREHOUSE` | ❌ No | Snowflake warehouse name | - |
| `SNOWFLAKE_ROLE` | ❌ No | Snowflake role | - |
| `SNOWFLAKE_AUTHENTICATOR` | ❌ No | Authentication method | `externalbrowser` |
| `SNOWFLAKE_PASSWORD` | ❌ No | Password (if not using external browser) | - |
| `CHAT_USER_EMAIL` | ❌ No | User email for chat interface | `demo@example.com` |

### Hardcoded Configuration

If environment variables are not set, the application uses hardcoded Snowflake credentials (see `streamlit_app.py`). **For production, always use environment variables.**

## 🏗️ Architecture

The application uses a multi-layered architecture:

1. **Streamlit UI Layer**: User interface and response display
2. **Agent Orchestration Layer**: Coordinates LLM, tools, and conversation
3. **LLM Service Layer**: Google Gemini API integration
4. **Tool Registry Layer**: RAG tools (schema search) + SQL execution
5. **Data & Schema Layer**: Excel knowledge base + Snowflake database

### Key Components

- **Agent**: Orchestrates the entire conversation flow
- **Gemini LLM**: Generates SQL queries and natural language responses
- **RAG Tools**: Provide on-demand schema information from Excel
- **SQL Execution**: Safely executes queries against Snowflake
- **Response Cleaning**: Filters technical details for business users

## 📁 Project Structure

```
vanna/
├── streamlit_app.py              # Main Streamlit application
├── requirements.txt               # Python dependencies
├── Promo Semantic Data_20250519.xlsx  # Excel knowledge base
├── vanna_repo/                    # Vanna framework source code
│   └── src/
│       └── vanna/
│           ├── examples/
│           │   └── gemini_excel_schema_example.py
│           ├── schema/            # Schema provider and tools
│           ├── integrations/      # LLM and database integrations
│           └── tools/             # SQL execution tools
└── README.md                      # This file
```

## 🔧 Troubleshooting

### Issue: "Gemini API key required"

**Solution**: Set the `GOOGLE_API_KEY` environment variable:
```powershell
$env:GOOGLE_API_KEY="your-api-key"
```

### Issue: "Excel file not found"

**Solution**: Ensure the Excel file exists and set `EXCEL_PATH`:
```powershell
$env:EXCEL_PATH="path/to/Promo Semantic Data_20250519.xlsx"
```

### Issue: "Snowflake connection failed"

**Solutions**:
1. Check your Snowflake credentials
2. Ensure `snowflake-connector-python` is installed: `pip install snowflake-connector-python`
3. For external browser auth, ensure you complete the browser authentication
4. Check network connectivity to Snowflake

### Issue: "Module not found: vanna"

**Solution**: Ensure you're running from the correct directory and PYTHONPATH is set:
```powershell
$env:PYTHONPATH="vanna_repo/src"
```

### Issue: Streamlit shows raw HTML

**Solution**: This is usually a caching issue. Clear Streamlit cache:
```bash
streamlit cache clear
```

Or restart the Streamlit server.

### Issue: Infinite loop in chat

**Solution**: Clear the conversation using the "Clear Chat" button, or restart the application.

## 🔐 Security Notes

- **Read-Only Access**: The application only executes SELECT queries
- **SQL Validation**: All queries are validated before execution
- **No Data Modification**: UPDATE, INSERT, DELETE operations are blocked
- **Environment Variables**: Never commit API keys or credentials to version control

## 📚 Additional Resources

- [Vanna Documentation](https://github.com/vanna-ai/vanna)
- [Google Gemini API Docs](https://ai.google.dev/docs)
- [Snowflake Documentation](https://docs.snowflake.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

See the LICENSE file in the repository for details.

## 🆘 Support

For issues and questions:
1. Check the Troubleshooting section above
2. Review the example script: `vanna_repo/src/vanna/examples/gemini_excel_schema_example.py`
3. Check application logs in the console/terminal

## 🎯 Example Queries

Try these example queries once the application is running:

- "What is the performance of the promotional campaign with AD_BLK_NBR 45737?"
- "Show me the average sale price during promotions for SKU 98765"
- "Which promotion types are most common for store 123?"
- "List the top 5 stores by number of promotions in Q1 2025"
- "What promotional activity occurred in the Northeast region?"

---

**Happy Querying! 🚀**

