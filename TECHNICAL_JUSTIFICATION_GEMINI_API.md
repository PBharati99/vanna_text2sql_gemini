# Technical Justification: Gemini API Key vs. Vertex AI Integration

**Document Version:** 1.0  
**Date:** January 2025  
**Project:** Vanna Text-to-SQL Pipeline with Gemini  
**Prepared For:** Management Review

---

## Executive Summary

This document provides technical justification for why the Vanna text-to-SQL pipeline requires a **Gemini API key** and why **Vertex AI integration via Cloud SDK/gcloud** is **not currently feasible** for this project. The limitation stems from the Vanna open-source framework's architecture and available integrations.

**Key Finding:** Vanna's open-source framework only supports Gemini through the `google-generativeai` Python SDK, which exclusively uses API key authentication. There is no Vertex AI adapter in the current Vanna codebase.

---

## 1. Current Architecture Overview

### 1.1 LLM Integration Layer

The Vanna framework uses a **plugin-based architecture** for LLM integrations:

```
Vanna Core (LlmService Interface)
    ↓
[Plugin Layer - LLM Adapters]
    ↓
Available Integrations:
- GeminiLlmService (google-generativeai SDK)
- OpenAILlmService (OpenAI SDK)
- AnthropicLlmService (Anthropic SDK)
- OllamaLlmService (Ollama SDK)
```

### 1.2 Gemini Integration Implementation

**File:** `vanna_repo/src/vanna/integrations/gemini/llm.py`

**Key Code:**
```python
class GeminiLlmService(LlmService):
    def __init__(self, api_key: Optional[str] = None, ...):
        try:
            import google.generativeai as genai
            self.genai = genai
        except Exception as e:
            raise ImportError(
                "google-generativeai package is required. "
                "Install with: pip install 'vanna[gemini]'"
            )
        
        # Get API key from parameter or environment variables
        api_key = (
            api_key
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )
        
        if not api_key:
            raise ValueError(
                "API key is required. Set GOOGLE_API_KEY or "
                "GEMINI_API_KEY environment variable"
            )
        
        # Configure Gemini with API key
        genai.configure(api_key=api_key)
```

**Critical Observation:** The `GeminiLlmService` class is hardcoded to use `genai.configure(api_key=api_key)`, which is the API key authentication method from the `google-generativeai` SDK.

---

## 2. Why Vertex AI Cannot Be Used

### 2.1 SDK Differences

| Aspect | Gemini API (google-generativeai) | Vertex AI (google-cloud-aiplatform) |
|--------|----------------------------------|-------------------------------------|
| **Package** | `google-generativeai` | `google-cloud-aiplatform` |
| **Authentication** | API Key | Service Account / Application Default Credentials (ADC) / OAuth |
| **Configuration** | `genai.configure(api_key="...")` | `aiplatform.init(project="...", location="...")` |
| **Model Initialization** | `genai.GenerativeModel(model_name)` | `GenerativeModel.from_pretrained(model_name)` |
| **Endpoint** | `generativelanguage.googleapis.com` | `{location}-aiplatform.googleapis.com` |
| **Billing** | Per-request (consumer API) | GCP project billing |
| **Use Case** | Developer/prototyping | Enterprise/production |

### 2.2 Code Incompatibility

The Vanna `GeminiLlmService` integration calls methods specific to the `google-generativeai` SDK:

**Current Implementation (Lines 112-138):**
```python
# Creates model using google-generativeai SDK
model = self.genai.GenerativeModel(
    self._base_model_name,
    system_instruction=system_instruction
)

# Calls generate_content from google-generativeai
resp = model.generate_content(
    contents=contents,
    generation_config=generation_config,
    tools=tools
)
```

**What Vertex AI Would Require:**
```python
# Different package and initialization
from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerativeModel

# Initialize with project/location
aiplatform.init(project="cvs-project-id", location="us-central1")

# Different model instantiation
model = GenerativeModel.from_pretrained("gemini-1.5-pro")

# Different method signatures
response = model.generate_content(
    contents=contents,
    generation_config=generation_config,
    tools=tools,
    # Additional Vertex AI-specific parameters
)
```

**Result:** These are **fundamentally different APIs** with incompatible method signatures, initialization patterns, and authentication mechanisms.

---

## 3. Technical Barriers to Vertex AI Integration

### 3.1 Vanna Framework Limitation

**Issue:** Vanna's open-source codebase does **not include a Vertex AI integration**.

**Evidence:**
- Codebase search for `vertex`, `VertexAI`, `aiplatform`, `gcloud`: **No matches found**
- Only available LLM integrations in Vanna v0.7.0+:
  - `GeminiLlmService` (API key-based)
  - `OpenAILlmService`
  - `AnthropicLlmService`
  - `OllamaLlmService` (local models)
  - `MockLlmService` (testing)

**Conclusion:** There is no Vertex AI adapter in the current Vanna open-source framework.

### 3.2 Authentication Architecture

The `GeminiLlmService` class is tightly coupled to API key authentication:

**Line 67:**
```python
genai.configure(api_key=api_key)
```

This method **only accepts API keys**. The `google-generativeai` SDK does not support:
- Service account JSON files
- Application Default Credentials (ADC)
- gcloud CLI authentication
- Workload Identity Federation
- OAuth 2.0 tokens

**Why This Matters:** Even if we authenticate via `gcloud auth login`, the `genai.configure()` method cannot utilize those credentials. It strictly requires an API key string.

### 3.3 Model Endpoint Differences

**Gemini API Endpoint:**
```
https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent
```

**Vertex AI Endpoint:**
```
https://us-central1-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/gemini-1.5-pro:generateContent
```

**Impact:** These are different services with different:
- URL structures
- Request/response formats
- Rate limits
- Pricing models
- Authentication requirements

---

## 4. Why Custom Integration Is Not Feasible

### 4.1 Effort Required

To integrate Vertex AI into this pipeline would require:

1. **Create a new Vertex AI adapter class** (`VertexAILlmService`)
   - Implement `LlmService` interface (~500-700 lines of code)
   - Handle Vertex AI-specific authentication
   - Translate between Vanna's message format and Vertex AI's format
   - Implement streaming support
   - Handle tool calling with Vertex AI's format
   - Error handling and retry logic

2. **Modify Vanna core** to register the new integration
   - Update `vanna_repo/src/vanna/integrations/` directory
   - Create `__init__.py` and adapter files
   - Update package dependencies

3. **Testing and validation**
   - Unit tests for the adapter
   - Integration tests with Vertex AI
   - Tool calling tests (search_schema, run_sql, etc.)
   - Error scenario testing

4. **Maintenance burden**
   - Keep adapter updated with Vanna core changes
   - Track Vertex AI API changes
   - Debug issues specific to custom integration

**Estimated Effort:** 2-3 weeks of development + ongoing maintenance

### 4.2 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| **API Drift** | High - Vertex AI API changes could break integration | Medium |
| **Vanna Updates** | High - Core Vanna changes could break custom adapter | High |
| **Debugging Complexity** | Medium - Issues could be in Vanna, adapter, or Vertex AI | High |
| **Lack of Community Support** | High - No community help for custom adapter | Certain |
| **Testing Coverage** | Medium - Difficult to test all edge cases | Medium |

### 4.3 Opportunity Cost

**Time spent building custom adapter = Time NOT spent on:**
- Business logic improvements
- Data quality enhancements
- User experience refinements
- Excel schema enrichment
- Performance optimizations
- New features

---

## 5. Comparison: Gemini API vs. Vertex AI

### 5.1 Feature Parity for This Use Case

| Feature | Gemini API | Vertex AI | Impact |
|---------|-----------|-----------|--------|
| **Model Access** | ✅ gemini-1.5-pro, gemini-2.0-flash | ✅ Same models | None |
| **Function Calling** | ✅ Supported | ✅ Supported | None |
| **System Instructions** | ✅ Supported | ✅ Supported | None |
| **Streaming** | ✅ Supported | ✅ Supported | None |
| **Context Window** | ✅ 1M+ tokens | ✅ 1M+ tokens | None |
| **Rate Limits** | 60 RPM (free), higher (paid) | Project-level quotas | Minimal |
| **Response Quality** | ✅ Production-grade | ✅ Production-grade | None |

**Conclusion:** For this pipeline's requirements, Gemini API and Vertex AI provide **functionally equivalent capabilities**.

### 5.2 Cost Comparison

**Gemini API Pricing (as of Jan 2025):**
- Input: $3.50 per 1M tokens (gemini-1.5-pro)
- Output: $10.50 per 1M tokens
- Free tier: 60 requests/minute

**Vertex AI Pricing:**
- Input: $3.50 per 1M tokens (gemini-1.5-pro)
- Output: $10.50 per 1M tokens
- No free tier, but GCP committed use discounts available

**Usage Estimate (per user query):**
- System prompt: ~5,000 tokens
- Retrieved schema context: ~3,000 tokens
- User query: ~50 tokens
- LLM response: ~500 tokens
- **Total per query: ~8,550 tokens**

**Monthly Cost Estimate (1000 queries/month):**
- Input: 8,000 tokens × 1000 × $3.50/1M = **$0.028**
- Output: 500 tokens × 1000 × $10.50/1M = **$0.0053**
- **Total: ~$0.03/month**

**Conclusion:** Cost difference is negligible for this use case (~$0.40/year). Authentication method choice should not be driven by cost.

---

## 6. Recommended Solution: Use Gemini API Key

### 6.1 Justification

**Primary Reason:** Vanna's open-source framework only supports Gemini via API key authentication through the `google-generativeai` SDK.

**Secondary Reasons:**
1. **Zero Development Effort** - Works out-of-the-box
2. **Community Support** - Standard Vanna integration with community backing
3. **Maintained by Vanna** - Updates and bug fixes included in Vanna releases
4. **Production-Ready** - Already tested and used in production by Vanna users
5. **Same Model Quality** - Identical Gemini models as Vertex AI
6. **Negligible Cost** - $0.03-0.05/month for expected usage

### 6.2 Security Considerations

**API Key Management:**
- Store in environment variables (`.env` file, not committed to Git)
- Use GCP Secret Manager for production deployments
- Rotate keys quarterly
- Restrict key permissions (if Google Console allows)

**Access Control:**
- API key should be accessible only to application runtime
- Use least-privilege principles
- Monitor API usage via Google AI Studio dashboard

**Alternative for Enterprise:** If API key management is a concern, consider:
- Google Cloud Secret Manager to store the key
- GCP service account with permissions to access Secret Manager
- Application fetches key at runtime from Secret Manager

This provides enterprise-grade security while still using the Gemini API.

---

## 7. Alternative Approaches (If Vertex AI is Mandatory)

If organizational policy **mandates** Vertex AI, here are the options:

### 7.1 Option A: Fork Vanna and Create Custom Integration
**Pros:**
- Full control over authentication
- Can use Vertex AI's ADC/service accounts
- Aligns with GCP-native architecture

**Cons:**
- 2-3 weeks development effort
- Ongoing maintenance burden
- Lose automatic Vanna updates
- No community support for custom code
- Risk of breaking changes

**Estimated Cost:** 120-160 hours of engineering time

### 7.2 Option B: Use Different Text-to-SQL Framework
**Alternative Frameworks:**
- **LangChain** - Has native Vertex AI support
- **LlamaIndex** - Supports Vertex AI integration
- **Custom Solution** - Build from scratch with Vertex AI

**Cons:**
- Lose Vanna's specialized text-to-SQL architecture
- Lose Excel-based schema provider
- Lose built-in RAG tooling (search_schema, table_info, etc.)
- Higher initial development effort
- Different tool ecosystem

**Estimated Cost:** 4-6 weeks to replicate current functionality

### 7.3 Option C: Proxy Layer
**Architecture:**
```
Vanna (Gemini API calls) → Proxy Service → Vertex AI
```

**How it works:**
- Create a proxy service that exposes Gemini API-compatible endpoints
- Internally, proxy translates calls to Vertex AI
- Vanna thinks it's talking to Gemini API
- Proxy authenticates to Vertex AI using ADC

**Cons:**
- Additional infrastructure to maintain
- Added latency (extra network hop)
- Complex debugging
- Single point of failure

**Estimated Cost:** 1-2 weeks + infrastructure costs

---

## 8. Comparison Matrix

| Approach | Dev Effort | Maintenance | Risk | Cost | Time to Deploy |
|----------|-----------|-------------|------|------|----------------|
| **Gemini API (Recommended)** | None | None | Low | $0.03/mo | Immediate |
| **Fork Vanna + Vertex AI** | 120h | High | High | $0.03/mo + eng. time | 2-3 weeks |
| **Different Framework** | 200h | Medium | High | $0.03/mo + eng. time | 4-6 weeks |
| **Proxy Layer** | 80h | Medium | Medium | $0.03/mo + infra | 1-2 weeks |

---

## 9. Addressing Common Concerns

### 9.1 "We want to use gcloud authentication for security"

**Response:** API keys can be stored in GCP Secret Manager and accessed via service accounts with gcloud authentication. The application authenticates to Secret Manager using ADC, then retrieves the Gemini API key at runtime.

**Implementation:**
```python
from google.cloud import secretmanager

def get_gemini_api_key():
    client = secretmanager.SecretManagerServiceClient()  # Uses ADC
    name = "projects/PROJECT_ID/secrets/gemini-api-key/versions/latest"
    response = client.access_secret_version(name=name)
    return response.payload.data.decode('UTF-8')

# Use in Vanna
api_key = get_gemini_api_key()
agent = create_excel_schema_agent(gemini_api_key=api_key, ...)
```

This gives you:
- ✅ gcloud/ADC authentication to Secret Manager
- ✅ Centralized key management
- ✅ Audit logs via Cloud Logging
- ✅ Works with existing Vanna integration

### 9.2 "Vertex AI is our standard for LLMs"

**Response:** Standards should serve business needs, not constrain them. When using a third-party framework (Vanna), we must work within its supported integrations. The framework provides significant value (RAG architecture, Excel schema provider, tool ecosystem) that justifies using its supported authentication method.

**Analogy:** If we use Terraform (open source), we use its authentication methods for cloud providers, even if they differ from our internal standards. The value Terraform provides outweighs the need to customize its authentication.

### 9.3 "What if Google deprecates Gemini API?"

**Response:** 
1. Google has not announced any deprecation plans for Gemini API
2. Gemini API is the **public developer interface** for Google's AI models
3. If deprecated, Google would provide migration path (likely TO Vertex AI)
4. At that point, Vanna community would also need to migrate, likely resulting in an official Vertex AI integration
5. We would benefit from community-developed migration, not custom code

---

## 10. Technical Recommendations

### 10.1 Immediate Action (Recommended)
✅ **Use Gemini API Key with Secret Manager**

**Implementation Steps:**
1. Generate Gemini API key in Google AI Studio
2. Store in GCP Secret Manager
3. Grant service account `secretmanager.secretAccessor` role
4. Application retrieves key at runtime using ADC
5. Pass key to Vanna's `GeminiLlmService`

**Timeline:** 1-2 hours  
**Risk:** Low  
**Cost:** $0.03/month + negligible Secret Manager cost

### 10.2 Future Consideration
Monitor Vanna's GitHub repository for Vertex AI integration:
- Repository: `https://github.com/vanna-ai/vanna`
- Watch for issues/PRs related to Vertex AI support
- If community develops Vertex AI integration, evaluate migration

### 10.3 Documentation Requirements
Document the following for compliance:
- Why API key is used (framework limitation)
- How API key is secured (Secret Manager + ADC)
- Usage monitoring (via Google AI Studio)
- Key rotation schedule (quarterly)
- Access control (service account only)

---

## 11. Conclusion

**The requirement for a Gemini API key is a technical constraint of the Vanna open-source framework, not a design choice.**

Vanna's `GeminiLlmService` integration is built on the `google-generativeai` Python SDK, which exclusively supports API key authentication. There is no Vertex AI integration in the current Vanna codebase, and building a custom one would require significant development effort (2-3 weeks) plus ongoing maintenance.

**For this project:**
- **Functionally:** Gemini API provides identical capabilities to Vertex AI
- **Cost:** Negligible difference ($0.03-0.05/month)
- **Security:** Can be addressed with Secret Manager + ADC
- **Risk:** Using standard Vanna integration is lower risk than custom code
- **Effort:** Zero development vs. 2-3 weeks for custom integration

**Recommendation:** Proceed with Gemini API key stored in GCP Secret Manager, accessed via service account authentication. This balances framework compatibility with enterprise security practices.

---

## 12. References

- **Vanna Documentation:** https://vanna.ai/docs/
- **Vanna GitHub (Gemini Integration):** https://github.com/vanna-ai/vanna/tree/main/src/vanna/integrations/gemini
- **Google Generative AI Python SDK:** https://github.com/google/generative-ai-python
- **Vertex AI Python SDK:** https://cloud.google.com/vertex-ai/docs/python-sdk/use-vertex-ai-python-sdk
- **Gemini API Documentation:** https://ai.google.dev/docs
- **Vertex AI Documentation:** https://cloud.google.com/vertex-ai/docs

---

## Appendix A: Code Evidence

**File:** `vanna_repo/src/vanna/integrations/gemini/llm.py`

**Line 46-67 (Authentication Logic):**
```python
try:
    import google.generativeai as genai
    self.genai = genai
except Exception as e:
    raise ImportError(
        "google-generativeai package is required. "
        "Install with: pip install 'vanna[gemini]'"
    )

# Get API key from parameter, environment variables, or .env file
api_key = (
    api_key
    or os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
)

if not api_key:
    raise ValueError(
        "API key is required. Set GOOGLE_API_KEY or GEMINI_API_KEY "
        "environment variable, or pass api_key parameter."
    )

# Configure Gemini
genai.configure(api_key=api_key)
```

**Key Observation:** The code explicitly requires an API key and uses `genai.configure(api_key=api_key)`, which is the only authentication method supported by the `google-generativeai` SDK.

---

## Appendix B: Proposed Secret Manager Implementation

```python
# File: config.py
from google.cloud import secretmanager
import os

def get_gemini_api_key_from_secret_manager():
    """
    Retrieve Gemini API key from GCP Secret Manager using ADC.
    
    Prerequisites:
    - Service account with 'secretmanager.secretAccessor' role
    - Secret created in Secret Manager: 'gemini-api-key'
    - Application authenticated via gcloud or service account JSON
    """
    try:
        # Initialize Secret Manager client (uses Application Default Credentials)
        client = secretmanager.SecretManagerServiceClient()
        
        # Get project ID from environment or metadata server
        project_id = os.getenv("GCP_PROJECT_ID", "cvs-analytics-prod")
        
        # Build secret name
        secret_name = f"projects/{project_id}/secrets/gemini-api-key/versions/latest"
        
        # Access secret
        response = client.access_secret_version(name=secret_name)
        api_key = response.payload.data.decode('UTF-8')
        
        print(f"[INFO] Successfully retrieved Gemini API key from Secret Manager")
        return api_key
        
    except Exception as e:
        print(f"[ERROR] Failed to retrieve API key from Secret Manager: {e}")
        print(f"[INFO] Falling back to environment variable")
        return os.getenv("GEMINI_API_KEY")

# Usage in application
if __name__ == "__main__":
    # This uses gcloud authentication to access Secret Manager
    api_key = get_gemini_api_key_from_secret_manager()
    
    # Pass to Vanna (which still uses API key, but we got it securely)
    agent = create_excel_schema_agent(
        excel_path="schema.xlsx",
        gemini_api_key=api_key,
        snowflake_config={...}
    )
```

**Benefits of This Approach:**
- ✅ Uses gcloud/ADC authentication to access Secret Manager
- ✅ Centralized key management in GCP
- ✅ Audit trail via Cloud Logging
- ✅ No code changes needed in Vanna
- ✅ Key rotation handled in Secret Manager
- ✅ Works with existing Gemini API integration

---

**End of Document**

