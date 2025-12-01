import os
import json
import time
from typing import Dict, List, Any
from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent, Agent
from google.adk.tools import FunctionTool
from google.adk.sessions import InMemorySessionService
LLM_MODEL = "gemini-2.5-flash-preview-09-2025"
GOOGLE_API_KEY = "AIzaSyDGfC8cpl6deriuHgTkaahWO7_mhMc4BX4"

def search_clinical_literature(query: str, max_results: int = 5) -> List[Dict]:
    """
    [Tool 1] Simulates querying clinical databases (like PubMed) for recent papers.
    Returns a list of structured research summaries.
    Args:
        query: The medical condition or topic to search.
        max_results: The maximum number of results to return. (CRITICAL FIX: Added default value)
    """
    time.sleep(1) # Simulate network latency
    print(f"-> Tool Call: Executing Literature Search for: {query}")
    if "kidney disease" in query.lower():
        return [
            {"title": "Meta-analysis on Kidney Disease Drug X", "abstract": "Shows 15% efficacy increase...", "source": "JAMA 2024"},
            {"title": "Safety profile of Drug Y in CKD", "abstract": "Suggests dose adjustment needed...", "source": "Lancet 2025"}
        ]
    return [{"title": f"No definitive literature found for {query}", "source": "N/A"}]

def find_matching_trials(condition: str, location: str) -> List[Dict]:
    """
    [Tool 2] Simulates querying a trial registry for active enrollment.
    Returns a list of structured trial details.
    Args:
        condition: The medical condition or disease.
        location: The geographic location for the trials (e.g., 'US', 'Europe').
    """
    time.sleep(1) 
    print(f"-> Tool Call: Executing Trial Match for: {condition} in {location}")
    if "kidney disease" in condition.lower() and "us" in location.lower():
        return [
            {"NCT_ID": "NCT001", "phase": "Phase 3", "status": "Recruiting", "criteria_summary": "Patients with Stage 3-4 CKD."},
            {"NCT_ID": "NCT002", "phase": "Phase 2", "status": "Enrolling by referral", "criteria_summary": "Pediatric patients only."}
        ]
    return [{"NCT_ID": "N/A", "status": "None Found", "criteria_summary": "Try a broader search."}]

LITERATURE_TOOL = FunctionTool(search_clinical_literature)
TRIAL_MATCHING_TOOL = FunctionTool(find_matching_trials)
greeting_agent = LlmAgent(
    name="Greeting_Agent",
    model=LLM_MODEL,
    instruction=(
        "You are a friendly receptionist for the Clinical Data Synthesis System. "
        "Your only job is to respond to greetings, small talk, or requests for help by asking the user "
        "for their specific medical research query (e.g., condition and treatment)."
    ),
    description="Handles general greetings, small talk, and introductions for the user.",
    output_key="greeting_response",
)

query_parsing_agent = LlmAgent(
    name="Query_Parsing_Agent",
    model=LLM_MODEL,
    instruction=(
        "You are a Query Analyst. Your task is to extract the **condition** and the **location** from the user's request. "
        "You must output ONLY a JSON object with keys 'condition' and 'location'. "
        "For example: {'condition': 'Stage 3 Chronic Kidney Disease (CKD)', 'location': 'US'}"
        "Save the JSON result to the session state key 'parsed_query'."
    ),
    description="Parses the initial unstructured user query into structured condition and location parameters.",
    output_key="parsed_query",
)

literature_agent = LlmAgent(
    name="Literature_Search_Agent",
    model=LLM_MODEL,
    instruction=(
        "You are an expert medical librarian. Use the 'condition' value from the session state key 'parsed_query' to call the `search_clinical_literature` tool. "
        "Do NOT use the raw user query. Do NOT summarize or interpret the data. Save the result to the session state key 'literature_results'."
    ),
    tools=[LITERATURE_TOOL],
    output_key="literature_results",
)

trial_agent = LlmAgent(
    name="Trial_Matching_Agent",
    model=LLM_MODEL,
    instruction=(
        "You are a clinical trial recruitment specialist. Use the 'condition' and 'location' values from the session state key 'parsed_query' to call the `find_matching_trials` tool. "
        "Do NOT use the raw user query. Save the raw JSON result to the session state key 'trial_results'."
    ),
    tools=[TRIAL_MATCHING_TOOL],
    output_key="trial_results",
)

# Agent 3: Synthesis Agent (Revised Instruction)
synthesis_agent = LlmAgent(
    name="Synthesis_Agent",
    model=LLM_MODEL,
    instruction=(
        "You are a Senior Clinical Analyst. Your role is to take the raw data from the session state keys 'literature_results' "
        "and 'trial_results', perform a comparison, and write a final, comprehensive clinical brief. "
        "The output **MUST** strictly follow this Markdown format, using the data provided:\n\n"
        "## Clinical Data Synthesis Brief\n\n"
        "### 1. Key Literature Findings\n"
        "Provide a concise summary (2-3 bullet points) of the most relevant papers found. Note the source and publication year for each finding.\n\n"
        "### 2. Active Clinical Trials\n"
        "Present the matching trials in a clear Markdown table with the following columns: **NCT ID**, **Phase**, **Status**, and **Summary of Key Criteria**.\n\n"
        "### 3. Clinical Relevance and Alignment\n"
        "Analyze the findings. Discuss how the patient criteria in the literature (e.g., condition stage) aligns with the eligibility criteria of the active trials. Provide a final conclusion on the potential relevance of the trial data to the current standard of care."
    ),
    output_key="final_report_raw",
)

data_gathering_parallel = ParallelAgent(
    name="Data_Gathering_Parallel",
    sub_agents=[literature_agent, trial_agent],
    description="Runs the Literature and Trial search agents concurrently to gather raw data quickly.",
)

synthesis_pipeline = SequentialAgent(
    name="Synthesis_Pipeline",
    sub_agents=[
        query_parsing_agent,       
        data_gathering_parallel,    
        synthesis_agent             
    ],
    instruction="Execute query parsing, data gathering in parallel, then synthesize the results into a report."
)

""" The Orchestrator Agent: The single entry point for the ADK Web UI and CLI."""
"""This must be named 'root_agent'.The Orchestrator Agent: The single entry point for the ADK Web UI and CLI. This must be named 'root_agent'."""
# --- 3. ORCHESTRATION WORKFLOW (The Manager) ---

# ... [Keep synthesis_pipeline definition the same] ...

# The Orchestrator Agent: The single entry point for the ADK Web UI and CLI.
# This must be named 'root_agent'.
root_agent = Agent(
    name="Clinical_Data_Orchestrator",
    model=LLM_MODEL,
    instruction=(
        "You are the central manager for the Clinical Data Synthesis System. "
        "1. If the user asks a medical research question (e.g., 'Find X for Y'), execute the 'Synthesis_Pipeline' workflow. "
        "Once the pipeline completes, your ONLY task is to **output the contents of the session state key 'final_report_raw' as the final answer.** "
        "2. If the user asks a simple question or a greeting, delegate the request to the 'Greeting_Agent'."
    ),
    description="The main intelligent agent that handles delegation to specialists (Greeting or Synthesis).",
    sub_agents=[synthesis_pipeline, greeting_agent], # The orchestrator manages both the pipeline and the conversational agent
    session_service=InMemorySessionService(), # Required for managing state/memory
    
    # Keeping output_key set is still good practice.
    output_key="final_report_raw", 
)
if __name__ == "__main__":
    from google.adk.runners import Runner
    GOOGLE_API_KEY = "AIzaSyDGfC8cpl6deriuHgTkaahWO7_mhMc4BX4"
    
    if GOOGLE_API_KEY == "":
        print("🚨 SETUP WARNING: Please set your Gemini API key in the ADK environment.")
        print("--- Running in ADK CLI/Programmatic SIMULATION Mode ---")

    query = "Find recent literature and open US trials for patients with Stage 3 Chronic Kidney Disease (CKD)."
    print(f"\n[USER QUERY]: {query}")
    print("-----------------------------------------------------------------")

    try:
        runner = Runner(agent=root_agent, app_name="ClinicalSynthesizer")
        
        """NOTE: Since the real LLM/Tool integration is not active without a key,this will likely trace tool calls but not give the full final report yet."""
        print("Starting ADK Runner... (Check terminal output for tool calls)")
        
        """The actual ADK invocation line (commented out to prevent errors in non-ADK environments): response = runner.invoke(query) print("Final Response:", response.text)"""
        
    except Exception as e:
        print("\n=================================================================")
        print(f"❌ EXECUTION ERROR: {e}")
        print("=================================================================")
        print("If you see an AttributeError or ModuleNotFoundError, ensure you run this inside")
        print("the correct virtual environment with 'pip install google-adk'.")