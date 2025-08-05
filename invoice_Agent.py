import os
import pandas as pd
import google.generativeai as genai
import json
import tempfile
from pathlib import Path
import io
from typing import Dict, Any, Tuple, Optional, List
import base64
import re

# --- Tool Utility Functions ---
def _lookup_master_data(
    file_key: str,
    lookup_column: str,
    lookup_value: str,
    return_column: str,
    master_data_dfs: Dict[str, pd.DataFrame],
) -> str:
    """Looks up a value in a loaded master data DataFrame."""
    try:
        if file_key not in master_data_dfs:
            return f"Master data key '{file_key}' not found. Available keys: {list(master_data_dfs.keys())}"

        df = master_data_dfs[file_key]
        lookup_column = lookup_column.strip()
        return_column = return_column.strip()

        if lookup_column not in df.columns:
            return f"Column '{lookup_column}' not found in {file_key}. Available: {list(df.columns)}"
        if return_column not in df.columns:
            return f"Column '{return_column}' not found in {file_key}. Available: {list(df.columns)}"

        result = df[df[lookup_column].astype(str).str.contains(lookup_value, case=False, na=False)]

        if not result.empty:
            found_value = result.iloc[0][return_column]
            return str(found_value)
        else:
            return f"Value '{lookup_value}' not found in {file_key} column '{lookup_column}'"
    except Exception as e:
        return f"Error during lookup in '{file_key}': {e}"

def _generate_output_csv(invoice_data_json: str) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
    """Generates CSV string and JSON object from JSON string."""
    try:
        data = json.loads(invoice_data_json)
        df = pd.DataFrame([data])
        csv_string = df.to_csv(index=False)
        return "Successfully generated CSV data", csv_string, data
    except Exception as e:
        return f"Error generating CSV: {e}", None, None

def _load_master_data_from_bytes(master_data_bytes_dict: Dict[str, bytes]) -> Dict[str, pd.DataFrame]:
    """Loads master data from a dictionary of file names to bytes content."""
    master_data = {}
    for file_name, file_bytes in master_data_bytes_dict.items():
        try:
            excel_file = pd.ExcelFile(file_bytes)
            if len(excel_file.sheet_names) == 1:
                df = pd.read_excel(excel_file, sheet_name=0)
                df.columns = df.columns.str.strip()
                master_data[file_name] = df
            else:
                for sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    df.columns = df.columns.str.strip()
                    key = f"{file_name}_{sheet_name}"
                    master_data[key] = df
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
    return master_data

# --- Agent Class for Vertex AI Agent Engine ---
class InvoiceProcessorAgent:
    def __init__(self):
        """Initializes the agent, configures API key."""
        try:
            self.api_key = os.environ.get("GOOGLE_API_KEY")
            if not self.api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set.")
            genai.configure(api_key=self.api_key)
            self.model_name = "gemini-1.5-pro"
            self.model = genai.GenerativeModel(model_name=self.model_name)
            print("InvoiceProcessorAgent initialized and Gemini configured.")
        except Exception as e:
            print(f"Error during agent initialization: {e}")
            raise

    def _get_master_data_structure(self, master_data_dfs: Dict[str, pd.DataFrame]) -> List[str]:
        """Gets the list of master data keys (file/sheet names)."""
        return list(master_data_dfs.keys())

    def query(
        self,
        invoice_b64: str,
        jira_b64: str,
        master_data_b64: Dict[str, str],
        contract_b64: Optional[str] = None,
        selected_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Processes invoice documents and master data to extract information and generate CSV.
        This method is called by ReasoningEngine.query().

        Args:
            invoice_b64: Base64 encoded content of the Tax Invoice PDF.
            jira_b64: Base64 encoded content of the Jira Ticket PDF.
            master_data_b64: Dictionary where keys are filenames
                                     and values are base64 encoded content of the Excel files.
            contract_b64: Optional base64 encoded content of the Contract PDF.
            selected_model: Optional Gemini model name to use for this query.

        Returns:
            A dictionary containing results, including 'csv_data', 'json_data',
            'tool_calls', and 'raw_response'.
        """
        print("Starting agent query...")
        try:
            invoice_bytes = base64.b64decode(invoice_b64)
            jira_bytes = base64.b64decode(jira_b64)
            master_data_bytes_dict = {
                name: base64.b64decode(data)
                for name, data in master_data_b64.items()
            }
            contract_bytes = base64.b64decode(contract_b64) if contract_b64 else None
        except Exception as e:
            print(f"Error decoding base64 inputs: {e}")
            return {"error": f"Error decoding base64 inputs: {e}"}

        current_model = self.model
        current_model_name = self.model_name
        if selected_model and selected_model != self.model_name:
            try:
                current_model = genai.GenerativeModel(model_name=selected_model)
                current_model_name = selected_model
                print(f"Using selected model: {current_model_name}")
            except Exception as e:
                print(f"Failed to switch to model {selected_model}: {e}. Falling back to default.")

        try:
            master_data_dfs = _load_master_data_from_bytes(master_data_bytes_dict)
            if not master_data_dfs:
                return {"error": "No master data could be loaded."}
            print(f"Loaded master data keys: {list(master_data_dfs.keys())}")

            tool_results = []
            final_csv_data = None
            final_json_data = None

            def lookup_tool(file_key: str, lookup_column: str, lookup_value: str, return_column: str) -> str:
                result = _lookup_master_data(file_key, lookup_column, lookup_value, return_column, master_data_dfs)
                tool_results.append({
                    "tool": "lookup_tool",
                    "args": {"file_key": file_key, "lookup_column": lookup_column, "lookup_value": lookup_value, "return_column": return_column},
                    "result": result
                })
                print(f"lookup_tool call: {file_key} - {lookup_column}='{lookup_value}' -> {return_column} | Result: {result}")
                return result

            def generate_csv_tool(invoice_data_json: str) -> str:
                nonlocal final_csv_data, final_json_data
                message, csv_data, json_data = _generate_output_csv(invoice_data_json)
                tool_results.append({"tool": "generate_csv_tool", "args": {"invoice_data_json": "true"}, "result": message})
                if csv_data:
                    final_csv_data = csv_data
                    final_json_data = json_data
                    print("generate_csv_tool call: Success")
                else:
                    print(f"generate_csv_tool call: Failure - {message}")
                return message

            model_with_tools = genai.GenerativeModel(
                model_name=current_model_name,
                tools=[lookup_tool, generate_csv_tool]
            )
            chat = model_with_tools.start_chat(enable_automatic_function_calling=True)

            with tempfile.TemporaryDirectory() as temp_dir:
                invoice_path = Path(temp_dir) / "invoice.pdf"
                jira_path = Path(temp_dir) / "jira.pdf"
                invoice_path.write_bytes(invoice_bytes)
                jira_path.write_bytes(jira_bytes)

                uploaded_files = []
                try:
                    invoice_gemini_file = genai.upload_file(path=invoice_path)
                    uploaded_files.append(invoice_gemini_file)
                    print(f"Uploaded invoice: {invoice_gemini_file.display_name}")
                    jira_gemini_file = genai.upload_file(path=jira_path)
                    uploaded_files.append(jira_gemini_file)
                    print(f"Uploaded JIRA: {jira_gemini_file.display_name}")

                    documents_to_process = [invoice_gemini_file, jira_gemini_file]
                    contract_info_prompt = "No contract document is provided."

                    if contract_bytes:
                        contract_path = Path(temp_dir) / "contract.pdf"
                        contract_path.write_bytes(contract_bytes)
                        contract_gemini_file = genai.upload_file(path=contract_path)
                        uploaded_files.append(contract_gemini_file)
                        documents_to_process.append(contract_gemini_file)
                        contract_info_prompt = "A Contract document is also available for reference."
                        print(f"Uploaded contract: {contract_gemini_file.display_name}")

                    available_files = ", ".join([f"`{key}`" for key in self._get_master_data_structure(master_data_dfs)])

                    prompt = f"""
                    **ROLE & GOAL:**
                    You are an AI agent specializing in procurement data processing. Your task is to extract information from the provided documents (a Tax Invoice and a Jira Ticket{', and optionally a Contract' if contract_bytes else ''}), use the `lookup_tool` function to find corresponding codes from master data files, and then call the `generate_csv_tool` function with the final, complete data.

                    **AVAILABLE DOCUMENTS:**
                    - Tax Invoice PDF (required)
                    - Jira Ticket PDF (required)
                    {contract_info_prompt}

                    **AVAILABLE MASTER DATA KEYS:**
                    {available_files}

                    **NOTE:** Master data keys are in the format `filename.xlsx` for single-sheet files, or `filename.xlsx_SheetName` for multi-sheet files.

                    **CRITICAL REQUIREMENTS:**
                    1.  **Use VENDOR CODE, not vendor name** - Extract the vendor code from the invoice.
                    2.  **Format dates as DD.MM.YYYY** (e.g., 19.07.2025).
                    3.  **Use 18% GST Tax Code** - Look up the appropriate tax code for 18% GST from master data (Expect 'I4' for 18% input credit).
                    4.  **Generate purchase order number** - Create a PO number based on the invoice and workflow information.

                    **INSTRUCTIONS:**
                    1.  **Extract Data:** Carefully read all provided documents.
                    2.  **Enrich Data with Tools:** Use the `lookup_tool` function to find:
                        - GL Account (based on service description)
                        - Tax Code for 18% GST (should be I4)
                        - Requestor ID (based on name from Jira)
                        - Any other required codes from master files.
                    3.  **Construct Final Data:** Assemble data for procurement processing.
                    4.  **Generate CSV:** Call the `generate_csv_tool` function with the completed JSON data matching the schema.

                    **OUTPUT JSON SCHEMA (for generate_csv_tool):**
                    {{
                      "Document Type": "ZNID", "PO Number": "", "Line Item Number": "10", "Vendor": "", "Document Date": "",
                      "Payment Terms": "P000", "Purchasing Organisation": "1001", "Purchase Group": "S05", "Invoice": "",
                      "SAP Database": "", "Jira": "", "Agreement": "", "Company Code": "1001", "Validity Start Date": "",
                      "Validity End Date": "", "WO Header Text": "", "Account Assignment": "", "Item Category": "",
                      "Short Text": "", "Delivery Date": "", "Plant": "DS01", "Requisitioner": "", "Service Number": "",
                      "Service Quantity": "", "Gross Price": "", "Cost Center": "", "WBS": "", "Tax Code": "",
                      "Material Group": "", "no of days": "", "Requestor": "", "Control Code": "", "GL Account": "",
                      "UOM": "", "Order Number": "", "Text 1": ""
                    }}

                    **IMPORTANT NOTES:**
                    - Use VENDOR CODE (e.g., 101347) NOT vendor name.
                    - For 18% GST, ensure the Tax Code is 'I4'.
                    - Format dates as DD.MM.YYYY.

                    Begin the process now.
                    """
                    documents_to_process.insert(0, prompt)

                    print("Sending request to Gemini model...")
                    response = chat.send_message(documents_to_process)
                    print("Received response from Gemini model.")

                finally:
                    # Clean up uploaded files
                    for uf in uploaded_files:
                        try:
                            uf.delete()
                            print(f"Deleted uploaded file: {uf.display_name}")
                        except Exception as e:
                            print(f"Error deleting file {uf.display_name}: {e}")

            return {
                "message": "Processing complete.",
                "raw_response": response.text,
                "tool_calls": tool_results,
                "csv_data": final_csv_data,
                "json_data": final_json_data,
            }

        except Exception as e:
            import traceback
            print(f"Error during query: {e}\n{traceback.format_exc()}")
            return {"error": str(e)}
