import streamlit as st
import os
import pandas as pd
import google.generativeai as genai
import json
import tempfile
from pathlib import Path
import zipfile
from io import BytesIO
import openpyxl

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="PO Processing Agent",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Tool Definitions ---
def lookup_master_data(file_name: str, lookup_column: str, lookup_value: str, return_column: str, master_data_files: dict) -> str:
    """
    Looks up a value in a master data Excel file and returns a corresponding value.
    """
    try:
        if file_name not in master_data_files:
            return f"Master data file '{file_name}' not found. Available files: {list(master_data_files.keys())}"
        
        df = master_data_files[file_name]
        
        # Clean column names (remove extra spaces)
        df.columns = df.columns.str.strip()
        lookup_column = lookup_column.strip()
        return_column = return_column.strip()
        
        # Check if columns exist
        if lookup_column not in df.columns:
            return f"Column '{lookup_column}' not found in {file_name}. Available columns: {list(df.columns)}"
        
        if return_column not in df.columns:
            return f"Column '{return_column}' not found in {file_name}. Available columns: {list(df.columns)}"
        
        # Perform a case-insensitive search
        result = df[df[lookup_column].astype(str).str.contains(lookup_value, case=False, na=False)]

        if not result.empty:
            found_value = result.iloc[0][return_column]
            st.success(f"✅ Found '{found_value}' in '{file_name}' for lookup '{lookup_value}'")
            return str(found_value)
        else:
            st.warning(f"⚠️ Could not find '{lookup_value}' in '{file_name}' column '{lookup_column}'")
            return f"Value '{lookup_value}' not found in {file_name}"
    except Exception as e:
        st.error(f"❌ Error looking up data in '{file_name}': {e}")
        return f"Error: {e}"

def generate_output_csv(po_data_json: str, output_filename: str) -> tuple:
    """
    Generates a final CSV file from a JSON object of processed PO data.
    Returns tuple of (success_message, csv_data, dataframe)
    """
    try:
        data = json.loads(po_data_json)
        df = pd.DataFrame([data])
        
        # Convert to CSV string
        csv_string = df.to_csv(index=False)
        
        st.success(f"✅ Successfully generated CSV data for: {output_filename}")
        return f"Successfully generated CSV file: {output_filename}", csv_string, df
    except Exception as e:
        st.error(f"❌ Error generating CSV: {e}")
        return f"Error generating CSV: {e}", None, None

# --- Master Data File Processing ---
@st.cache_data
def load_master_data_files(uploaded_files):
    """Load and cache master data files (Excel format)"""
    master_data = {}
    for file in uploaded_files:
        try:
            # Read Excel file - handle multiple sheets
            excel_file = pd.ExcelFile(file)
            
            if len(excel_file.sheet_names) == 1:
                # Single sheet - use file name as key
                df = pd.read_excel(file, sheet_name=0)
                # Clean column names
                df.columns = df.columns.str.strip()
                master_data[file.name] = df
                st.success(f"✅ Loaded master data file: {file.name} ({len(df)} rows)")
            else:
                # Multiple sheets - use filename_sheetname as key
                for sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(file, sheet_name=sheet_name)
                    # Clean column names
                    df.columns = df.columns.str.strip()
                    key = f"{file.name}_{sheet_name}"
                    master_data[key] = df
                    st.success(f"✅ Loaded sheet '{sheet_name}' from {file.name} ({len(df)} rows)")
                    
        except Exception as e:
            st.error(f"❌ Error loading {file.name}: {e}")
    return master_data

@st.cache_data
def get_master_data_info(uploaded_files):
    """Get information about uploaded Excel files and their sheets"""
    file_info = {}
    for file in uploaded_files:
        try:
            excel_file = pd.ExcelFile(file)
            file_info[file.name] = {
                'sheets': excel_file.sheet_names,
                'total_sheets': len(excel_file.sheet_names)
            }
        except Exception as e:
            st.error(f"❌ Error reading {file.name}: {e}")
    return file_info

@st.cache_data
def preview_excel_structure(uploaded_files):
    """Preview the structure of uploaded Excel files"""
    structure_info = {}
    for file in uploaded_files:
        try:
            excel_file = pd.ExcelFile(file)
            structure_info[file.name] = {}
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file, sheet_name=sheet_name, nrows=0)  # Just get column names
                df.columns = df.columns.str.strip()  # Clean column names
                structure_info[file.name][sheet_name] = {
                    'columns': list(df.columns),
                    'total_columns': len(df.columns)
                }
        except Exception as e:
            st.error(f"Error analyzing {file.name}: {e}")
    
    return structure_info

# --- Main Streamlit App ---
def main():
    st.title("📋 PO Processing Agent with Gemini AI")
    st.markdown("Upload your documents and master data Excel files to automatically generate procurement CSV output.")
    
    # Sidebar for configuration
    st.sidebar.header("🔧 Configuration")
    
    # API Key input
    api_key = st.sidebar.text_input(
        "Gemini API Key", 
        type="password",
        help="Enter your Google Gemini API key"
    )
    
    if api_key:
        genai.configure(api_key=api_key)
        st.sidebar.success("✅ API Key configured")
    else:
        st.sidebar.warning("⚠️ Please enter your Gemini API key")
        st.info("Please enter your Gemini API key in the sidebar to continue.")
        return
    
    # Model selection
    model_options = [
        "gemini-2.5-pro"
    ]
    selected_model = st.sidebar.selectbox(
        "Select Gemini Model",
        model_options,
        index=0,
        help="Choose the Gemini model for processing"
    )
    
    # Create two columns for better layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Document Upload")
        
        # File uploaders for the three required PDFs
        st.subheader("Required Documents")
        po_file = st.file_uploader(
            "Purchase Order PDF", 
            type=['pdf'], 
            key="po_file",
            help="Upload the Purchase Order document"
        )
        
        invoice_file = st.file_uploader(
            "Tax Invoice PDF", 
            type=['pdf'], 
            key="invoice_file",
            help="Upload the Tax Invoice document"
        )
        
        jira_file = st.file_uploader(
            "Jira Ticket PDF", 
            type=['pdf'], 
            key="jira_file",
            help="Upload the Jira Ticket document"
        )
        
        # Show uploaded file info
        uploaded_pdfs = [f for f in [po_file, invoice_file, jira_file] if f is not None]
        if uploaded_pdfs:
            st.info(f"📊 {len(uploaded_pdfs)}/3 PDF documents uploaded")
    
    with col2:
        st.header("📊 Master Data Files")
        
        # File uploader for master data Excel files
        master_data_files = st.file_uploader(
            "Master Data Excel Files",
            type=['xlsx', 'xls'],
            accept_multiple_files=True,
            key="master_data",
            help="Upload your master data Excel files (GL codes, Tax codes, etc.)"
        )
        
        # Display uploaded master data files with sheet information
        if master_data_files:
            st.subheader("Uploaded Master Data Files:")
            file_info = get_master_data_info(master_data_files)
            
            for file_name, info in file_info.items():
                with st.expander(f"📁 {file_name} ({info['total_sheets']} sheet{'s' if info['total_sheets'] > 1 else ''})"):
                    for sheet in info['sheets']:
                        st.write(f"📋 **Sheet:** {sheet}")
                        # Preview first few rows of each sheet
                        try:
                            preview_df = pd.read_excel(
                                next(f for f in master_data_files if f.name == file_name), 
                                sheet_name=sheet, 
                                nrows=3
                            )
                            preview_df.columns = preview_df.columns.str.strip()
                            st.dataframe(preview_df, use_container_width=True)
                            st.caption(f"Columns: {', '.join(preview_df.columns)}")
                        except Exception as e:
                            st.write(f"❌ Error previewing sheet: {e}")
    
    # Advanced Options
    with st.expander("⚙️ Advanced Options"):
        col_adv1, col_adv2 = st.columns(2)
        
        with col_adv1:
            enable_debug = st.checkbox("Enable Debug Mode", help="Show detailed processing information")
            
        with col_adv2:
            custom_output_name = st.text_input(
                "Custom Output Filename", 
                placeholder="Leave empty for auto-generated name",
                help="Custom name for the output CSV file"
            )
    
    # Process button
    st.markdown("---")
    
    process_button = st.button("🚀 Process Documents", type="primary", use_container_width=True)
    
    if process_button:
        # Validation
        if not all([po_file, invoice_file, jira_file]):
            st.error("❌ Please upload all three required PDF documents.")
            return
        
        if not master_data_files:
            st.error("❌ Please upload at least one master data Excel file.")
            return
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Load master data
            status_text.text("📊 Loading master data files...")
            progress_bar.progress(10)
            master_data = load_master_data_files(master_data_files)
            
            if enable_debug:
                st.subheader("🔍 Debug: Master Data Loaded")
                for key, df in master_data.items():
                    st.write(f"**{key}:** {len(df)} rows, {len(df.columns)} columns")
                    st.write(f"Columns: {', '.join(df.columns)}")
            
            # Process documents
            status_text.text("🤖 Processing documents with Gemini AI...")
            progress_bar.progress(30)
            
            # Create temporary files for the PDFs
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded files to temporary directory
                po_path = os.path.join(temp_dir, po_file.name)
                invoice_path = os.path.join(temp_dir, invoice_file.name)
                jira_path = os.path.join(temp_dir, jira_file.name)
                
                with open(po_path, "wb") as f:
                    f.write(po_file.getvalue())
                with open(invoice_path, "wb") as f:
                    f.write(invoice_file.getvalue())
                with open(jira_path, "wb") as f:
                    f.write(jira_file.getvalue())
                
                # Upload files to Gemini
                status_text.text("📤 Uploading files to Gemini...")
                progress_bar.progress(50)
                
                po_gemini_file = genai.upload_file(path=po_path)
                invoice_gemini_file = genai.upload_file(path=invoice_path)
                jira_gemini_file = genai.upload_file(path=jira_path)
                
                # Configure the model with custom tools
                def lookup_tool(file_name: str, lookup_column: str, lookup_value: str, return_column: str) -> str:
                    return lookup_master_data(file_name, lookup_column, lookup_value, return_column, master_data)
                
                def generate_csv_tool(po_data_json: str, output_filename: str) -> str:
                    result, csv_data, df = generate_output_csv(po_data_json, output_filename)
                    if csv_data:
                        st.session_state['csv_output'] = csv_data
                        st.session_state['output_df'] = df
                        st.session_state['output_filename'] = output_filename
                        st.session_state['processed_data'] = json.loads(po_data_json)
                    return result
                
                model = genai.GenerativeModel(
                    model_name=selected_model,
                    tools=[lookup_tool, generate_csv_tool]
                )
                
                # Create the prompt with available files information
                status_text.text("🧠 Building AI prompt...")
                progress_bar.progress(60)
                
                # Build available files list with sheet information
                available_files_list = []
                file_info = get_master_data_info(master_data_files)
                
                for file_name, info in file_info.items():
                    if info['total_sheets'] == 1:
                        available_files_list.append(f"`{file_name}`")
                    else:
                        for sheet in info['sheets']:
                            available_files_list.append(f"`{file_name}_{sheet}`")
                
                available_files = ", ".join(available_files_list)
                
                # Enhanced prompt with better instructions
                prompt = f"""
                **ROLE & GOAL:**
                You are an AI agent specializing in procurement data processing. Your task is to extract information from three provided documents (a Purchase Order, a Tax Invoice, and a Jira Ticket), use the `lookup_tool` function to find corresponding codes from master data files, and then call the `generate_csv_tool` function with the final, complete data.

                **AVAILABLE MASTER DATA FILES:**
                {available_files}
                
                **NOTE:** If an Excel file has multiple sheets, use the format `filename.xlsx_SheetName` to reference specific sheets.

                **INSTRUCTIONS:**
                1. **Extract Data:** Carefully read all three documents to gather the initial information including:
                   - PO Number, Vendor details, dates, amounts
                   - Service descriptions, tax information
                   - Requestor/approver information from Jira
                
                2. **Enrich Data with Tools:** Use the `lookup_tool` function to find all necessary codes:
                   - GL Account (based on the service description)
                   - Tax Code (based on the invoice's tax percentages)  
                   - Requestor ID (based on the approver's/requester's name from the Jira ticket)
                   - Any other codes needed from the master files
                   - When calling lookup_tool, use the exact file names as shown in the available files list above
                
                3. **Construct Final Data:** Assemble all the extracted and looked-up data into a single, structured JSON object matching the schema below.
                
                4. **Generate CSV:** Call the `generate_csv_tool` function with the completed JSON data.

                **OUTPUT JSON SCHEMA TO POPULATE:**
                {{
                  "Document Type": "ZNID",
                  "PO Number": "",
                  "Line Item Number": "10", 
                  "Vendor": "",
                  "Document Date": "",
                  "Payment Terms": "P000",
                  "Purchasing Organisation": "1001",
                  "Purchase Group": "S05",
                  "Company Code": "1001", 
                  "Validity Start Date": "19.07.2025",
                  "Validity End Date": "19.07.2026",
                  "Short Text": "Housekeeping Service",
                  "Plant": "DS01",
                  "Service Number": "21000020",
                  "Service Quantity": "1",
                  "Gross Price": "",
                  "Cost Center": "DSG0010001",
                  "WBS": "",
                  "Tax Code": "",
                  "Material Group": "MG021", 
                  "Requestor": "",
                  "Control Code": "999433",
                  "GL Account": "",
                  "UOM": "AU"
                }}

                **IMPORTANT NOTES:**
                - Extract actual values from the documents where possible
                - Use lookup_tool to find codes that match the extracted information
                - If a lookup fails, indicate this in your response but continue processing
                - Ensure all monetary values are properly formatted
                - Use the custom filename if processing is successful

                Begin the process now.
                """
                
                # Process with Gemini
                status_text.text("🤖 Processing with Gemini AI...")
                progress_bar.progress(80)
                
                chat = model.start_chat(enable_automatic_function_calling=True)
                response = chat.send_message([prompt, po_gemini_file, invoice_gemini_file, jira_gemini_file])
                
                progress_bar.progress(100)
                status_text.text("✅ Processing complete!")
                
                st.success("🎉 Document processing completed successfully!")
                
                # Display results
                st.markdown("---")
                st.subheader("🤖 AI Processing Response")
                
                with st.expander("View AI Response", expanded=True):
                    st.write(response.text)
                
                # Display and download CSV if generated
                if 'csv_output' in st.session_state:
                    st.markdown("---")
                    st.subheader("📊 Generated CSV Output")
                    
                    # Display the dataframe
                    st.dataframe(st.session_state['output_df'], use_container_width=True)
                    
                    # Create download section
                    col_dl1, col_dl2 = st.columns(2)
                    
                    with col_dl1:
                        # Determine filename
                        if custom_output_name:
                            filename = custom_output_name if custom_output_name.endswith('.csv') else f"{custom_output_name}.csv"
                        else:
                            filename = st.session_state['output_filename']
                        
                        st.download_button(
                            label="💾 Download CSV File",
                            data=st.session_state['csv_output'],
                            file_name=filename,
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col_dl2:
                        # Download JSON data
                        if 'processed_data' in st.session_state:
                            json_data = json.dumps(st.session_state['processed_data'], indent=2)
                            json_filename = filename.replace('.csv', '.json')
                            st.download_button(
                                label="📄 Download JSON Data",
                                data=json_data,
                                file_name=json_filename,
                                mime="application/json",
                                use_container_width=True
                            )
                    
                    # Show raw CSV data in expander
                    with st.expander("📋 View Raw CSV Data"):
                        st.code(st.session_state['csv_output'], language="csv")
                    
                    # Show processed JSON data
                    if 'processed_data' in st.session_state:
                        with st.expander("🔍 View Processed JSON Data"):
                            st.json(st.session_state['processed_data'])
                
        except Exception as e:
            progress_bar.progress(0)
            status_text.text("")
            st.error(f"❌ Error processing documents: {str(e)}")
            if enable_debug:
                st.exception(e)
    
    # Information section
    st.markdown("---")
    st.markdown("### ℹ️ How to Use This Application")
    
    tab1, tab2, tab3 = st.tabs(["📖 Instructions", "📋 File Requirements", "🔧 Troubleshooting"])
    
    with tab1:
        st.markdown("""
        **Step-by-step process:**
        
        1. **Configure API:** Enter your Gemini API key in the sidebar
        2. **Upload PDFs:** Add your three required PDF documents:
           - Purchase Order PDF
           - Tax Invoice PDF  
           - Jira Ticket PDF
        3. **Upload Master Data:** Add your Excel files containing lookup data
        4. **Configure Options:** Set advanced options if needed
        5. **Process:** Click 'Process Documents' to start AI processing
        6. **Download:** Get your generated CSV and JSON files
        """)
    
    with tab2:
        st.markdown("""
        **Required PDF Documents:**
        - 📄 **Purchase Order:** Contains PO number, vendor, dates, amounts
        - 🧾 **Tax Invoice:** Contains tax information and pricing details  
        - 🎫 **Jira Ticket:** Contains requestor/approver information
        
        **Master Data Excel Files:**
        - 📊 **GL Account Codes:** Service descriptions → GL codes
        - 💰 **Tax Codes:** Tax percentages → Tax codes
        - 👤 **Requestor Data:** Names → Requestor IDs
        - 📈 **Other Reference Data:** Any additional lookup tables
        
        **Excel File Support:**
        - ✅ `.xlsx` and `.xls` formats supported
        - ✅ Single sheet: Reference as `filename.xlsx`
        - ✅ Multiple sheets: Reference as `filename.xlsx_SheetName`
        - ✅ Automatic column name cleaning
        """)
    
    with tab3:
        st.markdown("""
        **Common Issues & Solutions:**
        
        **🔑 API Issues:**
        - Ensure your Gemini API key is valid and has sufficient quota
        - Check your Google Cloud project settings
        
        **📁 File Issues:**
        - Ensure PDF files are not corrupted or password-protected
        - Check Excel files have proper column headers
        - Verify file sizes are within limits
        
        **🔍 Lookup Issues:**
        - Enable debug mode to see detailed processing information
        - Check column names in your Excel files match expectations
        - Ensure lookup values exist in your master data
        
        **💡 Tips:**
        - Use clear, descriptive column names in Excel files
        - Test with smaller files first
        - Keep master data organized in separate sheets by type
        """)

if __name__ == "__main__":
    main()