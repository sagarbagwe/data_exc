# Invoice Processing Agent

This is a Streamlit application that uses Google's Gemini AI to automate the processing of procurement documents. The application extracts key information from a Tax Invoice, a Jira ticket, and an optional Contract, enriches this data by performing lookups on uploaded master data Excel files, and generates a final, structured CSV and JSON output for procurement systems.

## Features

- **Multi-Document Processing:** Processes multiple PDF documents (Invoice, Jira Ticket, and optional Contract) simultaneously.
- **Master Data Integration:** Supports uploading and querying custom master data from Excel files for lookups (e.g., GL codes, Tax codes, Requestor IDs).
- **Tool-Augmented AI:** The Gemini model is equipped with custom Python tools to perform data lookups and generate the final output.
- **Configurable:** Allows you to enter your Gemini API key and select a processing model directly within the app.
- **CSV & JSON Output:** Generates both a final CSV file and a JSON object of the processed data, ready for download.
- **User-Friendly Interface:** Built with Streamlit for a simple, interactive user experience.

## Prerequisites

Before running the application, you need to have the following installed:

- **Python 3.8 or later**
- **A Google Gemini API Key:** You can obtain one from the [Google AI Studio](https://aistudio.google.com/app/apikey).

## Installation

1.  **Clone the repository or save the Python code:**
    Save the provided Python code into a file named `app.py`.

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required Python packages:**
    The application uses several libraries. You can install them with a single command:
    ```bash
    pip install streamlit pandas google-generativeai openpyxl
    ```
    This command installs Streamlit for the web interface, pandas for data manipulation, `google-generativeai` for the Gemini API, and `openpyxl` to handle Excel files.

## How to Run the Application

1.  **Start the Streamlit server:**
    Open your terminal or command prompt, navigate to the directory where you saved `app.py`, and run the following command:
    ```bash
    streamlit run app.py
    ```

2.  **Access the application:**
    Your web browser will automatically open a new tab with the application running at `http://localhost:8501`.

## How to Use the Application

1.  **Configure:** In the sidebar, enter your **Gemini API Key** and select the desired Gemini model.

2.  **Upload Documents:**
    -   Upload the required **Tax Invoice PDF** and **Jira Ticket PDF**.
    -   Optionally, upload a **Contract PDF**.

3.  **Upload Master Data:**
    -   Upload one or more **Master Data Excel Files (.xlsx, .xls)**. These files are used for lookups (e.g., mapping service descriptions to GL Accounts).
    -   The application will display a preview of the uploaded files and their sheets.

4.  **Process Documents:**
    -   Click the **"🚀 Process Documents"** button. The application will use Gemini to analyze the documents, perform lookups, and generate the final output. A progress bar will show the status.

5.  **Review and Download:**
    -   Once processing is complete, the generated CSV and JSON data will be displayed on the page.
    -   Use the **"💾 Download CSV File"** and **"📄 Download JSON Data"** buttons to save the output to your local machine.

## Master Data File Structure

The AI agent expects to perform lookups on your master data. While the column names can be flexible, ensure your Excel files contain the necessary data for the lookups, such as:

-   A sheet with service descriptions and corresponding GL accounts.
-   A sheet with tax percentages and their corresponding tax codes (e.g., `I4` for 18% GST).
-   A sheet with vendor names/IDs and vendor codes.
-   A sheet with requestor names and requestor IDs.

If your Excel file has multiple sheets, the agent will reference them using the format `filename.xlsx_SheetName`.

## Troubleshooting

-   **"API Key not configured"**: Make sure you have entered a valid Gemini API key in the sidebar.
-   **"Error processing documents"**: Check the console for more details. This could be due to invalid file formats, issues with the API key, or a problem with the Gemini service itself.
-   **Lookup failures**: Ensure your master data Excel files are uploaded correctly and contain the expected column headers and data. The application can handle some errors, but a successful lookup depends on good quality master data.