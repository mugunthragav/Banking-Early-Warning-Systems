# Installation Guide

This guide will walk you through setting up the Credit Risk Analysis application on your local machine.

### **Step 1: Prerequisites**

Ensure you have the following software installed:
* Python 3.8+
* MySQL Server

### **Step 2: Get the Code**

Clone the repository to your local machine:
```bash
git clone <your-repository-url>
cd <your-repository-directory>
```

### **Step 3: Set Up Python Environment**

It's highly recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

### **Step 4: Set Up the Database and User**

This step prepares your MySQL database.

1.  **Edit `grant_permissions.py`**: Open the `grant_permissions.py` file. If your MySQL root password is not `'20112003'`, you must update it in this script.
2.  **Run the script**: Execute the script to create the database (`credit_risk_db`) and a dedicated user (`interns`).

    ```bash
    python grant_permissions.py
    ```
    *This script only needs to be run **once** during the initial setup.*

### **Step 5: Configure Environment Variables**

You need to provide your own database password and a secret key for the AI service (like OpenAI's ChatGPT).

1.  Create a new file named `.env` in the root directory of the project.
2.  Add the following content to the `.env` file.

    ```env
    # .env file

    # --- Database Configuration ---
    # The password for the 'interns' user you created in grant_permissions.py
    MYSQL_PASSWORD="mukunth"

    # --- External API Keys ---
    # Your secret API key from OpenAI or another LLM provider
    LLM_API_KEY="sk-your-openai-api-key-goes-here"
    LLM_MODEL_NAME="gpt-3.5-turbo"
    ```
    > **IMPORTANT**: You **must** replace the placeholder values with your actual database password and a valid API key for the AI features to work correctly.

### **Step 6: Create Database Tables**

Now, run the `db_connection.py` script to create the necessary tables in your new database.

```bash
python db_connection.py
```
*This script connects to the database using the credentials you provided in the `.env` file and creates the tables based on the models in the code.*

### **Step 7: All Done!**

The setup is complete. You are now ready to run the application.