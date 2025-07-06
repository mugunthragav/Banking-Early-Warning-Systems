import mysql.connector
from mysql.connector import Error

try:
    connection = mysql.connector.connect(
        host='localhost',
        user='interns',
        password='mukunth',
        database='credit_risk_db'
    )

    if connection.is_connected():
        db_info = connection.get_server_info()
        print("Connected to MySQL Server version ", db_info)
        cursor = connection.cursor()
        
        # Create the database if it doesn't exist
        cursor.execute("CREATE DATABASE IF NOT EXISTS credit_risk_db")
        cursor.execute("USE credit_risk_db")
          # Create the application_logs table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS application_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP NULL ON UPDATE CURRENT_TIMESTAMP,
                status_message VARCHAR(255),
                loan_amnt BIGINT,
                pd_ml_probability FLOAT,
                pd_ml_prediction INT,
                lgd_ml_ann FLOAT,
                recovery_rate_ml FLOAT,
                ead_ml_meta FLOAT,
                expected_loss_ml FLOAT,
                probability_of_repayment FLOAT,
                ai_interpretation_text TEXT
            )
        """)
        
        print("Successfully created database and table!")

except Error as e:
    print("Error while connecting to MySQL", e)
finally:
    if 'connection' in locals() and connection.is_connected():
        cursor.close()
        connection.close()
        print("MySQL connection is closed")