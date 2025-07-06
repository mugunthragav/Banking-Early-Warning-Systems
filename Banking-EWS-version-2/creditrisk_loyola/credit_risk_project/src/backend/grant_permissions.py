import mysql.connector
from mysql.connector import Error

try:
    # First connect as root to create user and database
    connection = mysql.connector.connect(
        host='localhost',        user='root',
        password='20112003'  # Root password
    )

    if connection.is_connected():
        cursor = connection.cursor()
        
        # Create database if it doesn't exist
        cursor.execute("CREATE DATABASE IF NOT EXISTS credit_risk_db")
        
        # Create user if it doesn't exist and grant privileges
        cursor.execute("CREATE USER IF NOT EXISTS 'interns'@'localhost' IDENTIFIED BY '20112003'")
        cursor.execute("GRANT ALL PRIVILEGES ON credit_risk_db.* TO 'interns'@'localhost'")
        cursor.execute("FLUSH PRIVILEGES")
        
        print("Successfully created user and granted permissions!")

except Error as e:
    print("Error while connecting to MySQL", e)
finally:
    if 'connection' in locals() and connection.is_connected():
        cursor.close()
        connection.close()
        print("MySQL connection is closed")