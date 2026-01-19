import psycopg2

# Connects to the database server using "localhost:port" with username and password
conn = psycopg2.connect(
    host="localhost", port=5432, user="postgres", password="postgres", dbname="postgres"
)
conn.autocommit = True # Ensures CREATEDATABASE runs outside a transaction 
cur = conn.cursor()

# Creates a database called "ms3_jokes"
cur.execute("SELECT 1 FROM pg_database WHERE datname = 'ms3_jokes'")
if not cur.fetchone():
    cur.execute("CREATE DATABASE ms3_jokes")
print("Created database 'ms3_jokes'")

cur.close()
conn.close()

# Connects to new db and creates a table called "jokes". Line 28 creates attribute ID as primary key, line 29 creates attribute joke as text
conn = psycopg2.connect(
    host="localhost", port=5432, user="postgres", password="postgres", dbname="ms3_jokes"
)
conn.autocommit = True
cur = conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS jokes (
        id SERIAL PRIMARY KEY,
        joke TEXT
    )
""")
print("Created table 'jokes'")

# Insert favorite joke into table
joke = "How much did the pirate pay to get his ears pierced? A buccaneer!"
cur.execute("INSERT INTO jokes (joke) VALUES (%s) RETURNING id", (joke,))
joke_id = cur.fetchone()[0]
print(f"Inserted joke with ID {joke_id}")

# Select, fetches and print joke
cur.execute("SELECT joke FROM jokes WHERE id = %s", (joke_id,))
fetched_joke = cur.fetchone()[0]
print("Your joke from database:", fetched_joke)

cur.close()
conn.close()
