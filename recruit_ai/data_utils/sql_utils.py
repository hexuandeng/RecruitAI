import os
import time
import random
import threading

sql_type = os.getenv('SQL_TYPE')

print(f'SQL_TYPE={sql_type}')

if sql_type == 'mysql':
    import mysql.connector

    tag_conn = mysql.connector.connect(host='localhost',
                                       user='root',
                                       password='123456',
                                       database='ai_interview_tag',
                                       charset='utf8mb4')
    tag_cursor = tag_conn.cursor()
    que_conn = mysql.connector.connect(host='localhost',
                                       user='root',
                                       password='123456',
                                       database='ai_interview_question',
                                       charset='utf8mb4',
                                       use_unicode=True)
    que_cursor = que_conn.cursor()
    place_holder = '%s'
elif sql_type == 'sqlite':
    import sqlite3

    tag_conn = sqlite3.connect('../datas/ai_interview_tag.db')
    tag_cursor = tag_conn.cursor()
    que_conn = sqlite3.connect('../datas/ai_interview_question.db')
    que_cursor = que_conn.cursor()
    place_holder = '?'
else:
    raise ValueError('SQL_TYPE must be mysql or sqlite')

tag_types = ['jd', 'resume']
for tag_type in tag_types:
    tag_cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS tagged_{tag_type} (
            id INTEGER PRIMARY KEY,
            category TEXT NOT NULL,
            labels TEXT NOT NULL
        );
    ''')
tag_conn.commit()


def check_if_id_exists_in_tag_db(types: str, id: int):
    tag_cursor.execute(
        f'SELECT * FROM tagged_{types} WHERE id = {place_holder}', (id, ))
    return tag_cursor.fetchone()


def insert_data_into_tag_db(
    types: str,
    id: int,
    category: str,
    labels: str,
):
    if sql_type == 'mysql':
        tag_cursor.execute(
            f'''
        INSERT INTO tagged_{types} (id, category, labels) 
        VALUES (%s, %s, %s) 
        ON DUPLICATE KEY UPDATE 
            category = VALUES(category), 
            labels = VALUES(labels)
        ''', (id, category, labels))
    else:
        tag_cursor.execute(
            f'INSERT OR REPLACE INTO tagged_{types} (id, category, labels) VALUES (?,?,?)',
            (id, category, labels))
    tag_conn.commit()


def select_category_labels_from_tag_db(
    table: str,
    id: int,
):
    tag_cursor.execute(
        f'SELECT category, labels FROM {table} WHERE id={place_holder}',
        (id, ))
    return tag_cursor.fetchone()


def select_question_from_que_db(question_type: str, label=str):
    que_cursor.execute(
        f'SELECT {question_type} FROM label_to_question WHERE label={place_holder}',
        (label, ))
    return que_cursor.fetchone()


def select_question_from_que_db_randomly(question_type: str):
    que_cursor.execute('SELECT COUNT(*) FROM label_to_question')
    total_rows = que_cursor.fetchone()[0]
    if total_rows == 0:
        raise ValueError("The table label_to_question is empty.")
    while True:
        # Generate random offsets
        random_offset = random.randint(0, total_rows - 1)
        # Querying random rows
        que_cursor.execute(
            f'SELECT {question_type} FROM label_to_question LIMIT 1 OFFSET {random_offset}'
        )
        row = que_cursor.fetchone()
        # If the result is not empty, return the data
        if row and row[0]:
            return row


def select_choice_question_from_que_db(question_id: int, difficulty: str):
    que_cursor.execute(
        f'SELECT question, options, answer FROM choice_question WHERE id={place_holder} AND difficulty={place_holder}',
        (question_id, difficulty))
    return que_cursor.fetchone()


def select_fill_bank_question_from_que_db(question_id: int, difficulty: str):
    que_cursor.execute(
        f'SELECT question, answer FROM fill_bank_question WHERE id={place_holder} AND difficulty={place_holder}',
        (question_id, difficulty))
    return que_cursor.fetchone()


def select_judge_question_from_que_db(question_id: int, difficulty: str):
    que_cursor.execute(
        f'SELECT questions, answers FROM judge_question WHERE id={place_holder} AND difficulty={place_holder}',
        (question_id, difficulty))
    return que_cursor.fetchone()


def insert_data_into_que_db(question_set_id: int, choice: str, fill_bank: str,
                            judge: str):
    if sql_type == 'mysql':
        que_cursor.execute(
            '''
            INSERT INTO generated_written_question (id, choice_ids, fill_bank_ids, judge_ids)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                choice_ids = VALUES(choice_ids),
                fill_bank_ids = VALUES(fill_bank_ids),
                judge_ids = VALUES(judge_ids)
            ''', (question_set_id, choice, fill_bank, judge))
    else:
        que_cursor.execute(
            '''
            INSERT OR REPLACE INTO generated_written_question (id, choice_ids, fill_bank_ids, judge_ids) 
            VALUES (?,?,?,?)
            ''', (question_set_id, choice, fill_bank, judge))
    que_conn.commit()


def select_interview_question_id_from_que_db(label: str):
    que_cursor.execute(
        f'SELECT questions_id FROM interview_label_id WHERE label={place_holder}',
        (label, ))
    return que_cursor.fetchone()


def select_interview_question_answer_from_que_db(question_id: int):
    print(f"{'=' * 25}{question_id}{'='*25}")
    que_cursor.execute(
        f'SELECT question, answer, difficult_level FROM interview_questions WHERE id={place_holder}',
        (question_id, ))
    return que_cursor.fetchone()


def select_question_set_from_que_db(question_set_id: int):
    que_cursor.execute(
        f'SELECT choice_ids, fill_bank_ids, judge_ids FROM generated_written_question WHERE id={place_holder}',
        (question_set_id, ))
    return que_cursor.fetchone()


def select_choice_from_que_db(choice_id: int):
    que_cursor.execute(
        f'SELECT labels, question, options, answer FROM choice_question WHERE id={place_holder}',
        (choice_id, ))
    return que_cursor.fetchone()


def select_fill_bank_from_que_db(fill_bank_id: int):
    que_cursor.execute(
        f'SELECT labels, question, answer FROM fill_bank_question WHERE id={place_holder}',
        (fill_bank_id, ))
    return que_cursor.fetchone()


def select_judge_from_que_db(judge_id: int):
    que_cursor.execute(
        f'SELECT labels, questions, answers FROM judge_question WHERE id={place_holder}',
        (judge_id, ))
    return que_cursor.fetchone()


# Heartbeat thread that maintains database connection
def keep_alive():
    while True:
        try:
            print("Keep sqlite connection")

            # Each thread creates its own connection and cursor
            que_connection = sqlite3.connect(
                "../datas/ai_interview_question.db",
                check_same_thread=False)  # Set to False to allow connections to be used across threads.
            que_cursor = que_connection.cursor()
            que_cursor.execute("SELECT 1;")  # Simple query to keep the connection alive
            que_cursor.fetchall()

            tag_connection = sqlite3.connect("../datas/ai_interview_tag.db",
                                             check_same_thread=False)
            tag_cursor = tag_connection.cursor()
            tag_cursor.execute("SELECT 1;")
            tag_cursor.fetchall()

            # Waiting for the next heartbeat
            time.sleep(9600)

            # Close the connection and cursor
            que_cursor.close()
            tag_cursor.close()
            que_connection.close()
            tag_connection.close()
        except sqlite3.Error as e:
            print(f"Error keeping connection alive: {e}")
            print("Keep SQLite connection error")
            break


# Periodically send keep-alive queries
# Start a new thread to keep the connection alive
heartbeat_thread = threading.Thread(target=keep_alive)
heartbeat_thread.daemon = True  # Set as a daemon thread, automatically exit when the main thread ends
heartbeat_thread.start()
