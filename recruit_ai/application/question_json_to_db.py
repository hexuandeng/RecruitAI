import json
import sqlite3

from tqdm import tqdm


# Initialize database connection
def get_database_connection(db_path='../datas/ai_interview_question.db'):
    """Establish and return a SQLite connection."""
    return sqlite3.connect(db_path)


# Create the required tables
def create_tables(cursor):
    """Create all necessary tables in the database."""
    tables = {
        "label_to_question":
        '''
            CREATE TABLE IF NOT EXISTS label_to_question (
                label TEXT PRIMARY KEY,
                choice TEXT NOT NULL,
                fill_bank TEXT NOT NULL,
                judge TEXT NOT NULL
            );
        ''',
        "choice_question":
        '''
            CREATE TABLE IF NOT EXISTS choice_question (
                id INTEGER PRIMARY KEY,
                field TEXT NOT NULL,
                labels TEXT NOT NULL,
                question TEXT NOT NULL,
                options TEXT NOT NULL,
                answer TEXT NOT NULL,
                difficulty TEXT NOT NULL
            );
        ''',
        "fill_bank_question":
        '''
            CREATE TABLE IF NOT EXISTS fill_bank_question (
                id INTEGER PRIMARY KEY,
                field TEXT NOT NULL,
                labels TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                difficulty TEXT NOT NULL
            );
        ''',
        "judge_question":
        '''
            CREATE TABLE IF NOT EXISTS judge_question (
                id INTEGER PRIMARY KEY,
                field TEXT NOT NULL,
                labels TEXT NOT NULL,
                questions TEXT NOT NULL,
                answers TEXT NOT NULL,
                difficulty TEXT NOT NULL
            );
        ''',
        "generated_written_question":
        '''
            CREATE TABLE IF NOT EXISTS generated_written_question (
                id INTEGER PRIMARY KEY,
                choice_ids TEXT NOT NULL,
                fill_bank_ids TEXT NOT NULL,
                judge_ids TEXT NOT NULL
            );
        ''',
        "interview_questions":
        '''
            CREATE TABLE IF NOT EXISTS interview_questions (
                id INTEGER PRIMARY KEY,
                field TEXT NOT NULL,
                labels TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                difficult_level TEXT NOT NULL
            );
        ''',
        "interview_label_id":
        '''
            CREATE TABLE IF NOT EXISTS interview_label_id (
                label TEXT PRIMARY KEY,
                questions_id INTEGER NOT NULL
            );
        '''
    }
    for name, query in tables.items():
        cursor.execute(query)


# Insert data into the table
def insert_data(cursor, data):
    """Insert data into choice_question, fill_bank_question, and judge_question."""
    for qtype in ['choice', 'fill_bank', 'judge']:
        if qtype in data['types'] and data['types'][qtype]:
            if qtype == 'choice':
                cursor.execute(
                    '''
                    INSERT OR REPLACE INTO choice_question (id, field, labels, question, options, answer, difficulty)
                    VALUES (?,?,?,?,?,?,?)
                    ''', (data['id'], data['field'] if 'field' in data else '', ','.join(
                        data['label']), data['types']['choice']['question'],
                          json.dumps(data['types']['choice']['option'],
                                     ensure_ascii=False),
                          data['types']['choice']['answer'],
                          data['types']['choice']['difficulty']))
            elif qtype == 'fill_bank':
                cursor.execute(
                    '''
                    INSERT OR REPLACE INTO fill_bank_question (id, field, labels, question, answer, difficulty)
                    VALUES (?,?,?,?,?,?)
                    ''', (data['id'], data['field'] if 'field' in data else '',
                          ','.join(data['label']),
                          data['types']['fill_bank']['question'],
                          data['types']['fill_bank']['answer'],
                          data['types']['fill_bank']['difficulty']))
            elif qtype == 'judge':
                cursor.execute(
                    '''
                    INSERT OR REPLACE INTO judge_question (id, field, labels, questions, answers, difficulty)
                    VALUES (?,?,?,?,?,?)
                    ''', (data['id'], data['field'] if 'field' in data else '',
                          ','.join(data['label']),
                          json.dumps(data['types']['judge']['question'],
                                     ensure_ascii=False),
                          json.dumps(data['types']['judge']['answer'],
                                     ensure_ascii=False),
                          data['types']['judge']['difficulty']))


# Insert the mapping between tags and questions
def insert_label_to_question(cursor, label_to_question_dict):
    """Insert data into label_to_question table."""
    for label, questions in label_to_question_dict.items():
        cursor.execute(
            '''
            INSERT OR REPLACE INTO label_to_question (label, choice, fill_bank, judge)
            VALUES (?,?,?,?)
            ''', (label, ','.join(map(str, questions['choice'])), ','.join(
                map(str, questions['fill_bank'])), ','.join(
                    map(str, questions['judge']))))


# Main function
def main():
    db_path = '../datas/ai_interview_question.db'
    data_path = '../datas/exam_question_set_0506.json'
    interview_data_path = '../datas/interview_question_set_0506.json'
    interview_label_path = '../datas/interview_question_set_label_to_id_0506.json'

    conn = get_database_connection(db_path)
    cursor = conn.cursor()

    # Delete the interview_questions table
    cursor.execute('DROP TABLE IF EXISTS label_to_question')
    cursor.execute('DROP TABLE IF EXISTS choice_question')
    cursor.execute('DROP TABLE IF EXISTS fill_bank_question')
    cursor.execute('DROP TABLE IF EXISTS judge_question')
    cursor.execute('DROP TABLE IF EXISTS interview_questions')
    cursor.execute('DROP TABLE IF EXISTS interview_label_id')

    # Create Table
    create_tables(cursor)

    # Load and insert multiple types of topic data
    with open(data_path, 'r', encoding='utf-8') as f:
        multi_questions = json.load(f)

    label_to_question_dict = {}
    for data in tqdm(multi_questions, desc='Processing multi_questions'):
        if data['label'][0] not in label_to_question_dict:
            label_to_question_dict[data['label'][0]] = {
                'choice': [],
                'fill_bank': [],
                'judge': []
            }
        for qtype in ['choice', 'fill_bank', 'judge']:
            if qtype in data['types'] and data['types'][qtype]:
                label_to_question_dict[data['label'][0]][qtype].append(
                    data['id'])
        insert_data(cursor, data)

    # Insert label_to_question data
    insert_label_to_question(cursor, label_to_question_dict)

    # Insert interview question data
    with open(interview_data_path, 'r', encoding='utf-8') as f:
        interview_questions = json.load(f)
    for data in tqdm(interview_questions, desc='Processing interview_questions'):
        cursor.execute(
            '''
            INSERT OR REPLACE INTO interview_questions (id, field, labels, question, answer, difficult_level)
            VALUES (?,?,?,?,?,?)
            ''', (data['id'], data['field'], ','.join(data['label']),
                  data['question'], data['answer'], data['difficult']))

    # Insert the mapping data of interview labels and question IDs
    with open(interview_label_path, 'r', encoding='utf-8') as f:
        label_to_question = json.load(f)
    for data in tqdm(label_to_question, desc='Processing interview_label_id'):
        cursor.execute(
            '''
            INSERT OR REPLACE INTO interview_label_id (label, questions_id)
            VALUES (?,?)
            ''', (data['label'], ','.join(map(str, data["questions_id"]))))

    # Commit the transaction and close the connection
    conn.commit()
    conn.close()
    print("Finished inserting data.")


if __name__ == '__main__':
    main()
