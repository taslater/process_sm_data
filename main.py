import psycopg2
import config


def connect_db(dbname):
    if dbname != config.db_dict['dbname']:
        raise ValueError("Couldn't not find DB with given name")
    conn = psycopg2.connect(
        host=config.db_dict['host'],
        user=config.db_dict['user'],
        password=config.db_dict['password'],
        dbname=config.db_dict['dbname'])
    return conn


print(connect_db('audubon_prev'))
