import psycopg2

class connect2pgSQL():
    def __init__(self, config=None):
        if not config:
            self.config = {
                'host': 'localhost',
                'port': '5432',
                'dbname': 'postgres',
                'user': 'postgres',
                'password': '1234'
            }
        else:
            self.config = config

        self.db = psycopg2.connect(host=self.config['host'],
                                   port=self.config['port'],
                                   dbname=self.config['dbname'],
                                   user=self.config['user'],
                                   password=self.config['password'])
        self.cursor = self.db.cursor()

    def __del__(self):
        self.db.close()
        self.cursor.close()

    def execute(self, query, args={}):
        self.cursor.exectue(query, args)
        row = self.cursor.fetchall()

        return row

    def commit(self):
        self.cursor.commit()


class CRUD(connect2pgSQL):
    def __init__(self):
        super().__init__()


    def createDB(self, schema, table, structure):
        # check schema first!
        sql_schema = " CREATE SCHEMA IF NOT EXISTS {schema} AUTHORIZATION {username} "\
            .format(schema=schema, username=self.config['user'])
        sql_head = " CREATE TABLE {schema}.{table} (".format(schema=schema, table=table)
        sql_body = []
        for i, line in enumerate(structure):
            sql_body.append(line)
            sql_body.append(',')

        sql_body.pop()
        sql_body = ''.join(sql_body)

        sql = f"{sql_head}{sql_body})"

        try:
            self.cursor.execute(sql_schema)
            self.cursor.execute(sql)
            self.db.commit()
        except Exception as e:
            print("create table error: ", e)


    def insertDB(self, schema, table, column, data) -> object:
        sql = " INSERT INTO {schema}.{table}({column}) VALUES ({data});"\
            .format(schema=schema, table=table, column=column, data=data)
        try:
            self.cursor.execute(sql)
        except Exception as e:
            print(sql)
            print("insert DB error: ", e)

    def commit(self):
        self.db.commit()

    def readDB(self, schema, table, column):
        sql = " SELECT {column} from {schema}.{table}"\
            .format(column=column, schema=schema, table=table)
        try:
            self.cursor.execute(sql)
            result = self.cursor.fetchall()
        except Exception as e:
            result = ("read DB error: ", e)

        print(result)


    def updateDB(self, schema, table, column, value, condition):
        sql = " UPDATE {schema}.{table} SET {column}=\'{value}\' WHERE {column}=\'{condition}\'"\
            .format(schema=schema, table=table, column=column, value=value, condition=condition)

        try:
            self.cursor.execute(sql)
            self.db.commit()
        except Exception as e:
            print(" update DB error: ", e)


    def deleteDB(self, schema, table, condition):
        sql = " DELETE from {schema}.{table} WHERE {condition}"\
            .format(schema=schema, table=table, condition=condition)

        try:
            self.cursor.execute(sql)
            self.db.commit()
        except Exception as e:
            print(" delete DB error: ", e)


    def run_query(self, query):
        try:
            self.cursor.execute(query)
            result = self.cursor.fetchall()
        except Exception as e:
            result = ("query error: ", e)

        return result


if __name__ == "__main__":
    db = CRUD()
    db.insertDB(schema='myschema',table='table',colum='ID',data='유동적변경')
    print(db.readDB(schema='myschema',table='table',colum='ID'))
    db.updateDB(schema='myschema',table='table',colum='ID', value='와우',condition='유동적변경')
    db.deleteDB(schema='myschema',table='table',condition ="id != 'd'")