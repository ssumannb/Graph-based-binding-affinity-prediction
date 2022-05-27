from neo4j import GraphDatabase

# information for access
config = dict(IP_ADDRESS='localhost',
              BOLT_PORT = '7687',
              USER_NAME = 'neo4j',
              PASSWORD = '0000')


class Connect2neo4j:
    def __init__(self):
        self.config = dict(IP_ADDRESS='localhost',
              BOLT_PORT = '7687',
              USER_NAME = 'neo4j',
              PASSWORD = '0000')
        self.driver = GraphDatabase.driver(
            uri=f"bolt://{config['IP_ADDRESS']}:{config['BOLT_PORT']}",
            auth=(config['USER_NAME'], config['PASSWORD']))

    def close(self):
        self.driver.close()

    def run_query(self, message):
        with self.driver.session() as session:
            # creation = session.write_transaction(self._create_and_return, message)
            session.write_transaction(self._create_and_return, message)

            # print(creation)

    def _create_and_return(self, tx, message):
        result = tx.run(message)
        print(result)

        # return result.single()[0]



if __name__ == "__main__":
    complexGraph = Connect2neo4j(config)
    complex = '3i3b', '2010  B8LFD6  BETA-GALACTOSIDASE'
    complexGraph.print_creation(complex)

    complexGraph.close()