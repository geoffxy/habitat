import argparse
import os
import re
import sqlite3
import sys

from features import FEATURES

file_name_regex = re.compile(
    '(?P<operation>[a-zA-Z0-9]+)-(?P<device>[a-zA-Z0-9]+)-\S+\.sqlite'
)

features_template = '{feature} INTEGER NOT NULL'


def extract_relevant_files(args):
    relevant_files = []
    for file_name in os.listdir(args.in_dir):
        match = file_name_regex.match(file_name)
        if match is None:
            continue
        if (match.group('operation') != args.operation or
                match.group('device') != args.device):
            continue
        relevant_files.append(file_name)
    return relevant_files


class Combiner:
    def __init__(self, operation, output_database_file):
        self._connection = sqlite3.connect(output_database_file)
        self._create_tables(operation)
        self._initialize_queries(operation)

    def _initialize_queries(self, operation):
        insert_query_template = """
          INSERT INTO {table_name}
          ({features}, is_forward, run_time_ms)
          VALUES ({placeholders}, ?, ?)
        """
        features_joined = ', '.join(FEATURES[operation])
        placeholders = ', '.join(['?'] * len(FEATURES[operation]))
        self._insert_temp = insert_query_template.format(
            table_name='recordings_temp',
            features=features_joined,
            placeholders=placeholders,
        )
        self._insert_final = insert_query_template.format(
            table_name='recordings',
            features=features_joined,
            placeholders=placeholders,
        )
        self._select_from_data_file = (
            'SELECT {features}, is_forward, run_time_ms FROM recordings'
            .format(features=features_joined)
        )
        self._deduplicate_select = """
          SELECT {features}, is_forward, AVG(run_time_ms)
          FROM recordings_temp
          GROUP BY {features}, is_forward
        """.format(features=features_joined)

    def _create_tables(self, operation):
        features_sql = ', '.join(map(
            lambda feature: features_template.format(feature=feature),
            FEATURES[operation],
        ))
        table_query = """
          CREATE TABLE {table_name} (
            id INTEGER PRIMARY KEY,
            {features},
            is_forward INTEGER NOT NULL,
            run_time_ms REAL NOT NULL
          )
        """

        cursor = self._connection.cursor()
        cursor.execute(table_query.format(
            table_name='recordings',
            features=features_sql,
        ))
        cursor.execute(table_query.format(
            table_name='recordings_temp',
            features=features_sql,
        ))
        self._connection.commit()

    def process_data_file(self, data_file):
        sub_connection = sqlite3.connect(data_file)
        try:
            select_subdata = sub_connection.cursor()
            select_subdata.execute(self._select_from_data_file)

            insert_subdata = self._connection.cursor()
            insert_subdata.executemany(self._insert_temp, select_subdata)
            self._connection.commit()
        except:
            self._connection.rollback()
            raise
        finally:
            sub_connection.close()

    def deduplicate_and_commit(self):
        try:
            select_cursor = self._connection.cursor()
            select_cursor.execute(self._deduplicate_select)

            insert_cursor = self._connection.cursor()
            insert_cursor.executemany(self._insert_final, select_cursor)
            insert_cursor.execute('DROP TABLE recordings_temp')
            self._connection.commit()
        except:
            self._connection.rollback()
            raise

    def __del__(self):
        if self._connection is not None:
            self._connection.close()


def main():
    """
    This script combines all the collected data related to an operation and
    device combination. We only combine the data in the "recordings" table.

    e.g.: conv2d_P100_0.sqlite + conv2d_P100_1.sqlite -> conv2d_P100.sqlite
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('operation')
    parser.add_argument('device')
    parser.add_argument('--in-dir', type=str, required=True)
    args = parser.parse_args()

    output_database_file = '{}-{}.sqlite'.format(args.operation, args.device)
    if os.path.exists(output_database_file):
        print(
            'ERROR: The output database file {} already exists. '
            'Aborting to avoid overwriting.'.format(output_database_file),
            file=sys.stderr,
        )
        sys.exit(1)

    relevant_files = extract_relevant_files(args)
    print('Detected files:')
    print(relevant_files)

    if len(relevant_files) == 0:
        print('ERROR: No files to combine.', file=sys.stderr)
        sys.exit(1)

    combiner = Combiner(args.operation, output_database_file)
    for idx, file in enumerate(relevant_files):
        print('Processing [{}/{}]:'.format(idx + 1, len(relevant_files)), file)
        combiner.process_data_file(os.path.join(args.in_dir, file))

    print('Deduplicating...')
    combiner.deduplicate_and_commit()
    print('Done!')


if __name__ == '__main__':
    main()
