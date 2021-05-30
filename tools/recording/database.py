import sqlite3
import logging

logger = logging.getLogger(__name__)

FEATURES_TEMPLATE = '{feature} INTEGER NOT NULL,'


class Recorder:
    def __init__(self, file_name, features):
        self._file_name = file_name
        self._features = features
        self._strings = {}
        self._connection, self._cursor = self._initialize()

    def _generate_queries(self):
        features_sql = ''.join(map(
            lambda f: FEATURES_TEMPLATE.format(feature=f),
            self._features,
        ))
        self._create_recordings = """
          CREATE TABLE IF NOT EXISTS recordings (
            id INTEGER PRIMARY KEY,
            {features}
            is_forward INTEGER NOT NULL,
            run_time_ms REAL NOT NULL
          )
        """.format(features=features_sql)
        self._insert_recording = """
          INSERT INTO recordings (
            {features},
            is_forward,
            run_time_ms
          )
          VALUES ({values} ?, ?)
        """.format(
            features=','.join(self._features),
            values='?,' * len(self._features),
        )

    def _initialize(self):
        self._generate_queries()
        connection = sqlite3.connect(self._file_name)
        cursor = connection.cursor()
        cursor.execute(self._create_recordings)
        cursor.execute("""
          CREATE TABLE IF NOT EXISTS kernels (
            id INTEGER PRIMARY KEY,
            recording_id INTEGER NOT NULL,
            kernel_name INTEGER NOT NULL,
            run_time_ns INTEGER NOT NULL
          )
        """)
        cursor.execute("""
          CREATE TABLE IF NOT EXISTS strings (
            id INTEGER PRIMARY KEY,
            value TEXT NOT NULL
          )
        """)
        connection.commit()
        return connection, cursor

    def get_num_recordings(self):
        self._cursor.execute("SELECT COUNT(*) FROM recordings")
        return self._cursor.fetchone()[0]

    def record(self, config, is_forward, run_time_ms, recorded_kernels):
        try:
            self._cursor.execute(
                self._insert_recording,
                (*tuple(map(int, config)), int(is_forward), run_time_ms),
            )
            recording_id = self._cursor.lastrowid
            for kernel in recorded_kernels:
                if kernel.name in self._strings:
                    kernel_name = self._strings[kernel.name]
                else:
                    self._cursor.execute(Recorder.insert_string, (kernel.name,))
                    kernel_name = self._cursor.lastrowid
                    self._strings[kernel.name] = kernel_name

                self._cursor.execute(
                    Recorder.insert_kernel,
                    (recording_id, kernel_name, kernel.run_time_ns)
                )
        except OverflowError:
            logger.warn(
                'Could not record a kernel because its run time overflowed the'
                'SQLite integer datatype.'
            )

    def commit(self):
        self._connection.commit()

    def __del__(self):
        self._connection.commit()
        self._connection.close()


Recorder.insert_kernel = """
  INSERT INTO kernels (recording_id, kernel_name, run_time_ns) VALUES (?, ?, ?)
"""

Recorder.insert_string = """
  INSERT INTO strings (value) VALUES (?)
"""
