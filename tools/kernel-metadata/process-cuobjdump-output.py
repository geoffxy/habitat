import argparse
import sqlite3
import sys
import re

ARCH_LINE_REGEX = re.compile('^arch = sm_(?P<arch>[0-9]+)$')
FUNC_LINE_REGEX = re.compile('^\sFunction\s(?P<name>.+):$')
RES_LINE_REGEX = re.compile('^\s\sREG:(?P<registers>[0-9]+)\s.*$')


class Parser:
    """
    Parses cuobjdump output that uses the -res-usage flag.

    This parser is implemented using a coroutine. Use the consume() method to
    send input lines to the parser. The consume() method returns a parsed
    kernel or None (when more input is required).
    """
    def __init__(self):
        self._impl = self._parser_coroutine()
        next(self._impl)

    def consume(self, line):
        result = self._impl.send(line)
        if result is not None:
            next(self._impl)
        return result

    def _parser_coroutine(self):
        arch = None

        while True:
            line = (yield)[:-1]

            arch_match = ARCH_LINE_REGEX.match(line)
            if arch_match is not None:
                arch = int(arch_match.group('arch'))
                continue

            func_line_match = FUNC_LINE_REGEX.match(line)
            if func_line_match is None:
                continue

            # When we find a function, we expect the next line to be its
            # corresponding resource string
            func_name = func_line_match.group('name')

            res_line = (yield)[:-1]

            resource_match = RES_LINE_REGEX.match(res_line)
            if resource_match is None:
                raise AssertionError(
                    'Missing resource information for function: ' + func_name)

            registers_per_thread = int(resource_match.group('registers'))
            yield (func_name, arch, registers_per_thread)


def ensure_tables_exist(connection):
    create_table = """
    CREATE TABLE IF NOT EXISTS kernels (
      name TEXT NOT NULL,
      arch INT NOT NULL,
      registers_per_thread INT NOT NULL,
      PRIMARY KEY (name, arch)
    )
    """
    cursor = connection.cursor()
    cursor.execute(create_table)
    connection.commit()


def insert_kernel(connection, name, arch, registers_per_thread):
    query = """
    INSERT INTO kernels (name, arch, registers_per_thread) VALUES (?, ?, ?)
    """
    cursor = connection.cursor()
    cursor.execute(query, (name, arch, registers_per_thread))


def process_cuobjdump_output(connection):
    parser = Parser()

    for line in iter(sys.stdin.readline, ''):
        kernel_info = parser.consume(line)
        if kernel_info is None:
            continue

        try:
            insert_kernel(connection, *kernel_info)
        except sqlite3.IntegrityError:
            # cuobjdump duplicates kernel entries - skip them for now
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, required=True)
    args = parser.parse_args()

    connection = sqlite3.connect(args.database)
    ensure_tables_exist(connection)
    process_cuobjdump_output(connection)
    connection.commit()
    connection.close()


if __name__ == '__main__':
    main()
