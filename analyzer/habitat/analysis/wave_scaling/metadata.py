import sqlite3
import logging

logger = logging.getLogger(__name__)


class MetadataManager:
    def __init__(self, path_to_lut):
        self._connection = sqlite3.connect(path_to_lut)


    def kernel_registers_for(self, kernel, device):
        arch = int(''.join(map(lambda x: str(x), device.compute_capability)))
        cursor = self._connection.cursor()
        result = cursor.execute(
            MetadataManager.kernel_registers_query,
            (kernel.name, arch),
        ).fetchone()

        if result is None:
            logger.debug(
                'Missing kernel metadata entry for "%s" on arch %d.',
                kernel.name,
                arch,
            )
            return result

        actual_arch, registers_per_thread = result
        if actual_arch != arch:
            logger.debug(
                'Using substitute entry for "%s" at arch %d instead of %d.',
                kernel.name,
                actual_arch,
                arch,
            )

        return registers_per_thread


MetadataManager.kernel_registers_query = """
  SELECT arch, registers_per_thread FROM kernels
  WHERE name = ? AND arch <= ?
  ORDER BY arch DESC LIMIT 1
"""
