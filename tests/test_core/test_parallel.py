from unittest import TestCase
from core.parallel import *
import numpy as np
import numpy.testing as np_test
from multiprocessing import Process, Pool, cpu_count


def read_and_verify(name, output):
    with read_array_shared_memory(name) as read_memory:
        for i in range(len(output)):
            np_test.assert_equal(
                output[i], read_memory[i]
            )
    return True


class TestReadWriteSharedMemory(TestCase):
    test_vecs1 = (np.arange(10), np.arange(4))
    test_vecs2 = (np.ones(4), np.zeros(4))

    def test_read_write_array_shared_memory(self):
        for vec in [self.test_vecs1, self.test_vecs2]:
            with create_parameters_shared_memory(vec) as memory:
                read_and_verify(memory.name, vec)

    def test_read_write_shared_memory_separate_process(self):
        for vec in [self.test_vecs1, self.test_vecs2]:
            # Test singular
            with create_parameters_shared_memory(vec) as memory:
                p = Process(target=read_and_verify, args=(memory.name, vec))
                p.start()
                p.join()
                p.close()

            # Test many concurrent
            with create_parameters_shared_memory(vec) as memory:
                p = Pool(cpu_count() - 2)
                mapper = p.starmap_async(
                    read_and_verify,
                    zip(
                        [memory.name] * 50,
                        [vec] * 50
                    )
                )
                mapper.get()
                p.close()
