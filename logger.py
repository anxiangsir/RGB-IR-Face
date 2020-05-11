import logging
import os
import sys


def setlogger(models_root, rank):
    formatter = logging.Formatter("worker-id:" + str(rank) + ":%(asctime)s-%(message)s")
    logger = logging.getLogger()

    file_handler = logging.FileHandler(os.path.join(models_root, "hist.log"))
    stream_handler = logging.StreamHandler(sys.stdout)

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)
    logger.info('worker_id: %d' % rank)
    return logger
