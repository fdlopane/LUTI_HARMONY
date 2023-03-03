from config_ATH import *
import logging

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.warning("main_ATH started")
    from HARMONY_LUTI_ATH.main import start_main as run_ATH
    run_ATH(inputs, outputs, logger)
    logger.warning("Finished.")
