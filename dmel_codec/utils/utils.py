import glob
import os

from utils.logger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def find_lastest_ckpt(directory):
    ckpt_file = glob.glob(os.path.join(directory, "*.ckpt"))

    if not ckpt_file:
        log.info(f"No ckpt files found in this directory: {directory}")
        return None

    latest_ckpt_file = max(ckpt_file, key=os.path.getmtime)
    return latest_ckpt_file
