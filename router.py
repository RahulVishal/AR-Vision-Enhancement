from rp import *
from tv import *
from cb import *
from gaze import gay_ze
from ocr import ocr

def _manage_gaze(frame):
    """
    Internal helper that:
      1. Calls gay_ze(frame) to get a boolean gaze_flag.
      2. Tracks 10 consecutive True → enters OCR mode, runs OCR once, stores that result.
      3. While in OCR mode, returns the stored OCR’d image until 10 consecutive False → exits OCR mode.
      4. If not in OCR mode, returns None.
    """
    # — Initialize static attributes on first call —
    if not hasattr(_manage_gaze, "in_ocr_mode"):
        _manage_gaze.in_ocr_mode = False
        _manage_gaze.true_count = 0
        _manage_gaze.false_count = 0
        _manage_gaze.ocr_frame = None

    gaze_flag = gay_ze(frame)

    if not _manage_gaze.in_ocr_mode:
        # Count consecutive True
        if gaze_flag:
            _manage_gaze.true_count += 1
        else:
            _manage_gaze.true_count = 0

        # Once we hit 10 Trues, switch into OCR mode
        if _manage_gaze.true_count >= 10:
            _manage_gaze.in_ocr_mode = True
            _manage_gaze.false_count = 0
            _manage_gaze.ocr_frame = ocr(frame)  # run OCR exactly once

    else:
        # We are in OCR mode: count consecutive False to exit
        if not gaze_flag:
            _manage_gaze.false_count += 1
        else:
            _manage_gaze.false_count = 0

        if _manage_gaze.false_count >= 10:
            # Exit OCR mode
            _manage_gaze.in_ocr_mode = False
            _manage_gaze.true_count = 0
            _manage_gaze.false_count = 0
            _manage_gaze.ocr_frame = None

    # If currently in OCR mode, return that stored OCR’d image
    if _manage_gaze.in_ocr_mode:
        return _manage_gaze.ocr_frame

    # Otherwise, return None to signal “no special OCR output”
    return None


def router(frame, disease_name, disease_sim=False):
    ocr_result = _manage_gaze(frame)
    if ocr_result is not None:
        return ocr_result

    if disease_name == 'rp':
        return rp_simulate(frame) if disease_sim else rp_heal(frame)
    elif disease_name == 'tv':
        return tv_simulate(frame) if disease_sim else tv_heal(frame)
    elif disease_name == 'cb':
        return cb_simulate(frame) if disease_sim else cb_heal(frame)
    else:
        return frame
