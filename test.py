from tagifai import evaluate
import numpy as np
import logging
import sys

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# # Logging levels (from lowest to highest priority)
# logging.debug("Used for debugging your code.")
# logging.info("Informative messages from your code.")
# logging.warning("Everything works but there is something to be aware of.")
# logging.error("There's been a mistake with the process.")
# logging.critical("There is something terribly wrong and process may terminate.")


def sum_func(a: int, b: int) -> int:
    """_summary_

    Args:
        a (int): a first number
        b (int): a second number

    Returns:
        int:
    """
    c = a + b
    return c


def test_get_metrics():
    y_true = np.array([0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    classes = ["a", "b"]
    performance = evaluate.get_metrics(y_true, y_pred, classes)
    print(performance)


test_get_metrics()
