from tests.test_tensor_transformation import test_index_copy
import tensor_transformation


###
### Hack - run tests without pytest, get normal stack traces
###

if __name__ == "__main__":
    test_index_copy()