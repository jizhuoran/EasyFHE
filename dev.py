import torch.fhe.example.dev_test as dev_test

dev_test.app_example_debug(mode="debug")
dev_test.app_example_release(mode="release")
dev_test.encode_test_case(mode="debug")
dev_test.ct_pt_test_case(mode="debug")

