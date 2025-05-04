#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <fstream>

std::vector<char> get_the_bytes(std::string filename) {
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes(
        (std::istreambuf_iterator<char>(input)),
        (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
}

int main() {
    // std::vector<char> f = get_the_bytes("../train/test_input/embedding_batch_0.pt");
    std::vector<char> f;
    torch::IValue x;
    f = get_the_bytes("../train/trained_fc_weight.pt");
    x = torch::pickle_load(f);
    torch::Tensor fc_weight = x.toTensor();
    torch::Tensor fc_weight_t = fc_weight.transpose(0, 1);
    f = get_the_bytes("../train/trained_fc_bias.pt");
    x = torch::pickle_load(f);
    torch::Tensor fc_bias = x.toTensor();

    f = get_the_bytes("../train/trained_rnn_ih.pt");
    x = torch::pickle_load(f);
    torch::Tensor rnn_ih = x.toTensor();
    torch::Tensor rnn_ih_t = rnn_ih.transpose(0, 1);
    f = get_the_bytes("../train/trained_rnn_hh.pt");
    x = torch::pickle_load(f);
    torch::Tensor rnn_hh = x.toTensor();
    torch::Tensor rnn_hh_t = rnn_hh.transpose(0, 1);

    // std::cout << fc_weight << fc_bias << rnn_ih << rnn_hh << std::endl;

    uint32_t n_inference = 0, n_correct = 0;
    for (uint32_t i = 0;i < 196; ++i) {
        f = get_the_bytes(std::string("../train/test_input/embedding_batch_") + std::to_string(i) + std::string(".pt"));
        x = torch::pickle_load(f);
        torch::Tensor embedding = x.toTensor();
        f = get_the_bytes(std::string("../train/test_input/ground_truth_batch_") + std::to_string(i) + std::string(".pt"));
        x = torch::pickle_load(f);
        torch::Tensor ground_truth = x.toTensor();

        // Start evaluating the batch
        for (uint32_t j = 0;j < embedding.size(0); ++j) {
            // Initialize hidden_state as all 0s
            torch::Tensor hidden_state = torch::zeros(128);
            for (uint32_t k = 0; k < embedding[j].size(0); ++k) {
                hidden_state = torch::matmul(hidden_state, rnn_hh_t) + torch::matmul(embedding[j][k], rnn_ih_t);
                hidden_state.tanh_();
            }
            torch::Tensor result = torch::matmul(hidden_state, fc_weight_t) + fc_bias;
            result.sigmoid_();
            bool correct = (result[0].item().toDouble()>0.5 && ground_truth[j].item().toInt()==0) || (result[1].item().toDouble()>0.5 && ground_truth[j].item().toInt()==1);
            // std::cout << result << " " << ground_truth[j] << (correct ? " CORRECT!" : " FAILED!") << std::endl;
            ++ n_inference;
            n_correct += uint32_t(correct);
        }
        printf("succ/total : %5d/%5d  succ rate: %5.2f%%\n", n_correct, n_inference, 100.0 * n_correct / n_inference);
    }
}
