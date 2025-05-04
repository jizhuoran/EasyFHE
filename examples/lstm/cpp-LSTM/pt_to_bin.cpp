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

void extract_to_float_array(float *out, const torch::Tensor &t) {
    c10::IntArrayRef t_sizes = t.sizes();
    if (t_sizes.size() > 1) {
        // We have a high dimension tensor, iteratively saving it
        uint64_t chunk_size = 1;
        for (int i = 0;i < t_sizes.size() - 1; ++i) chunk_size *= t_sizes[i + 1];
        for (int i = 0;i < t.size(0); ++i) {
            extract_to_float_array(out + (i * chunk_size), t[i]);
        }
    } else {
        // We have reached the inner most slice
        for (int i = 0;i < t.size(0); ++i) {
            out[i] = t[i].item().toFloat();
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cout << "Usage: pt_to_bin [input.pt] [output.bin]" << std::endl;
    }
    // std::vector<char> f = get_the_bytes("../train/test_input/embedding_batch_0.pt");
    std::vector<char> f;
    torch::IValue x;
    f = get_the_bytes(argv[1]);
    x = torch::pickle_load(f);
    torch::Tensor t = x.toTensor();
    std::cout << "Got Pytorch Tensor size: " << t.sizes() << std::endl;
    // std::cout << "Tensor preview: " << t << std::endl;

    uint64_t tensor_size = 1;
    for (int i = 0;i < t.sizes().size(); ++i) tensor_size *= t.sizes()[i];
    float *output_array = new float[tensor_size];
    extract_to_float_array(output_array, t);

    FILE *file = fopen(argv[2], "wb");
    fwrite(output_array, sizeof(float), tensor_size, file);
    fclose(file);

    delete[] output_array;
    return 0;
}
