#include "Includes/Layer.h"
#include <cmath>
#include <random>
#include <iostream>

Layer::Layer(int input_size, int output_size)
    : input_size(input_size), output_size(output_size), iterations(0), layer_type("fc") {
    float limit = std::sqrt(6.0f / (input_size + output_size));

    // Randomly initialize weights
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-limit, limit);

    weights.resize(output_size, std::vector<float>(input_size));
    for (auto &row : weights) {
        for (auto &elem : row) {
            elem = dist(gen);
        }
    }

    // Initialize bias with zeros
    bias.resize(output_size, 0.0f);
}

std::vector<float> Layer::forward(const std::vector<float>& input_data) {
    inputs = input_data;
    output.resize(output_size, 0.0f);
    
    for (int i = 0; i < output_size; ++i) {
        output[i] = bias[i];
        for (int j = 0; j < input_size; ++j) {
            output[i] += weights[i][j] * input_data[j];
        }
    }

    return output;
}

std::pair<std::vector<float>, std::string> Layer::backward(const std::vector<float>& dloss, const std::string& prev_layer_type) {
    std::vector<float> dinput(input_size, 0.0f);

    // Backward logic to be properly implemented
    return {dinput, layer_type};
}

void Layer::printWeight() {
    for (auto &row : weights) {
        for (auto &elem : row) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }
}
