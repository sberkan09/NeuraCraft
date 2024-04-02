#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <string>

class Layer {
public:
    Layer(int input_size, int output_size);

    std::vector<float> forward(const std::vector<float>& input_data);
    std::pair<std::vector<float>, std::string> backward(const std::vector<float>& dloss, const std::string& prev_layer_type);
    void printWeight();

private:
    int input_size;
    int output_size;
    int iterations;
    std::string layer_type;
    std::vector<float> bias;
    std::vector<std::vector<float>> weights;
    std::vector<float> inputs;
    std::vector<float> output;
};

#endif // LAYER_H
