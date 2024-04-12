#include "model.h"
#include <torch/torch.h>

bool network_instantiated = false; // Define the global variable

// Define your neural network class
struct EdgeCostModel : torch::nn::Module {
    torch::nn::Sequential model{
        torch::nn::Linear(torch::nn::LinearOptions(5, 5)),
        torch::nn::ReLU(),
        torch::nn::Linear(torch::nn::LinearOptions(5, 5)),
        torch::nn::ReLU(),
        torch::nn::Linear(torch::nn::LinearOptions(5, 5)),
        torch::nn::ReLU(),
        torch::nn::Softmax(torch::nn::SoftmaxOptions(1))
    };

    torch::Tensor forward(torch::Tensor x) {
        return model->forward(x);
    }
};

int main() {
    // Initialize the network
    auto net = std::make_shared<EdgeCostModel>();
    net->to(torch::kCUDA); // Move the model to GPU

    // Set your neural network to training mode
    net->train();

    // Create an example input tensor
    torch::Tensor input = torch::randn({1, 100});

    // Forward pass
    torch::Tensor output = net->forward(input);

    // Print the output tensor
    std::cout << "Output tensor: " << output << std::endl;

    return 0;
}

// Train the cost model
// Declare the train function
void train_model(const uint32_t a_c, const uint32_t b_c, const uint32_t a_b) {
    // Create an instance of your neural network
    // MyNetwork net(costing_options.a_b(), costing_options.a_c(), costing_options.b_c());

    if (!network_instantiated) {
        // Create an instance of your neural network
        auto net = std::make_shared<EdgeCostModel>();

        // Set the flag to indicate that the network has been instantiated
        network_instantiated = true;

        std::cout << "Instantiated model..." << std::endl;
    }
    // // Set your neural network to training mode
    // net.train();
    std::cout << "Training model..." << std::endl;

    // Your training logic here...
}