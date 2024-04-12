#include "model.h"
#include <torch/torch.h>

// Define your neural network class
struct EdgeCostModel : torch::nn::Module {
    torch::nn::Sequential model{
        torch::nn::Linear(torch::nn::LinearOptions(5, 5)),
        torch::nn::ReLU(),
        torch::nn::Linear(torch::nn::LinearOptions(5, 5)),
        torch::nn::ReLU(),
        torch::nn::Linear(torch::nn::LinearOptions(5, 1)),
    };

    torch::Tensor forward(torch::Tensor x) {
        return x = model->forward(x);
    }
};

struct TransitionCostModel : torch::nn::Module {
    torch::nn::Sequential model{
        torch::nn::Linear(torch::nn::LinearOptions(5, 5)),
        torch::nn::ReLU(),
        torch::nn::Linear(torch::nn::LinearOptions(5, 5)),
        torch::nn::ReLU(),
        torch::nn::Linear(torch::nn::LinearOptions(5, 1)),
    };
    
    torch::Tensor forward(torch::Tensor x) {
        return x = model->forward(x);
    }
};

std::shared_ptr<EdgeCostModel> e_net_a;
std::shared_ptr<TransitionCostModel> t_net_a;
std::shared_ptr<EdgeCostModel> e_net_b;
std::shared_ptr<TransitionCostModel> t_net_b;

int main() {
    return 0;
}

// Train the cost model
// Declare the train function
void train_models_a(const uint32_t a_c, const uint32_t b_c, const uint32_t a_b) {
    // Create an instance of your neural network
    // MyNetwork net(costing_options.a_b(), costing_options.a_c(), costing_options.b_c());

    if (!e_net_a) {
        // Create an instance of your neural network
        e_net_a = std::make_shared<EdgeCostModel>();
        std::cout << "Instantiated edge cost model A ..." << std::endl;
    }

    if (!t_net_a) {
        // Create an instance of your neural network
        t_net_a = std::make_shared<TransitionCostModel>();
        std::cout << "Instantiated transition cost model A ..." << std::endl;
    }
    
    // // Set your neural network to training mode
    e_net_a->to(torch::kCUDA); // Move the model to GPU
    t_net_a->to(torch::kCUDA); // Move the model to GPU

    // Set your neural network to training mode
    // e_net_a->train();
    // t_net_a->train();
    std::cout << "Training models..." << std::endl;

    // Your training logic here...
}

float e_net_a_forward() {
    if (!e_net_a) {
        // Create an instance of your neural network
        e_net_a = std::make_shared<EdgeCostModel>();
        std::cout << "Instantiated edge cost model A ..." << std::endl;
    }
    
    // Define the size of the tensor
    std::vector<int64_t> sizes = {1, 5}; // Size {1, 5} means 1 row and 5 columns
    torch::Tensor input = torch::ones(sizes, torch::kFloat); // Create a tensor filled with ones

    // Forward pass
    torch::Tensor output = e_net_a->forward(input);
    return e_net_a->forward(input).item().toFloat();
}

float t_net_a_forward() {
    if (!t_net_a) {
        // Create an instance of your neural network
        t_net_a = std::make_shared<TransitionCostModel>();
        std::cout << "Instantiated transition cost model A ..." << std::endl;
    }
    
    // Define the size of the tensor
    std::vector<int64_t> sizes = {1, 5}; // Size {1, 5} means 1 row and 5 columns
    torch::Tensor input = torch::ones(sizes, torch::kFloat); // Create a tensor filled with ones

    // Forward pass
    torch::Tensor output = t_net_a->forward(input);
    return t_net_a->forward(input).item().toFloat();
}

float e_net_b_forward() {
    if (!e_net_b) {
        // Create an instance of your neural network
        e_net_b = std::make_shared<EdgeCostModel>();
        std::cout << "Instantiated edge cost model B ..." << std::endl;
    }
    
    // Define the size of the tensor
    std::vector<int64_t> sizes = {1, 5}; // Size {1, 5} means 1 row and 5 columns
    torch::Tensor input = torch::ones(sizes, torch::kFloat); // Create a tensor filled with ones

    // Forward pass
    torch::Tensor output = e_net_b->forward(input);
    return e_net_b->forward(input).item().toFloat();
}

float t_net_b_forward() {
    if (!t_net_b) {
        // Create an instance of your neural network
        t_net_b = std::make_shared<TransitionCostModel>();
        std::cout << "Instantiated transition cost model B ..." << std::endl;
    }
    
    // Define the size of the tensor
    std::vector<int64_t> sizes = {1, 5}; // Size {1, 5} means 1 row and 5 columns
    torch::Tensor input = torch::ones(sizes, torch::kFloat); // Create a tensor filled with ones

    // Forward pass
    torch::Tensor output = t_net_b->forward(input);
    return t_net_b->forward(input).item().toFloat();
}

