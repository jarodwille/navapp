#include <torch/torch.h>
#include <iostream>

// Define the neural network module
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
    auto nn = std::make_shared<EdgeCostModel>();
    nn->to(torch::kCUDA); // Move the model to GPU

    // Optimizer
    torch::optim::Adam optimizer(nn->parameters(), torch::optim::AdamOptions(1e-3));

    // DataLoader
    auto data_loader = torch::data::make_data_loader(
        torch::data::datasets::MNIST("./data").map(
            torch::data::transforms::Stack<>()),
        /*batch_size=*/32);

    // Training loop
    for (size_t epoch = 0; epoch < 10; ++epoch) {
        size_t batch_index = 0;
        for (auto& batch : *data_loader) {
            auto data = batch.data.to(torch::kCUDA), targets = batch.target.to(torch::kCUDA);
            optimizer.zero_grad();
            auto output = net->forward(data);
            auto loss = torch::nn::functional::cross_entropy(output, targets);
            loss.backward();
            optimizer.step();

            std::cout << "Epoch: " << epoch << " | Batch: " << batch_index++ << " | Loss: " << loss.item<float>() << std::endl;
        }
    }

    // Example for inference (simplified)
    torch::Tensor img_tensor; // Assume this is loaded correctly
    img_tensor = img_tensor.unsqueeze(0).to(torch::kCUDA); // Add batch dimension and move to GPU
    auto prediction = net->forward(img_tensor);
    auto pred_label = prediction.argmax(1);
    std::cout << "Predicted label: " << pred_label.item<int>() << std::endl;

    return 0;
}