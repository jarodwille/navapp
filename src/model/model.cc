#include "model.h"
#include <torch/torch.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

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
        return model->forward(x);
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
        return model->forward(x);
    }
};

int main() {
    return 0;
}

// edge and transition cost models
std::shared_ptr<EdgeCostModel> e_net_a;
std::shared_ptr<TransitionCostModel> t_net_a;
std::shared_ptr<EdgeCostModel> e_net_b;
std::shared_ptr<TransitionCostModel> t_net_b;

// // historical route submission lists (edgecost) [OUTDATED]
// std::vector<torch::Tensor> route_a_list_e; 
// std::vector<torch::Tensor> route_b_list_e; 
// std::vector<torch::Tensor> route_c_list_e;

// // historical route submission lists (transitioncost) [OUTDATED]
// std::vector<torch::Tensor> route_a_list_t; 
// std::vector<torch::Tensor> route_b_list_t; 
// std::vector<torch::Tensor> route_c_list_t;

// historical route submission lists (edgecost)
std::vector<torch::Tensor> winner_list_e;
std::vector<torch::Tensor> loser_list_e;
std::vector<torch::Tensor> tie_list_1_e;
std::vector<torch::Tensor> tie_list_2_e;

// historical route submission lists (transitioncost)
std::vector<torch::Tensor> winner_list_t;
std::vector<torch::Tensor> loser_list_t;
std::vector<torch::Tensor> tie_list_1_t;
std::vector<torch::Tensor> tie_list_2_t;


// // feedback for each submission [outdated?]
// std::vector<int> a_c_hf;
// std::vector<int> b_c_hf;
// std::vector<int> a_b_hf;

// Function to parse the file and create a tensor
torch::Tensor parseAndCreateTensor(const std::string& filepath) {
    std::ifstream file(filepath);
    std::string line;
    std::vector<float> data;
    int num_edges = 0;

    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        while (getline(ss, value, ',')) {
            data.push_back(stof(value));
        }
        ++num_edges;
    }

    // Assuming there are always 3 values per edge
    int num_features = 3; // length_km, speed, begin_heading

    // Create a tensor from the parsed data
    torch::Tensor tensor = torch::from_blob(data.data(), {num_edges, num_features});
    return tensor.clone(); // Clone the tensor to own its memory
}

void initialize_models() { //TODO:  probably will not be called in train_models_a. should be called before first .forward() route call
    if (!e_net_a) {
        // Create an instance of your neural network
        e_net_a = std::make_shared<EdgeCostModel>();
        e_net_a->to(torch::kCUDA); // move model to GPU
        e_net_a->train(); // training mode
        std::cout << "Instantiated edge cost model A ..." << std::endl;
    }

    if (!t_net_a) {
        // Create an instance of your neural network
        t_net_a = std::make_shared<TransitionCostModel>();
        t_net_a->to(torch::kCUDA); // move model to GPU
        t_net_a->train(); // training mode
        std::cout << "Instantiated transition cost model A ..." << std::endl;
    }

    if (!e_net_b) {
        // Create an instance of your neural network
        e_net_b = std::make_shared<EdgeCostModel>();
        e_net_b->to(torch::kCUDA); // move model to GPU
        e_net_b->train(); // training mode
        std::cout << "Instantiated edge cost model B ..." << std::endl;
    }

    if (!t_net_b) {
        // Create an instance of your neural network
        t_net_b = std::make_shared<TransitionCostModel>();
        t_net_b->to(torch::kCUDA); // move model to GPU
        t_net_b->train(); // training mode
        std::cout << "Instantiated transition cost model B ..." << std::endl;
    }
}

void update_route_lists_e(const uint32_t a_c, const uint32_t b_c, const uint32_t a_b) {
    // {edges, features} tensors of of new routes (edgecost)
    torch::Tensor new_route_a_e = parseAndCreateTensor("./data/route_a_e.txt");
    torch::Tensor new_route_b_e = parseAndCreateTensor("./data/route_b_e.txt");
    torch::Tensor new_route_c_e = parseAndCreateTensor("./data/route_c_e.txt");

    // store routes in them in historical submission lists (edgecost)
    if (a_c == 1) { // a > c
        winner_list_e.push_back(new_route_a_e);
        loser_list_e.push_back(new_route_c_e);
    } else if (a_c == 2) { // a = c
        tie_list_1_e.push_back(new_route_a_e);
        tie_list_2_e.push_back(new_route_c_e);
    } else if (a_c == 3) { // a < c
        winner_list_e.push_back(new_route_c_e);
        loser_list_e.push_back(new_route_a_e);
    } else {
        std::cerr << "Invalid human feedback for a_c" << std::endl;
    }
}

void update_route_lists_t(const uint32_t a_c, const uint32_t b_c, const uint32_t a_b) {
    // {edges, features} tensors of of new routes (transitioncost)
    torch::Tensor new_route_a_t = parseAndCreateTensor("./data/route_a_t.txt");
    torch::Tensor new_route_b_t = parseAndCreateTensor("./data/route_b_t.txt");
    torch::Tensor new_route_c_t = parseAndCreateTensor("./data/route_c_t.txt");

    // store routes in them in historical submission lists (transitioncost)
    if (a_c == 1) { // a > c
        winner_list_t.push_back(new_route_a_t);
        loser_list_t.push_back(new_route_c_t);
    } else if (a_c == 2) { // a = c
        tie_list_1_t.push_back(new_route_a_t);
        tie_list_2_t.push_back(new_route_c_t);
    } else if (a_c == 3) { // a < c
        winner_list_t.push_back(new_route_c_t);
        loser_list_t.push_back(new_route_a_t);
    } else {
        std::cerr << "Invalid human feedback for a_c" << std::endl;
    }
}

torch::Tensor calculate_route_costs(std::vector<torch::Tensor>& route_list_e, std::vector<torch::Tensor>& route_list_t) {
    std::vector<torch::Tensor> route_costs;
    for (int i = 0;  i < route_list_e.size(); ++i) {
        torch::Tensor edge_costs = e_net_a->forward(route_list_e[i]);
        torch::Tensor trans_costs = t_net_a->forward(route_list_t[i]);
        torch::Tensor route_cost = edge_costs.sum() + trans_costs.sum();
        route_costs.push_back(route_cost);
    }
    torch::Tensor costs = torch::stack(route_costs);
    return costs;
}

torch::Tensor calculate_loss() {
    // calculate route costs
    torch::Tensor winner_costs = calculate_route_costs(winner_list_e, winner_list_t);
    torch::Tensor loser_costs = calculate_route_costs(loser_list_e, loser_list_t);
    torch::Tensor tie_costs_1 = calculate_route_costs(tie_list_1_e, tie_list_1_t);
    torch::Tensor tie_costs_2 = calculate_route_costs(tie_list_2_e, tie_list_2_t);
    
    // calculate win-loss bce loss
    auto prob = torch::nn::functional::cross_entropy(winner_costs, loser_costs);
    torch::nn::BCELoss bce_loss(torch::nn::BCELossOptions().reduction(torch::kMean));
    torch::Tensor win_lose_loss = bce_loss(prob, torch::ones_like(prob));

    // calculate tie mse loss
    torch::Tensor tie_loss = torch::mse_loss(tie_costs_1, tie_costs_2, torch::Reduction::Mean);

    float tie_alpha = 1.0;
    torch::Tensor loss = win_lose_loss + tie_alpha*tie_loss;
    return loss;
}

void train_models_a(const uint32_t a_c, const uint32_t b_c, const uint32_t a_b) {

    // initialize_models(); // probably move this to somewhere else
    update_route_lists_e(a_c, b_c, a_b);
    update_route_lists_t(a_c, b_c, a_b);

    // initialize optimizers
    torch::optim::Adam edgeModelOptimizer(e_net_a->parameters(), torch::optim::AdamOptions(1e-3)); 
    torch::optim::Adam transModelOptimizer(t_net_a->parameters(), torch::optim::AdamOptions(1e-3));

    int epochs = 10; // number of training epochs

    // Training loop
    std::cout << "Training models..." << std::endl;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        torch::Tensor loss = calculate_loss();
        std::cout << "Loss" << loss.item().toFloat() << std::endl;
        edgeModelOptimizer.zero_grad();
        transModelOptimizer.zero_grad();

        // Backward propogation
        loss.backward();

        // Step
        edgeModelOptimizer.step();
        transModelOptimizer.step();
    }
    
}


float e_net_a_forward() {
    if (!e_net_a) {
        // Create an instance of your neural network
        e_net_a = std::make_shared<EdgeCostModel>();
        e_net_a->to(torch::kCUDA); // move model to GPU
        e_net_a->train(); // training mode
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
        t_net_a->to(torch::kCUDA); // move model to GPU
        t_net_a->train(); // training mode
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
        e_net_b->to(torch::kCUDA); // move model to GPU
        e_net_b->train(); // training mode
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
        t_net_b->to(torch::kCUDA); // move model to GPU
        t_net_b->train(); // training mode
        std::cout << "Instantiated transition cost model B ..." << std::endl;
    }
    
    // Define the size of the tensor
    std::vector<int64_t> sizes = {1, 5}; // Size {1, 5} means 1 row and 5 columns
    torch::Tensor input = torch::ones(sizes, torch::kFloat); // Create a tensor filled with ones

    // Forward pass
    torch::Tensor output = t_net_b->forward(input);
    return t_net_b->forward(input).item().toFloat();
}

