#include "model.h"
#include <torch/torch.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

// Define your neural network class
#include <torch/torch.h>

struct EdgeCostModel : torch::nn::Module {
    // Embedding layer for road type
    torch::nn::Embedding embedding{nullptr};  // Using nullptr to emphasize no default constructor

    // Models for edge cost: model1 for time distance by-pass, model2 is the main model 
    torch::nn::Linear model1{nullptr};
    torch::nn::Sequential model2{nullptr};

    EdgeCostModel() {
        // Initialize the embedding layer
        int route_types = 8;
        int embedding_dim = 2;
        embedding = register_module("embedding", torch::nn::Embedding(route_types, embedding_dim));
        embedding->to(torch::kFloat);

        int num_total_features = 6;
        int num_other_features = num_total_features - 1;
        int flattened_size = num_other_features + embedding_dim;

        // Time distance by-pass
        model1 = register_module("model_1", torch::nn::Linear(2, 1));
        model1->to(torch::kFloat);

        int dim2 = 8;
        // Main model for edge cost
        model2 = register_module("model_2", torch::nn::Sequential(
            torch::nn::Linear(flattened_size, dim2),
            torch::nn::ReLU(),
            torch::nn::Linear(dim2, dim2),
            torch::nn::ReLU(),
            torch::nn::Linear(dim2, 1)
        ));
        model2->to(torch::kFloat);
    }

    torch::Tensor forward(torch::Tensor x) {
        // route_type is at the last dimension
        torch::Tensor route_type = x.select(1, x.size(1) - 1); // Extracts the last feature for embedding
        torch::Tensor x_other = x.slice(1, 0, x.size(1) - 1); // Extracts all but the last feature

        // Pass route_type through the embedding layer
        torch::Tensor embedded_route_type = embedding(route_type);
        torch::Tensor embedded_route_type_flat = embedded_route_type.view({-1, embedding->options.num_embeddings()});

        // Concatenate the embedded output with the other features
        torch::Tensor x_full = torch::cat({x_other, embedded_route_type_flat}, 1);

        // Pass the concatenated features through the rest of the model
        torch::Tensor x_time_distance = x_full.slice(1, 0, 2);
        torch::Tensor cost_comp_1 = model1->forward(x_time_distance);
        torch::Tensor cost_comp_2 = model2->forward(x_full);
        return cost_comp_1 + cost_comp_2;
    }
};

struct TransitionCostModel : torch::nn::Module {

    // transition cost model layers
    torch::nn::Sequential model{nullptr};

    TransitionCostModel() {
        int num_total_features = 6;
        int proj_dim = 8;
        // main model for transition cost
        model = register_module("model", torch::nn::Sequential(
            torch::nn::Linear(num_total_features, proj_dim),
            torch::nn::ReLU(),
            torch::nn::Linear(proj_dim, proj_dim),
            torch::nn::ReLU(),
            torch::nn::Linear(proj_dim, 1)
        ));
        model->to(torch::kFloat);
    }

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

// Function to parse the file and create a tensor
torch::Tensor parseAndCreateTensor(const std::string& filepath, int num_features) {
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
    int num_features = 6;
    // {edges, features} tensors of of new routes (edgecost)
    torch::Tensor new_route_a_e = parseAndCreateTensor("./data/route_a_e.txt", num_features);
    torch::Tensor new_route_b_e = parseAndCreateTensor("./data/route_b_e.txt", num_features);
    torch::Tensor new_route_c_e = parseAndCreateTensor("./data/route_c_e.txt", num_features);

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
    int num_features = 6;
    // {edges, features} tensors of of new routes (transitioncost)
    torch::Tensor new_route_a_t = parseAndCreateTensor("./data/route_a_t.txt", num_features);
    torch::Tensor new_route_b_t = parseAndCreateTensor("./data/route_b_t.txt", num_features);
    torch::Tensor new_route_c_t = parseAndCreateTensor("./data/route_c_t.txt", num_features);

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

float e_net_a_forward() { //TODO: UPDATE ARGUMENTS AND UPDATE AUTOCOST.CC CALLS. THEN FIX ENET B TRAINING LOGIC? STORE WEIGHTS SOMEWHERE?
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

