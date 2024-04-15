#include "model.h"
#include <torch/torch.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

// Define your neural network class
#include <torch/torch.h>

struct EdgeCostModel : torch::nn::Module { 
    // features: float length_km, float speed_kph, float sec, float lane_count, float road_class, float toll

    // Embedding layer for road type
    torch::nn::Embedding embedding{nullptr};  // Using nullptr to emphasize no default constructor

    // Models for edge cost: model1 for time distance by-pass, model2 is the main model 
    torch::nn::Linear model1{nullptr};
    torch::nn::Sequential model2{nullptr};
    // torch::nn::Linear comb_model{nullptr}; getting rid to speed things up

    int embedding_dim; // output dim for embedding layer

    EdgeCostModel() {
        // Initialize the embedding layer
        int route_types = 8;
        embedding_dim = 2;
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

        // // model to combine models 1 and 2
        // comb_model = register_module("comb_model", torch::nn::Linear(2, 1));
        // comb_model->to(torch::kFloat); // getting rid of this to speed things up
    }

    torch::Tensor forward(torch::Tensor x) {
        // route_type is at the last dimension
        torch::Tensor route_type = x.select(1, x.size(1) - 1); // route type is last features
        torch::Tensor x_other = x.slice(1, 0, x.size(1) - 1); // Extracts all but the last feature

        // Pass route_type through the embedding layer
        torch::Tensor embedded_route_type = embedding->forward(route_type.to(torch::kInt32)); // need int for indexing
        torch::Tensor embedded_route_type_flat = embedded_route_type.view({-1, embedding_dim});

        // Concatenate the embedded output with the other features
        torch::Tensor x_full = torch::cat({x_other, embedded_route_type_flat}, 1);

        // Pass the concatenated features through the rest of the model
        torch::Tensor x_time_distance = x_full.slice(1, 0, 2);
        torch::Tensor comp1 = model1->forward(x_time_distance);
        torch::Tensor comp2 = model2->forward(x_full);

        return comp1 + comp2;
    }
};

struct TransitionCostModel : torch::nn::Module {
    // features: float has_left, float has_right, float has_slight, float has_sharp, float has_uturn, float has_roundabout, float has_toll
    // transition cost model layers
    torch::nn::Sequential model{nullptr};

    TransitionCostModel() {
        int num_total_features = 7;
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
    std::cout << "Parsed tensor from" 
               << filepath 
               << " with shape: {"
               << num_edges
               << ", "
               << num_features
               << "}"
               << std::endl;

    // Create a tensor from the parsed data
    torch::Tensor tensor = torch::from_blob(data.data(), {num_edges, num_features}).clone();
    return tensor.to(torch::kCUDA); // Clone the tensor to own its memory

}

void initialize_models() {
    //TODO: consider loading model_a model_b weights from file
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
    torch::Tensor new_route_a_e = parseAndCreateTensor("/home/jj/thesis/valhalla/src/model/data/route_a_e.txt", num_features);
    torch::Tensor new_route_b_e = parseAndCreateTensor("/home/jj/thesis/valhalla/src/model/data/route_b_e.txt", num_features);
    torch::Tensor new_route_c_e = parseAndCreateTensor("/home/jj/thesis/valhalla/src/model/data/route_c_e.txt", num_features);

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
    
    if (b_c == 1) { // b > c
        winner_list_e.push_back(new_route_b_e);
        loser_list_e.push_back(new_route_c_e);
    } else if (b_c == 2) { // b = c
        tie_list_1_e.push_back(new_route_b_e);
        tie_list_2_e.push_back(new_route_c_e);
    } else if (b_c == 3) { // b < c
        winner_list_e.push_back(new_route_c_e);
        loser_list_e.push_back(new_route_b_e);
    } else {
        std::cerr << "Invalid human feedback for b_c" << std::endl;
    }

    if (a_b == 1) { // a > b
        winner_list_e.push_back(new_route_a_e);
        loser_list_e.push_back(new_route_b_e);
    } else if (a_b == 2) { // a = b
        tie_list_1_e.push_back(new_route_a_e);
        tie_list_2_e.push_back(new_route_b_e);
    } else if (a_b == 3) { // a < b
        winner_list_e.push_back(new_route_b_e);
        loser_list_e.push_back(new_route_a_e);
    } else {
        std::cerr << "Invalid human feedback for a_b" << std::endl;
    }

    std::cout << "Updated human feedback for edges!" << std::endl;
    std::cout << "Winner list size: " << winner_list_e.size() << std::endl;
    std::cout << "Loser list size: " << loser_list_e.size() << std::endl;
    std::cout << "Tie 1 list size: " << tie_list_1_e.size() << std::endl;
    std::cout << "Tie 2 list size: " << tie_list_2_e.size() << std::endl;
}

void update_route_lists_t(const uint32_t a_c, const uint32_t b_c, const uint32_t a_b) {
    int num_features = 7;
    // {edges, features} tensors of of new routes (transitioncost)
    torch::Tensor new_route_a_t = parseAndCreateTensor("/home/jj/thesis/valhalla/src/model/data/route_a_t.txt", num_features);
    torch::Tensor new_route_b_t = parseAndCreateTensor("/home/jj/thesis/valhalla/src/model/data/route_b_t.txt", num_features);
    torch::Tensor new_route_c_t = parseAndCreateTensor("/home/jj/thesis/valhalla/src/model/data/route_c_t.txt", num_features);

    // store routes in them in historical submission lists (edgecost)
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
    
    if (b_c == 1) { // b > c
        winner_list_t.push_back(new_route_b_t);
        loser_list_t.push_back(new_route_c_t);
    } else if (b_c == 2) { // b = c
        tie_list_1_t.push_back(new_route_b_t);
        tie_list_2_t.push_back(new_route_c_t);
    } else if (b_c == 3) { // b < c
        winner_list_t.push_back(new_route_c_t);
        loser_list_t.push_back(new_route_b_t);
    } else {
        std::cerr << "Invalid human feedback for b_c" << std::endl;
    }

    if (a_b == 1) { // a > b
        winner_list_t.push_back(new_route_a_t);
        loser_list_t.push_back(new_route_b_t);
    } else if (a_b == 2) { // a = b
        tie_list_1_t.push_back(new_route_a_t);
        tie_list_2_t.push_back(new_route_b_t);
    } else if (a_b == 3) { // a < b
        winner_list_t.push_back(new_route_b_t);
        loser_list_t.push_back(new_route_a_t);
    } else {
        std::cerr << "Invalid human feedback for a_b" << std::endl;
    }

    std::cout << "Updated human feedback for edges!" << std::endl;
}

torch::Tensor calculate_route_costs(std::vector<torch::Tensor>& route_list_e, std::vector<torch::Tensor>& route_list_t) {
    if (route_list_e.size() != route_list_t.size()) {
        std::cerr << "Route lists for edge and transition costs are not the same size!" << std::endl;
    }
    if (route_list_e.size() == 0) {
        return torch::zeros({1}).to(torch::kCUDA);
    }
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

torch::Tensor calculate_loss(bool print_costs = false) {
    // calculate route costs
    torch::Tensor winner_costs = calculate_route_costs(winner_list_e, winner_list_t);
    torch::Tensor loser_costs = calculate_route_costs(loser_list_e, loser_list_t);
    torch::Tensor tie_costs_1 = calculate_route_costs(tie_list_1_e, tie_list_1_t);
    torch::Tensor tie_costs_2 = calculate_route_costs(tie_list_2_e, tie_list_2_t);
    
    // calculate win-loss bce loss
    torch::Tensor win_lose_loss;
    if (winner_costs.size(0) == 0) {
        win_lose_loss = torch::zeros({1});
        win_lose_loss.to(torch::kCUDA);
    } else {
        // negate and combine to be used in softmax
        torch::Tensor combined_costs = torch::stack({-winner_costs, -loser_costs}, 1);
        torch::Tensor probabilities = torch::softmax(combined_costs, 1);
        torch::Tensor winner_prob = probabilities.select(1, 0);

        torch::nn::BCELoss bce_loss(torch::nn::BCELossOptions().reduction(torch::kMean));
        win_lose_loss = bce_loss(winner_prob, torch::ones_like(winner_prob));
    }

    torch::Tensor tie_loss;
    if (tie_costs_1.size(0) == 0) {
        tie_loss = torch::zeros({1});
        tie_loss.to(torch::kCUDA);
    } else {
        // calculate tie mse loss
        tie_loss = torch::mse_loss(tie_costs_1, tie_costs_2, torch::Reduction::Mean);
    }

    if (print_costs) {
        std::cout << "Winner costs: " << winner_costs << std::endl;
        std::cout << "Loser costs: " << loser_costs << std::endl;
        std::cout << "Tie costs 1: " << tie_costs_1 << std::endl;
        std::cout << "Tie costs 2: " << tie_costs_2 << std::endl;
        std::cout << "Win-lose loss: " << win_lose_loss << std::endl;
        std::cout << "Tie loss: " << tie_loss << std::endl;
    }

    float tie_alpha = 1.0;
    torch::Tensor loss = win_lose_loss + tie_alpha*tie_loss;
    return loss;
}

void train_models(const uint32_t a_c, const uint32_t b_c, const uint32_t a_b) {
    update_route_lists_e(a_c, b_c, a_b);
    update_route_lists_t(a_c, b_c, a_b);

    // set current model b weights to equal previous model a weights
    torch::save(e_net_a, "/home/jj/thesis/valhalla/src/model/weights/e_net_a_weights.pt");
    torch::load(e_net_b, "/home/jj/thesis/valhalla/src/model/weights/e_net_a_weights.pt");

    torch::save(t_net_a, "/home/jj/thesis/valhalla/src/model/weights/t_net_a_weights.pt");
    torch::load(t_net_b, "/home/jj/thesis/valhalla/src/model/weights/t_net_a_weights.pt");

    std::cout << "assigned old A to B weights!" << std::endl;

    // initialize optimizers
    torch::optim::Adam edgeModelOptimizer(e_net_a->parameters(), torch::optim::AdamOptions(5e-3)); 
    torch::optim::Adam transModelOptimizer(t_net_a->parameters(), torch::optim::AdamOptions(5e-3));

    int epochs = 50; // number of training epochs

    // Training loop
    torch::Tensor loss = calculate_loss(true); // true means prints loss of routes
    std::cout << "Starting loss" << loss.item().toFloat() << std::endl;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        loss = calculate_loss();
        edgeModelOptimizer.zero_grad();
        transModelOptimizer.zero_grad();

        // Backward propogation
        loss.backward();

        // Step
        edgeModelOptimizer.step();
        transModelOptimizer.step();
    }
    loss = calculate_loss(true); // true means prints loss of routes
    std::cout << "Final loss loss" << loss.item().toFloat() << std::endl;
}

float e_net_a_forward(float length_km, float speed_kph, float sec, float lane_count, float toll, float road_type) { //TODO: UPDATE ARGUMENTS AND UPDATE AUTOCOST.CC CALLS. THEN FIX ENET B TRAINING LOGIC? STORE WEIGHTS SOMEWHERE?
    if (!e_net_a) {
        // Create an instance of your neural network
        e_net_a = std::make_shared<EdgeCostModel>();
        e_net_a->to(torch::kCUDA); // move model to GPU
        e_net_a->train(); // training mode
        std::cout << "Instantiated edge cost model A inside forward ..." << std::endl;
    }
    
    // create a tensor from input values
    std::vector<float> input_vec = {length_km, speed_kph, sec, lane_count, toll, road_type};
    torch::Tensor input_tensor = torch::from_blob(input_vec.data(), {1, 6}, torch::kFloat32).to(torch::kCUDA);

    // forward pass
    return e_net_a->forward(input_tensor).item().toFloat();
}

float t_net_a_forward(float has_left, float has_right, float has_slight, float has_sharp, float has_uturn, float has_roundabout, float has_toll) {
    if (!t_net_a) {
        // Create an instance of your neural network
        t_net_a = std::make_shared<TransitionCostModel>();
        t_net_a->to(torch::kCUDA); // move model to GPU
        t_net_a->train(); // training mode
        std::cout << "Instantiated transition cost model A inside forward ..." << std::endl;
    }
    
    // create a tensor from input values
    std::vector<float> input_vec = {has_left, has_right, has_slight, has_sharp, has_uturn, has_roundabout, has_toll};
    torch::Tensor input_tensor = torch::from_blob(input_vec.data(), {1, 7}, torch::kFloat32).to(torch::kCUDA);
    
    return t_net_a->forward(input_tensor).item().toFloat();
}

float e_net_b_forward(float length_km, float speed_kph, float sec, float lane_count, float toll, float road_type) {
    if (!e_net_b) {
        // Create an instance of your neural network
        e_net_b = std::make_shared<EdgeCostModel>();
        e_net_b->to(torch::kCUDA); // move model to GPU
        e_net_b->train(); // training mode
        std::cout << "Instantiated edge cost model B inside forward..." << std::endl;
    }
    
    // create a tensor from input values
    std::vector<float> input_vec = {length_km, speed_kph, sec, lane_count, toll, road_type};
    torch::Tensor input_tensor = torch::from_blob(input_vec.data(), {1, 6}, torch::kFloat32).to(torch::kCUDA);

    // forward pass
    return e_net_b->forward(input_tensor).item().toFloat();
}

float t_net_b_forward(float has_left, float has_right, float has_slight, float has_sharp, float has_uturn, float has_roundabout, float has_toll) {
    if (!t_net_b) {
        // Create an instance of your neural network
        t_net_b = std::make_shared<TransitionCostModel>();
        t_net_b->to(torch::kCUDA); // move model to GPU
        t_net_b->train(); // training mode
        std::cout << "Instantiated transition cost model B inside forward..." << std::endl;
    }
    
    // create a tensor from input values
    std::vector<float> input_vec = {has_left, has_right, has_slight, has_sharp, has_uturn, has_roundabout, has_toll};
    torch::Tensor input_tensor = torch::from_blob(input_vec.data(), {1, 7}, torch::kFloat32).to(torch::kCUDA);
    
    return t_net_b->forward(input_tensor).item().toFloat();
}

