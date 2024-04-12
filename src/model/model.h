#ifndef VALHALLA_MODEL_MODEL_H_
#define VALHALLA_MODEL_MODEL_H_
#include <cstdint>

// Declare the train function
void train_models_a(const uint32_t a_c, const uint32_t b_c, const uint32_t a_b);

// run forward on edge cost model a
float e_net_a_forward();

// run forward on transition cost model a
float t_net_a_forward();

// run forward on edge cost model b
float e_net_b_forward();

// run forward on transition cost model b
float t_net_b_forward();

#endif // VALHALLA_MODEL_MODEL_H_