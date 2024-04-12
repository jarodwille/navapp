#ifndef VALHALLA_MODEL_MODEL_H_
#define VALHALLA_MODEL_MODEL_H_
#include <cstdint>

extern bool network_instantiated; // Declare the global variable

// Declare the train function
void train_model(const uint32_t a_c, const uint32_t b_c, const uint32_t a_b);
#endif // VALHALLA_MODEL_MODEL_H_