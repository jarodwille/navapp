#ifndef VALHALLA_MODEL_MODEL_H_
#define VALHALLA_MODEL_MODEL_H_
#include <cstdint>

// initialize models
void initialize_models();

// Declare the train function
void train_models(const uint32_t a_c, const uint32_t b_c, const uint32_t a_b);

// run forward on edge cost model a
float e_net_a_forward(float length_km, float speed_kph, float sec, float lane_count, float toll, float road_type);

// run forward on transition cost model a
float t_net_a_forward(float has_left, float has_right, float has_slight, float has_sharp, float has_uturn, float has_roundabout, float has_toll);

// run forward on edge cost model b
float e_net_b_forward(float length_km, float speed_kph, float sec, float lane_count, float toll, float road_type);

// run forward on transition cost model b
float t_net_b_forward(float has_left, float has_right, float has_slight, float has_sharp, float has_uturn, float has_roundabout, float has_toll);

#endif // VALHALLA_MODEL_MODEL_H_