/*
 * Copyright 2025 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum float8_rounding {
   FLOAT8_ROUND_TO_ZERO,
   FLOAT8_ROUND_UP,
   FLOAT8_ROUND_DOWN,
};

uint8_t _mesa_float_to_e4m3fn(float val);
uint8_t _mesa_float_to_e4m3fn_sat(float val);
uint8_t _mesa_float_to_e4m3fn_round(float val, enum float8_rounding rounding,
                                    bool saturate);
float _mesa_e4m3fn_to_float(uint8_t val);

uint8_t _mesa_float_to_e5m2(float val);
uint8_t _mesa_float_to_e5m2_sat(float val);
uint8_t _mesa_float_to_e5m2_round(float val, enum float8_rounding rounding,
                                  bool saturate);
float _mesa_e5m2_to_float(uint8_t val);

#ifdef __cplusplus
} /* extern C */
#endif
