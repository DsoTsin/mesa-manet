#include <setjmp.h>
#include <string.h>

#include "agc_debug.h"
#include "agc_graphics.h"
#include "agc_register_defaults.h"
#include "test.h"

static jmp_buf fatal_jump;
static char fatal_text[512];

static void
on_fatal(const char *message)
{
   strncpy(fatal_text, message, sizeof(fatal_text) - 1);
   longjmp(fatal_jump, 1);
}

static const char *const host_names[AGC_HOST_COUNT] = {"gfx10.3", "gfx11", "gfx11.5", "gfx12"};

int
main(void)
{
   agc_set_fatal_handler(on_fatal);
   for (unsigned host = 0; host < AGC_HOST_COUNT; host++) {
      CHECK(agc_host_select((enum agc_host)host));
      for (uint32_t version = 12; version <= 13; version++) {
         fatal_text[0] = 0;
         if (setjmp(fatal_jump)) {
            printf("%s v%u: conversion stopped: %s\n", host_names[host], version, fatal_text);
            if (host == AGC_HOST_GFX115)
               CHECK(!"the gfx11.5 default state must convert");
            continue;
         }
         uint32_t counts[AGC_REG_BANK_COUNT];
         uint32_t skipped[AGC_REG_BANK_COUNT];
         for (unsigned bank = 0; bank < AGC_REG_BANK_COUNT; bank++) {
            const struct agc_register *registers = agc_default_state_host(version, (enum agc_reg_bank)bank,
                                                                          &counts[bank]);
            skipped[bank] = agc_default_state_host_skipped(version, (enum agc_reg_bank)bank);
            for (uint32_t i = 1; i < counts[bank]; i++)
               CHECK(registers[i - 1].offset < registers[i].offset);
         }
         printf("%s v%u: host registers cx %u sh %u uc %u; skipped (no rule yet) cx %u sh %u uc %u\n",
                host_names[host], version, counts[0], counts[1], counts[2], skipped[0], skipped[1], skipped[2]);
         if (host == AGC_HOST_GFX115)
            CHECK(counts[AGC_REG_BANK_CX] > 0);
      }
   }
   return TEST_RESULT();
}
