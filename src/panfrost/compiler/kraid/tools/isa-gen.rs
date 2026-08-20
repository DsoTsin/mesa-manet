// Copyright © 2026 Collabora, Ltd.
// SPDX-License-Identifier: MIT

extern crate kraid_proc;

use std::env;

fn main() {
    let mut args: Vec<String> = env::args().collect();
    // Remove the binary name
    args.remove(0);

    let ts =
        kraid_proc::isa::encoder::gen_encoder(args.clone(), 9..16).unwrap();
    println!("{ts}");

    let ts = kraid_proc::isa::decoder::gen_decoder(args, 9..16).unwrap();
    println!("{ts}");
}
