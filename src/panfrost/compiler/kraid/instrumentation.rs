// Copyright © 2026 Pix Philosophy (HK) Limited
// SPDX-License-Identifier: MIT

use crate::ir::*;
use crate::ops::*;
use crate::stats::exec_class;

const METRICS: usize = 6;

fn count_block(model: &dyn Model, block: &BasicBlock) -> [u32; METRICS] {
    let mut counts = [0_u32; METRICS];
    for instr in &block.instrs {
        if matches!(instr.op, Op::Nop(_) | Op::BlendCall(_)) {
            continue;
        }
        if let Some(class) = exec_class(model, &instr.op) {
            counts[class as usize] += 1;
        }
    }
    counts
}

fn atom_metric(instr: &Instr) -> Option<usize> {
    match &instr.op {
        Op::Atom(atom) => Some(usize::from(atom.offset / 8) % METRICS),
        _ => None,
    }
}

fn has_all_metric_atoms(block: &BasicBlock) -> bool {
    let metrics: Vec<usize> =
        block.instrs.iter().filter_map(atom_metric).collect();
    metrics.len() == METRICS && (0..METRICS).all(|m| metrics.contains(&m))
}

impl Shader<'_> {
    pub fn write_instrumentation_counts(&mut self) {
        let Some(pool) = self.constant_pool.as_mut() else {
            return;
        };

        for &(label, offset) in &pool.instrumented_blocks {
            let Some(bi) = self.blocks.iter().position(|b| b.label == label)
            else {
                continue;
            };

            let counts = count_block(self.model, &self.blocks[bi]);
            for (i, count) in counts.iter().enumerate() {
                let at = offset + 4 * i;
                pool.data[at..at + 4].copy_from_slice(&count.to_le_bytes());
            }

            let counter_block = self
                .blocks
                .pred_indices(bi)
                .iter()
                .copied()
                .find(|&p| has_all_metric_atoms(&self.blocks[p]));
            let Some(pi) = counter_block else {
                continue;
            };

            for instr in &mut self.blocks[pi].instrs {
                if atom_metric(instr).is_some_and(|m| counts[m] == 0) {
                    instr.op = Op::Nop(OpNop {});
                }
            }
        }
    }
}
