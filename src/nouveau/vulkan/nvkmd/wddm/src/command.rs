use crate::error::{Error, ErrorKind, Result};

pub struct PushBuffer {
    words: Vec<u32>,
}

impl Default for PushBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl PushBuffer {
    pub fn new() -> Self {
        Self { words: Vec::new() }
    }
    pub fn method(
        &mut self,
        subchannel: u8,
        method: u32,
        values: &[u32],
    ) -> Result<()> {
        if subchannel > 7
            || method % 4 != 0
            || method / 4 >= 8192
            || values.is_empty()
            || values.len() > 8191
        {
            return Err(Error::new(ErrorKind::Invalid, "NVIDIA method packet"));
        }
        self.words.push(
            0x2000_0000
                | (values.len() as u32) << 16
                | u32::from(subchannel) << 13
                | (method >> 2),
        );
        self.words.extend_from_slice(values);
        Ok(())
    }
    pub fn copy_linear(
        &mut self,
        source: u64,
        destination: u64,
        size: u32,
    ) -> Result<()> {
        if source == 0
            || destination == 0
            || size == 0
            || source.checked_add(size.into()).is_none()
            || destination.checked_add(size.into()).is_none()
        {
            return Err(Error::new(ErrorKind::Invalid, "COPY addresses"));
        }
        self.method(
            4,
            0x400,
            &[
                (source >> 32) as u32,
                source as u32,
                (destination >> 32) as u32,
                destination as u32,
                size,
                size,
                size,
                1,
            ],
        )?;
        self.method(4, 0x300, &[0x386])
    }
    pub fn words(&self) -> &[u32] {
        &self.words
    }
    pub fn compute_launch(&mut self, qmd_address: u64) -> Result<()> {
        if qmd_address == 0
            || qmd_address % 256 != 0
            || qmd_address >> 8 > u64::from(u32::MAX)
        {
            return Err(Error::new(ErrorKind::Invalid, "QMD address"));
        }
        self.method(1, 0x110, &[0])?;
        self.method(1, 0x310, &[0x0a04])?;
        self.method(1, 0x288, &[0x0001_0050])?;
        self.method(1, 0x2a0, &[64000, 0])?;
        self.method(1, 0x7b0, &[64001, 0x0100_0000])?;
        self.method(1, 0x298, &[0])?;
        self.method(1, 0x2cc, &[1])?;
        self.method(1, 0x21c, &[0x1011])?;
        self.method(1, 0x2b4, &[(qmd_address >> 8) as u32])?;
        self.method(1, 0x2c0, &[3])?;
        self.method(1, 0x110, &[0])?;
        self.method(1, 0x21c, &[4])
    }
    pub fn bytes(&self) -> Vec<u8> {
        self.words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect()
    }
}
