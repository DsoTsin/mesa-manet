use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ErrorKind {
    Invalid = 1,
    Unsupported = 2,
    HostMemory = 3,
    DeviceMemory = 4,
    DeviceLost = 5,
    Timeout = 6,
    Busy = 7,
    System = 8,
    SurfaceLost = 9,
    OutOfDate = 10,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub kind: ErrorKind,
    pub operation: &'static str,
    pub status: i32,
}

impl Error {
    pub const fn new(kind: ErrorKind, operation: &'static str) -> Self {
        Self {
            kind,
            operation,
            status: 0,
        }
    }

    pub fn from_status(operation: &'static str, status: i32) -> Self {
        let kind = match status as u32 {
            0xC000000D | 0xC0000023 | 0xC0000008 | 0xC01E0106 => {
                ErrorKind::Invalid
            }
            0xC0000017 | 0xC000009A => ErrorKind::HostMemory,
            0xC01E0100 => ErrorKind::DeviceMemory,
            0xC00000BB | 0xC0000002 => ErrorKind::Unsupported,
            0xC00002B6 | 0xC01E0003 | 0xC01E0200 => ErrorKind::DeviceLost,
            0xC01E0102 => ErrorKind::Busy,
            0x00000102 => ErrorKind::Timeout,
            _ => ErrorKind::System,
        };
        Self {
            kind,
            operation,
            status,
        }
    }

    pub fn check(operation: &'static str, status: i32) -> Result<()> {
        if status < 0 {
            Err(Self::from_status(operation, status))
        } else {
            Ok(())
        }
    }

    pub fn vk_result(&self) -> i32 {
        match self.kind {
            ErrorKind::HostMemory => -1,
            ErrorKind::DeviceMemory => -2,
            ErrorKind::DeviceLost => -4,
            ErrorKind::Unsupported => -8,
            ErrorKind::Timeout => 2,
            ErrorKind::Busy => 1,
            ErrorKind::SurfaceLost => -1_000_000_000,
            ErrorKind::OutOfDate => -1_000_001_004,
            ErrorKind::Invalid | ErrorKind::System => -13,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {:?} (NTSTATUS 0x{:08x})",
            self.operation, self.kind, self.status as u32
        )
    }
}

impl std::error::Error for Error {}
