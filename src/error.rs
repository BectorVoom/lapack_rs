use std::fmt;

/// Errors from the LAPACK-style drivers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LapackError {
    /// An argument is inconsistent (`INFO < 0` in LAPACK).
    InvalidArgument { arg: &'static str, reason: String },
    /// An iteration failed to converge (`INFO > 0` in LAPACK). `info` is the
    /// value the reference routine returns.
    NoConvergence { routine: &'static str, info: usize },
    /// `work` (or `iwork`, or a device budget) is shorter than required.
    WorkspaceTooSmall { need: usize, got: usize },
    /// The device cannot run the requested precision (e.g. no f64).
    DeviceUnsupported { reason: String },
    /// A device runtime error.
    Device(String),
}

impl fmt::Display for LapackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LapackError::InvalidArgument { arg, reason } => write!(f, "invalid argument `{arg}`: {reason}"),
            LapackError::NoConvergence { routine, info } => write!(f, "{routine} failed to converge (info = {info})"),
            LapackError::WorkspaceTooSmall { need, got } => write!(f, "workspace too small: need {need}, got {got}"),
            LapackError::DeviceUnsupported { reason } => write!(f, "device unsupported: {reason}"),
            LapackError::Device(msg) => write!(f, "device error: {msg}"),
        }
    }
}

impl std::error::Error for LapackError {}
