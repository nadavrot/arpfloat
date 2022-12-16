
#[derive(Debug, Clone, Copy)]
pub enum Sign {
    Positive,
    Negative,
}

impl Sign {
    pub fn is_positive(&self) -> bool {
        match self {
            Sign::Positive => true,
            Sign::Negative => false,
        }
    }
}
