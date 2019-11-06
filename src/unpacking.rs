use std::fs;

pub fn unpack_labels(filename: &str) -> Vec<u8> {
    let content = fs::read(filename)
        .expect("Something went wrong while reading labels")
        [8..].to_vec();
    content
}
