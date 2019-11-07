use std::fs;

pub fn unpack_labels(filename: &str) -> Vec<u8> {
    let content = fs::read(filename)
        .expect("Something went wrong while reading labels")
        [8..].to_vec();
    content
}

pub fn turn_to_result(input: Vec<u8>) -> Vec<Vec<f32>> {
    let mut output = Vec::new();
    for number in input {
        let mut element = vec![0.0;10];
        element[number as usize] = 1.0;
        output.push(element);
    }
    output
}

pub fn unpack_images(filename: &str) -> Vec<Vec<f32>> {
        let raw_content = fs::read(filename)
            .expect("Something went wrong while reading images")
            [16..].to_vec();
        let raw_content = turn_to_float(raw_content);
        let content_as_iter = raw_content.chunks_exact(28*28);
        if content_as_iter.remainder().len() > 0 {
            panic!("An image was only partially formed. It had only {:?} elements, when it should have {:?}", content_as_iter.remainder().len(), 28*28)
        }
        let mut content = Vec::new();
        for chonk in content_as_iter {
            content.push(chonk.to_vec());
        }
        content
}

pub fn turn_to_float(input: Vec<u8>) -> Vec<f32> {
    let mut output = Vec::new();
    for number in input {
        output.push(number as f32 / 255.0)
    }
    output
}
