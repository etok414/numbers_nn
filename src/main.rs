use nannou::prelude::*;
// use std::thread::sleep;
// use std::time;

pub mod nodes_layers;
pub mod inout;

fn main() {

    nannou::app(model)
        .update(update)
        .run()
}

struct Model {
    _window: WindowId,
    network: nodes_layers::Network,
    is_training: bool,
    images: Vec<Vec<f32>>,
    labels: Vec<Vec<f32>>,
    pos_counter: usize,
    success_counter: usize,
}

fn model(app: &App) -> Model {

    let _window = app
    .new_window()
    .with_dimensions(700, 700)
    .with_title("Simple Neural Network")
    .view(view)
    .build()
    .unwrap();
    let mut network = nodes_layers::Network::new(&[0,0], 0.0);
    let args: Vec<String> = std::env::args().collect();
    let is_training = match args[1].as_str() {
        "training" => true,
        "testing" => false,
        _ => panic!("Invalid parameters"),
    };
    if is_training && args[2] == "new" {
        network = nodes_layers::Network::new(&[28*28, 16, 16, 10], 0.5);
    } else {
        network = inout::read_network(vec![
            r"datas\network1.csv",
            r"datas\network2.csv",
            r"datas\network3.csv",
            ], 0.5).expect("Something went wrong while reading the network");
    }
    let images = if is_training {
            inout::unpack_images(r".\datas\train-images.idx3-ubyte")
        } else {
            inout::unpack_images(r".\datas\t10k-images.idx3-ubyte")
        };
    let labels = if is_training {
            inout::unpack_labels(r".\datas\train-labels.idx1-ubyte")
        } else {
            inout::unpack_labels(r".\datas\t10k-labels.idx1-ubyte")
        };
    let labels = inout::turn_to_result(labels);
    let pos_counter = 0;
    let success_counter = 0;

    Model {
        _window,
        network,
        is_training,
        images,
        labels,
        pos_counter,
        success_counter,
    }
}


fn update(_app: &App, model: &mut Model, _update: Update) {

    if model.network.compare_success(&model.images[model.pos_counter], &model.labels[model.pos_counter], 0.5) {
        model.success_counter += 1;
    }
    if model.is_training {
        model.network.find_make_adjust(&model.images[model.pos_counter], &model.labels[model.pos_counter]);
    }
    model.pos_counter += 1;
    if model.pos_counter % 1000 == 0 {
    // if model.pos_counter % 10000 == 0 {
        println!("{:?} Total successes thus far: {:?}", model.pos_counter, model.success_counter);
    }
    if model.pos_counter >= model.images.len() {
        println!("Number of successes: {:?}", model.success_counter);
        println!("Number of attempts: {:?}", model.images.len());
        if model.is_training {
            let file_paths = vec![
                r"datas\network1.csv",
                r"datas\network2.csv",
                r"datas\network3.csv",
            ];
            inout::write_network(model.network.clone(), file_paths).expect("Something went wrong with writing the data");
            println!("Data successfully written to file!");
        }
    }
    // if 5 > 4 {
    //     println!("time: ");
    //
    //     sleep(time::Duration::new(0, 500_000_000)); // sec, nano sec
    // }
}


fn view(app: &App, model: &Model, frame: &Frame) {

    let draw = app.draw();

    draw_results(model, &draw);

    draw.to_frame(app, frame).unwrap();
}

fn draw_results(model: &Model, draw: &nannou::app::Draw) {

    for y in 0..28 {
        for x in 0..28 {
            let sq_col = model.images[model.pos_counter][y*28 + x];
            draw.rect().x_y(2.0 * x as f32 - 200.0, -2.0 * y as f32 + 50.0).w_h(2.0, 2.0)
                            .color(rgb(sq_col, sq_col, sq_col));
        }
    }
    let values = model.network.calculate(&model.images[model.pos_counter]);
    for y in 0..10 {
        let cir_col = values[values.len()-1][y];
        draw.ellipse().x_y(200.0, 12.0 * y as f32).radius(6.0)
                           .color(rgb(cir_col, cir_col, cir_col));
        let cir_col = model.labels[model.pos_counter][y];
        draw.ellipse().x_y(212.0, 12.0 * y as f32).radius(6.0)
                          .color(rgb(cir_col, cir_col, cir_col));
    }
}
