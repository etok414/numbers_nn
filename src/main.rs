use nannou::prelude::*;
use std::thread::sleep;
use std::time;

pub mod nodes_layers;
pub mod unpacking;

fn main() {

    nannou::app(model)
        .update(update)
        .run()
}

struct Model {
    _window: WindowId,
    network: nodes_layers::Network,
    training_images: Vec<Vec<f32>>,
    training_labels: Vec<Vec<f32>>,
    counter: usize,
}

fn model(app: &App) -> Model {

    let _window = app
    .new_window()
    .with_dimensions(700, 700)
    .with_title("Simple Neural Network")
    .view(view)
    .build()
    .unwrap();

    let network = nodes_layers::Network::new(vec![28*28, 16, 16, 9], 0.5);
    let training_images = unpacking::unpack_images(r".\datas\train-images.idx3-ubyte");
    let training_labels = unpacking::unpack_labels(r".\datas\train-labels.idx1-ubyte");
    let training_labels = unpacking::turn_to_result(training_labels);
    let counter = 0;

    Model {
        _window,
        network,
        training_images,
        training_labels,
        counter,
    }
}


fn update(_app: &App, model: &mut Model, _update: Update) {

    //if model.time % 512 < 16 {
        // println!("time: {:?} cost: {:?}", model.time, find_cost(model));
    model.network.find_make_adjust(&model.training_images[model.counter], &model.training_labels[model.counter]);
    model.counter += 1;
    if 5 > 4 {
        println!("time: ");

        sleep(time::Duration::new(0, 500_000_000)); // sec, nano sec
    }
}


fn view(app: &App, model: &Model, frame: &Frame) {

    let draw = app.draw();

    draw_results(model, &draw);

    draw.to_frame(app, frame).unwrap();
}

fn draw_results(model: &Model, draw: &nannou::app::Draw) {

    // let sq_col = 255.0; // model.input
    // let cir_col = 255.0; // model.output

    for y in (0..28).rev() {
        for x in 0..28 {
            let sq_col = model.training_images[model.counter][y*28 + x] * 255.0;
            draw.rect().x_y(2.0 * x as f32 - 200.0, 2.0 * y as f32 + 50.0).w_h(2.0, 2.0)
                            .color(rgb(sq_col, sq_col, sq_col));
        }
    }
    for y in 0..9 {
        let cir_col = model.training_labels[model.counter][y] * 255.0;
        draw.ellipse().x_y(200.0, 10.0 * y as f32).radius(6.0)
                           .color(rgb(cir_col, cir_col, cir_col));
    }
}
