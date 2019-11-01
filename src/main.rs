extern crate rand;

use nannou::prelude::*;
use std::thread::sleep;
use std::time;
use rand::Rng;

fn main() {

    nannou::app(model)
        .update(update)
        .run()
}

struct Model {
    _window: WindowId,
}

fn model(app: &App) -> Model {

    let _window = app
    .new_window()
    .with_dimensions(700, 700)
    .with_title("Simple Neural Network")
    .view(view)
    .build()
    .unwrap();

    Model {
        _window
    }
}


fn update(_app: &App, model: &mut Model, _update: Update) {

    //if model.time % 512 < 16 {
        // println!("time: {:?} cost: {:?}", model.time, find_cost(model));
    if 5 > 4 {
        println!("time: ");

        sleep(time::Duration::new(0, 500000000)); // sec, nano sec
    }
}


fn view(app: &App, model: &Model, frame: &Frame) {

    let draw = app.draw();

    draw_results(model, &draw);

    draw.to_frame(app, frame).unwrap();
}

fn draw_results(model: &Model, draw: &nannou::app::Draw) {

    let sq_col = 255.0; // model.input
    let cir_col = 255.0; // model.output

    for y in 0..27 {
        for x in 0..27 {
            draw.rect().x_y(2.0 * x as f32 - 200.0, 2.0 * y as f32 + 50.0).w_h(2.0, 2.0)
                            .color(rgb(sq_col, sq_col, sq_col));
        }
    }
    for y in 0..9 {
        draw.ellipse().x_y(200.0, 10.0 * y as f32).radius(6.0)
                           .color(rgb(cir_col, cir_col, cir_col));
    }
}
