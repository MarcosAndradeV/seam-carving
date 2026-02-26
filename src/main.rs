use std::cmp::min;

use raylib::prelude::*;
fn main() {
    // let mut args = std::env::args().skip(1);

    // let Some(path) = args.next() else {
    //     eprintln!("Usage: seam <image>");
    //     return;
    // };
    let path = "Broadway_tower_edit.jpg".to_string();

    let mut image = Image::load_image(&path).expect("Failed to load image");
    image.set_format(PixelFormat::PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);

    let (mut rl, thread) = raylib::init()
        .size(image.width, image.height)
        .resizable()
        .title("Seam Carving - Resize Window | Press S to Save")
        .build();

    let mut texture = rl.load_texture_from_image(&thread, &image).unwrap();

    while !rl.window_should_close() {
        if rl.is_key_pressed(KeyboardKey::KEY_S) {
            image.export_image("output.png");
            println!("Saved as output.png");
        }

        if rl.is_key_pressed(KeyboardKey::KEY_C) {
            println!("Begin Carving");
            for _ in 0..60 {
                let mut back_image = image.clone();
                sobel_filter(&mut back_image);
                let dp = compute_dp(&back_image);
                let seams = backtrack_seam(&dp, back_image.width, back_image.height);
                for seam in seams {
                    remove_seam(&mut image, &seam);
                }
            }
            texture = rl.load_texture_from_image(&thread, &image).unwrap();
            println!("End Carving");
        }

        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::BLACK);
        d.draw_texture(&texture, 0, 0, Color::WHITE);
    }
}

fn sobel_filter(image: &mut Image) {
    image.color_grayscale();

    #[rustfmt::skip]
    let sobel_x = &[
        -1.0, 0.0, 1.0,
        -2.0, 0.0, 2.0,
        -1.0, 0.0, 1.0
    ];
    #[rustfmt::skip]
    let sobel_y = &[
        -1.0, -2.0, -1.0,
         0.0,  0.0,  0.0,
         1.0,  2.0,  1.0
    ];
    image.kernel_convolution(sobel_x).unwrap();
    image.kernel_convolution(sobel_y).unwrap();
}

fn compute_dp(image: &Image) -> Vec<u32> {
    let w = image.width as usize;
    let h = image.height as usize;

    let pixel_data_size = image.get_pixel_data_size();
    let mut dp: Vec<u32> = Vec::with_capacity(pixel_data_size);
    unsafe {
        let pixels = image.get_image_data().as_ptr() as *const u32;
        std::ptr::copy_nonoverlapping(pixels, dp.as_mut_ptr(), pixel_data_size);
        dp.set_len(pixel_data_size);
    }

    for y in 1..h {
        for x in 0..w {
            let mut min_above = dp[(y - 1) * w + x];

            if x > 0 {
                min_above = min(min_above, dp[(y - 1) * w + x - 1]);
            }
            if x + 1 < w {
                min_above = min(min_above, dp[(y - 1) * w + x + 1]);
            }

            dp[y * w + x] = dp[y * w + x].saturating_add(min_above);
        }
    }

    dp.to_vec()
}

fn backtrack_seam(dp: &[u32], w: i32, h: i32) -> Vec<Vec<usize>> {
    let w = w as usize;
    let h = h as usize;

    let mut seams = vec![vec![0usize; h]; 10];

    for seam in seams.iter_mut() {
        let mut min_x = 0;
        let mut min_val = dp[(h - 1) * w];

        for x in 1..w {
            let val = dp[(h - 1) * w + x];
            if val < min_val {
                min_val = val;
                min_x = x;
            }
        }

        (*seam)[h - 1] = min_x;

        for y in (0..h - 1).rev() {
            let prev_x = seam[y + 1];

            let mut best_x = prev_x;
            let mut best_val = dp[y * w + prev_x];

            for dx in [-1isize, 0, 1] {
                let nx = prev_x as isize + dx;
                if nx >= 0 && (nx as usize) < w {
                    let val = dp[y * w + nx as usize];
                    if val < best_val {
                        best_val = val;
                        best_x = nx as usize;
                    }
                }
            }

            seam[y] = best_x;
        }
    }
    seams
}

fn remove_seam(image: &mut Image, seam: &[usize]) {
    let w = image.width as usize;
    let h = image.height as usize;

    let data = image.get_image_data();
    let mut new_data = Vec::with_capacity((w - 1) * h);

    for y in 0..h {
        for x in 0..w {
            // new_data.push(Color::RED);
            if x != seam[y] {
                new_data.push(data[y * w + x]);
            }
        }
    }

    *image = unsafe {
        Image::from_raw(ffi::Image {
            data: Box::new(new_data).leak().as_mut_ptr() as *mut _,
            width: w as i32 - 1,
            height: h as i32,
            mipmaps: image.mipmaps,
            format: image.format,
        })
    };
}
