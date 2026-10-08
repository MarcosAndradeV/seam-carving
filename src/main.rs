use raylib::*;
use std::{ptr, slice};

fn main() {
    unsafe {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let path = args
            .first()
            .cloned()
            .unwrap_or_else(|| "Broadway_tower_edit.jpg".to_string());

        let mut image = LoadImage(cstr!(&path).as_ptr());
        if !IsImageValid(image) {
            panic!("Failed to load image: {}", path);
        }

        ImageResizeNN(&mut image, image.width / 4, image.height / 4);
        ImageFormat(
            &mut image,
            PixelFormat_PIXELFORMAT_UNCOMPRESSED_R8G8B8A8 as _,
        );

        let initial_width = image.width;
        let initial_height = image.height;
        let seams_to_remove = (image.width * 2 / 3) as usize;

        init_window(
            initial_width,
            initial_height,
            cstr!("Seam Carving - Press C to Carve | Press S to Save"),
        );

        let mut texture = LoadTextureFromImage(image);
        if !is_texture_valid(texture) {
            panic!("Failed to load texture");
        }

        while !window_should_close() {
            if is_key_pressed(KEY_C) {
                if image.width > 2 {
                    for _ in 0..seams_to_remove {
                        let energy = compute_energy(&image);
                        let dp = compute_dp(&energy, image.width as usize, image.height as usize);
                        let seam = backtrack_seam(&dp, image.width as usize, image.height as usize);
                        image = remove_seam(image, &seam);
                        println!(
                            "Removed 1 seam. New dimensions: {}x{}",
                            image.width, image.height
                        );
                    }
                    SetWindowSize(image.width, image.height);
                    unload_texture(texture);
                    texture = LoadTextureFromImage(image);
                }
            }

            if is_key_pressed(KEY_D) {
                if image.width > 2 {
                    for _ in 0..seams_to_remove {
                        let energy = compute_energy(&image);
                        let dp = compute_dp(&energy, image.width as usize, image.height as usize);
                        let seam = backtrack_seam(&dp, image.width as usize, image.height as usize);
                        image = insert_seam(image, &seam);
                        println!(
                            "Removed 1 seam. New dimensions: {}x{}",
                            image.width, image.height
                        );
                    }
                    SetWindowSize(image.width, image.height);
                    unload_texture(texture);
                    texture = LoadTextureFromImage(image);
                }
            }

            if is_key_pressed(KEY_S) {
                ExportImage(image, cstr!("output.png").as_ptr());
                println!("Saved output.png");
            }

            begin_drawing();
            clear_background(BLACK);
            DrawTexture(texture, 0, 0, WHITE);
            end_drawing();
        }

        unload_texture(texture);
        UnloadImage(image);
        close_window();
    }
}

/// Returns a slice of Color pixels from an uncompressed R8G8B8A8 image.
fn get_image_colors(image: &Image) -> &[Color] {
    unsafe {
        slice::from_raw_parts(
            image.data as *const Color,
            (image.width * image.height) as usize,
        )
    }
}

/// Computes Sobel gradient magnitude energy map for each pixel.
fn compute_energy(image: &Image) -> Vec<f32> {
    let width = image.width as usize;
    let height = image.height as usize;
    let colors = get_image_colors(image);

    // Compute luminance for each pixel
    let mut lum = vec![0.0f32; width * height];
    for i in 0..colors.len() {
        let c = colors[i];
        lum[i] = 0.2126 * (c.r as f32) + 0.7152 * (c.g as f32) + 0.0722 * (c.b as f32);
    }

    let get_lum = |y: usize, x: usize| -> f32 { lum[y * width + x] };

    let mut energy = vec![0.0f32; width * height];

    for y in 0..height {
        let y_prev = y.saturating_sub(1);
        let y_next = (y + 1).min(height - 1);
        for x in 0..width {
            let x_prev = x.saturating_sub(1);
            let x_next = (x + 1).min(width - 1);

            // 3x3 Sobel kernels
            // Gx kernel:
            // -1  0  1
            // -2  0  2
            // -1  0  1
            let gx = -1.0 * get_lum(y_prev, x_prev) + 1.0 * get_lum(y_prev, x_next)
                - 2.0 * get_lum(y, x_prev)
                + 2.0 * get_lum(y, x_next)
                - 1.0 * get_lum(y_next, x_prev)
                + 1.0 * get_lum(y_next, x_next);

            // Gy kernel:
            // -1 -2 -1
            //  0  0  0
            //  1  2  1
            let gy = -1.0 * get_lum(y_prev, x_prev)
                - 2.0 * get_lum(y_prev, x)
                - 1.0 * get_lum(y_prev, x_next)
                + 1.0 * get_lum(y_next, x_prev)
                + 2.0 * get_lum(y_next, x)
                + 1.0 * get_lum(y_next, x_next);

            energy[y * width + x] = (gx * gx + gy * gy).sqrt();
        }
    }

    energy
}

/// Computes the Dynamic Programming cumulative minimum energy matrix.
fn compute_dp(energy: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut dp = vec![0.0f32; width * height];

    // Top row
    for x in 0..width {
        dp[x] = energy[x];
    }

    // Process row by row
    for y in 1..height {
        let prev_row = (y - 1) * width;
        let curr_row = y * width;
        for x in 0..width {
            let start_x = x.saturating_sub(1);
            let end_x = (x + 1).min(width - 1);

            let mut min_prev = dp[prev_row + start_x];
            for px in (start_x + 1)..=end_x {
                let val = dp[prev_row + px];
                if val < min_prev {
                    min_prev = val;
                }
            }

            dp[curr_row + x] = energy[curr_row + x] + min_prev;
        }
    }

    dp
}

/// Backtracks the minimum energy vertical seam.
fn backtrack_seam(dp: &[f32], width: usize, height: usize) -> Vec<usize> {
    if height == 0 || width == 0 {
        return vec![];
    }
    let mut seam = vec![0usize; height];
    let bottom_y = height - 1;
    let bottom_row = bottom_y * width;

    // Find minimum energy entry in bottom row
    let mut min_x = 0;
    let mut min_val = dp[bottom_row];
    for x in 1..width {
        let val = dp[bottom_row + x];
        if val < min_val {
            min_val = val;
            min_x = x;
        }
    }
    seam[bottom_y] = min_x;

    // Backtrack upwards
    for y in (0..bottom_y).rev() {
        let prev_x = seam[y + 1];
        let curr_row = y * width;

        let start_x = prev_x.saturating_sub(1);
        let end_x = (prev_x + 1).min(width - 1);

        let mut best_x = start_x;
        let mut best_val = dp[curr_row + start_x];

        for x in (start_x + 1)..=end_x {
            let val = dp[curr_row + x];
            if val < best_val {
                best_val = val;
                best_x = x;
            }
        }
        seam[y] = best_x;
    }

    seam
}

/// Removes a vertical seam from the image and returns a new Raylib Image.
fn remove_seam(image: Image, seam: &[usize]) -> Image {
    unsafe {
        let width = image.width as usize;
        let height = image.height as usize;
        if width <= 1 || height == 0 || seam.len() != height {
            return image;
        }
        let src_pixels = slice::from_raw_parts(image.data as *const Color, width * height);

        let new_width = width - 1;
        let mut dest_pixels: Vec<Color> = Vec::with_capacity(new_width * height);

        for y in 0..height {
            let seam_x = seam[y].min(width - 1);
            let row_start = y * width;

            // Copy pixels left of seam
            dest_pixels.extend_from_slice(&src_pixels[row_start..row_start + seam_x]);
            // Copy pixels right of seam
            dest_pixels.extend_from_slice(&src_pixels[row_start + seam_x + 1..row_start + width]);
        }

        // Unload old raylib image data
        UnloadImage(image);

        // Allocate memory for new image using Raylib's allocator
        let byte_count = dest_pixels.len() * std::mem::size_of::<Color>();
        let data_ptr = MemAlloc(byte_count as u32);
        ptr::copy_nonoverlapping(
            dest_pixels.as_ptr() as *const u8,
            data_ptr as *mut u8,
            byte_count,
        );

        Image {
            data: data_ptr,
            width: new_width as i32,
            height: height as i32,
            mipmaps: 1,
            format: PixelFormat_PIXELFORMAT_UNCOMPRESSED_R8G8B8A8 as _,
        }
    }
}

fn insert_seam(image: Image, seam: &[usize]) -> Image {
    unsafe {
        let width = image.width as usize;
        let height = image.height as usize;
        let src_pixels = slice::from_raw_parts(image.data as *const Color, width * height);

        let new_width = width + 1;
        let mut dest_pixels: Vec<Color> = Vec::with_capacity(new_width * height);

        for y in 0..height {
            let seam_x = seam[y].min(width - 1);
            let row_start = y * width;

            // Copy pixels up to the seam
            dest_pixels.extend_from_slice(&src_pixels[row_start..row_start + seam_x + 1]);

            // Interpolate new pixel color with neighbor
            let p1 = src_pixels[row_start + seam_x];
            let p2 = if seam_x + 1 < width {
                src_pixels[row_start + seam_x + 1]
            } else {
                p1
            };
            let interpolated = Color {
                r: ((p1.r as u16 + p2.r as u16) / 2) as u8,
                g: ((p1.g as u16 + p2.g as u16) / 2) as u8,
                b: ((p1.b as u16 + p2.b as u16) / 2) as u8,
                a: p1.a,
            };
            dest_pixels.push(interpolated);

            // Copy remaining pixels in row
            dest_pixels.extend_from_slice(&src_pixels[row_start + seam_x + 1..row_start + width]);
        }

        UnloadImage(image);

        let byte_count = dest_pixels.len() * std::mem::size_of::<Color>();
        let data_ptr = MemAlloc(byte_count as u32);
        ptr::copy_nonoverlapping(
            dest_pixels.as_ptr() as *const u8,
            data_ptr as *mut u8,
            byte_count,
        );

        Image {
            data: data_ptr,
            width: new_width as i32,
            height: height as i32,
            mipmaps: 1,
            format: PixelFormat_PIXELFORMAT_UNCOMPRESSED_R8G8B8A8 as _,
        }
    }
}
