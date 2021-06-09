extern crate blas_src;
extern crate lapack_src;
extern crate opensrdk_kernel_method;
extern crate opensrdk_linear_algebra;
extern crate plotters;
extern crate rayon;

pub mod ey;
pub mod kernel_matrix;

use crate::opensrdk_linear_algebra::*;
use ey::*;
use kernel_matrix::kernel_matrix;
use opensrdk_kernel_method::*;
use plotters::{coord::Shift, prelude::*};
use rand::prelude::*;
use rand_distr::StandardNormal;
use rayon::prelude::*;

fn lkxx<T>(kernel: impl Kernel<T>, x: &Vec<T>, theta: &[f64]) -> Matrix
where
    T: Value,
{
    let kxx = kernel_matrix(&kernel, theta, x, x, 1.0).unwrap();
    let lkxx = kxx.potrf().unwrap();
    lkxx
}

fn exact_gp<T>(
    kernel: impl Kernel<T>,
    y_ey: &[f64],
    ey: f64,
    x: &Vec<T>,
    lkxx: &Matrix,
    xs: T,
    theta: &[f64],
) -> (f64, f64)
where
    T: Value,
{
    let kxsxs = kernel.value(theta, &xs, &xs).unwrap();
    let kxxst = kernel_matrix(&kernel, theta, &[xs], x, 0.0).unwrap();

    let kxxinv_kxxs = lkxx.potrs(kxxst.t()).unwrap();

    let mu = ey + (y_ey.to_vec().row_mat() * &kxxinv_kxxs)[(0, 0)];
    let sigma = kxsxs - (kxxst * &kxxinv_kxxs)[(0, 0)];

    (mu, sigma)
}

fn approx_gp<T>(
    kernel: impl Kernel<T>,
    y_ey: &[f64],
    ey: f64,
    x: &Vec<T>,
    xs: T,
    theta: &[f64],
) -> (f64, f64)
where
    T: Value,
{
    let n = y_ey.len();
    let sigma = 1.0;
    let cii = (0..n)
        .into_par_iter()
        .map(|i| kernel.value(theta, &x[i], &x[i]).unwrap() + sigma)
        .collect::<Vec<_>>();

    let mut mu = 0.0;

    let mut b = vec![false; n];
    let mut l = 0.0;

    let mut p;
    let mut q;
    let mut r = vec![0.0; n];

    let mut p_candidate = (0..n)
        .into_iter()
        .map(|i| kernel.value(theta, &x[i], &xs).unwrap().powi(2) / cii[i])
        .collect::<Vec<_>>();
    let mut q_candidate = p_candidate.clone();

    let mut m = 0;

    while m < n {
        let (j, lj): (usize, f64) = p_candidate
            .par_iter()
            .zip(q_candidate.par_iter())
            .map(|(&pj, &qj)| pj - 2.0 * qj)
            .zip(b.par_iter())
            .enumerate()
            .filter(|(_, (_, &bj))| !bj)
            .map(|(j, (lj, _))| (j, lj))
            .reduce(
                || (0usize, 0.0 / 0.0),
                |(jopt, lopt), (j, lj)| {
                    if lopt.is_nan() || lj < lopt {
                        (j, lj)
                    } else {
                        (jopt, lopt)
                    }
                },
            );

        if lj - l >= 0.0 {
            break;
        }

        let cjj = cii[j];
        let kj = kernel.value(theta, &x[j], &xs).unwrap();
        mu = 1.0 / (m + 1) as f64 * (m as f64 * mu + kj / cjj * y_ey[j]);

        b[j] = true;
        l = lj;

        p = p_candidate[j];
        q = q_candidate[j];

        b.par_iter()
            .zip(p_candidate.par_iter_mut())
            .zip(q_candidate.par_iter_mut())
            .zip(r.par_iter_mut())
            .enumerate()
            .filter(|(_, (((&bi, _), _), _))| !bi)
            .for_each(|(i, (((_, pi), qi), ri))| {
                let cii = cii[i];
                let cij = kernel.value(theta, &x[i], &x[j]).unwrap();
                let ki = kernel.value(theta, &x[i], &xs).unwrap();
                let m = m as f64;

                *ri = *ri + ki * (cij / (cii * cjj)) * kj;

                *pi = 1.0 / (m + 1.0).powi(2) * (m.powi(2) * p + 2.0 * *ri - ki.powi(2) / cii);
                *qi = 1.0 / (m + 1.0) * (m * q + ki.powi(2) / cii);
            });

        m += 1;
    }
    println!("{}", m);

    let kxsxs = kernel.value(theta, &xs, &xs).unwrap();

    let sigma = kxsxs - l;

    (mu + ey, sigma)
}

fn main() {
    println!("Hello, world!");

    draw_gif(true).unwrap();
}

fn func(x: f64) -> f64 {
    0.1 * x + x.sin() + 2.0 * (-x.powi(2)).exp()
}

fn samples(size: usize) -> Vec<(f64, f64)> {
    let mut rng = StdRng::from_seed([1; 32]);
    let mut rng2 = StdRng::from_seed([32; 32]);

    (0..size)
        .into_iter()
        .map(|_| {
            let x = rng2.gen_range(-8.0..=8.0);
            let y = func(x) + rng.sample::<f64, _>(StandardNormal);

            (x, y)
        })
        .collect()
}

fn draw(
    exact: bool,
    size: usize,
    root: &DrawingArea<BitMapBackend, Shift>,
) -> Result<(), Box<dyn std::error::Error>> {
    let samples = samples(size);
    let x = samples.par_iter().map(|v| vec![v.0]).collect::<Vec<_>>();
    let y = samples.par_iter().map(|v| v.1).collect::<Vec<_>>();
    let ey = ey(&y);
    let y_ey = y_ey(&y, ey);
    let kernel = RBF + Periodic;
    let theta = vec![1.0; kernel.params_len()];
    let lkxx = lkxx(kernel, &x, &theta);

    let x_axis = (-8.0..8.0).step(0.1);

    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .set_all_label_area_size(50)
        .build_cartesian_2d(-8.0..8.0, -6.0..6.0)?;

    chart
        .configure_mesh()
        .x_labels(10)
        .y_labels(10)
        .disable_mesh()
        .x_label_formatter(&|v| format!("{:.1}", v))
        .y_label_formatter(&|v| format!("{:.1}", v))
        .draw()?;

    chart.draw_series(LineSeries::new(
        x_axis.values().map(|x| (x, func(x))),
        &GREEN,
    ))?;

    chart.draw_series(PointSeries::of_element(
        samples.iter(),
        2,
        ShapeStyle::from(&BLACK.mix(0.1)).filled(),
        &|&coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
    ))?;

    let gp = if exact {
        x_axis
            .values()
            .map(|xs| {
                let (mu, sigma) = exact_gp(RBF + Periodic, &y_ey, ey, &x, &lkxx, vec![xs], &theta);

                (xs, mu, sigma)
            })
            .collect::<Vec<_>>()
    } else {
        x_axis
            .values()
            .map(|xs| {
                let (mu, sigma) = approx_gp(RBF + Periodic, &y_ey, ey, &x, vec![xs], &theta);

                (xs, mu, sigma)
            })
            .collect::<Vec<_>>()
    };

    // chart.draw_series(AreaSeries::new(
    //     x_axis.values().map(|xs| {
    //         let (mu, sigma) = exact_gp(RBF, &y, &x, &lkxx, vec![xs], &theta);

    //         (xs, mu + sigma)
    //     }),
    //     0.0,
    //     &BLUE.mix(0.5),
    // ))?;
    chart.draw_series(LineSeries::new(
        gp.iter().map(|&(xs, mu, sigma)| (xs, mu + 3.0 * sigma)),
        &RED.mix(0.5),
    ))?;
    chart.draw_series(LineSeries::new(
        gp.iter().map(|&(xs, mu, _)| (xs, mu)),
        &RGBColor(255, 0, 255).mix(0.5),
    ))?;
    chart.draw_series(LineSeries::new(
        gp.iter().map(|&(xs, mu, sigma)| (xs, mu - 3.0 * sigma)),
        &BLUE.mix(0.5),
    ))?;

    root.present()?;

    Ok(())
}

fn draw_png(exact: bool) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(
        if exact {
            "exact_gp.png"
        } else {
            "approx_gp.png"
        },
        (1600, 900),
    )
    .into_drawing_area();

    draw(exact, 1024, &root)?;

    Ok(())
}

fn draw_gif(exact: bool) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::gif(
        if exact {
            "exact_gp.gif"
        } else {
            "approx_gp.gif"
        },
        (1600, 900),
        1_000,
    )?
    .into_drawing_area();

    for k in 0..8 {
        println!("iter: {}", k);
        draw(exact, 2usize.pow(3 + k), &root)?;
    }

    Ok(())
}
