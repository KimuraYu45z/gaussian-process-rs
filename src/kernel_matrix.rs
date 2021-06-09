use crate::opensrdk_linear_algebra::*;
use opensrdk_kernel_method::*;
use rayon::prelude::*;

pub fn kernel_matrix<T>(
    kernel: &impl Kernel<T>,
    params: &[f64],
    x: &[T],
    x_prime: &[T],
    sigma: f64,
) -> Result<Matrix, KernelError>
where
    T: Value,
{
    let m = x.len();
    let n = x_prime.len();

    let elems = (0..n)
        .into_par_iter()
        .flat_map(|j| {
            (0..m).into_par_iter().map(move |i| {
                Ok(kernel.value(params, &x[i], &x_prime[j])? + if i == j { sigma } else { 0.0 })
            })
        })
        .collect::<Result<Vec<_>, KernelError>>()?;

    let k = Matrix::from(m, elems);

    Ok(k)
}
