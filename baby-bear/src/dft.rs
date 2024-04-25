
use alloc::vec::Vec;

use icicle_babybear::field::ScalarField;
use icicle_core::ntt::{initialize_domain, ntt_inplace, release_domain, FieldImpl, NTTConfig, NTTDir};
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::memory::HostSlice;
use p3_dft::{Radix2Dit, Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::{AbstractField, Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_util::log2_strict_usize;

use crate::BabyBear;

type F = BabyBear;

/// GPU DFT/NTT for BabyBear with Ingonyama
#[derive(Debug, Default, Clone)]
pub struct BabyBearGpuDft;

impl BabyBearGpuDft {
    /// Compute the DFT of each column of `mat`.
    ///
    /// NB: The DFT works by packing pairs of `Mersenne31` values into
    /// a `Mersenne31Complex` and doing a (half-length) DFT on the
    /// result. In particular, the type of the result elements are in
    /// the extension field, not the domain field.
    pub fn dft_batch(mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        let ctx = DeviceContext::default();

        let h = mat.height();
        let w = mat.width();
        let log_h = log2_strict_usize(h);
        let plonky3_rou = BabyBear::two_adic_generator(log_h);
        // To compute FFTs using icicle, we first need to initialize it using plonky3's "two adic generator"
        initialize_domain(ScalarField::from([plonky3_rou.as_canonical_u32()]), &ctx, false).unwrap();

        let ntt_size = 1 << log_h;

        let mut scalars: Vec<ScalarField> = <ScalarField as FieldImpl>::Config::generate_random(w * ntt_size);
        let scalars_p3: Vec<BabyBear> = scalars
            .iter()
            .map(|x| BabyBear::from_wrapped_u32(Into::<[u32; 1]>::into(*x)[0]))
            .collect();
        let matrix_p3 = RowMajorMatrix::new(scalars_p3, w);

        let mut ntt_cfg: NTTConfig<'_, ScalarField> = NTTConfig::default();
        // Next two lines signalize that we want to compute `nrows` FFTs in column-ordered fashion
        ntt_cfg.batch_size = w as i32;
        ntt_cfg.columns_batch = true;
        ntt_inplace(HostSlice::from_mut_slice(&mut scalars[..]), NTTDir::kForward, &ntt_cfg).unwrap();

        let result_p3 = Radix2DitParallel.dft_batch(matrix_p3);

        for i in 0..w {
            for j in 0..ntt_size {
                assert_eq!(
                    Into::<[u32; 1]>::into(scalars[i + j * w])[0],
                    result_p3.inner.values[i + j * w].as_canonical_u32()
                );
            }
        }

        release_domain::<ScalarField>(&ctx).unwrap();

    }

    // /// Compute the inverse DFT of each column of `mat`.
    // ///
    // /// NB: See comment on `dft_batch()` for information on packing.
    // pub fn idft_batch<Dft: TwoAdicSubgroupDft<C>>(mat: RowMajorMatrix<C>) -> RowMajorMatrix<F> {
    //     let dft = Dft::default();
    //     idft_postprocess(dft.idft_batch(idft_preprocess(mat)))
    // }
}