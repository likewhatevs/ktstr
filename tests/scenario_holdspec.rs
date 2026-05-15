//! [`HoldSpec`] is `Copy` so a single value reuses across multiple
//! [`Step::new`] / [`Step::with_defs`] / [`Step::with_payload`]
//! calls in a construction loop without an explicit `.clone()`. Pin
//! the bound at the type level via a `const fn` (rejects a future
//! revert at every compile, no dev-dep) and at the expression level
//! via use-after-move proofs that would be E0382 without `Copy`.

use std::time::Duration;

use ktstr::scenario::ops::HoldSpec;

const fn requires_copy<T: Copy>() {}
const _: () = requires_copy::<HoldSpec>();

#[test]
fn holdspec_is_copy_across_loop_reuse() {
    for original in [
        HoldSpec::fixed(Duration::from_millis(10)),
        HoldSpec::frac(0.5),
        HoldSpec::loop_at(Duration::from_millis(100)),
        HoldSpec::FULL,
    ] {
        for _ in 0..3 {
            let _hold = original;
            let _again = original;
        }
        match original {
            HoldSpec::Fixed(d) => assert!(d <= Duration::from_secs(1)),
            HoldSpec::Frac(f) => assert!(f > 0.0 && f.is_finite()),
            HoldSpec::Loop { interval } => assert!(!interval.is_zero()),
        }
    }
}

#[test]
fn holdspec_full_reused_in_construction_loop() {
    let template = HoldSpec::FULL;
    let mut holds: Vec<HoldSpec> = Vec::new();
    for _ in 0..5 {
        holds.push(template);
    }
    assert_eq!(holds.len(), 5);
    for h in holds {
        assert!(matches!(h, HoldSpec::Frac(f) if (f - 1.0).abs() < f64::EPSILON));
    }
}
