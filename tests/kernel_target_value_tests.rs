//! Minimal type-construction tests for [`KernelTarget`] and
//! [`KernelValue`]. The variants are pure data carriers with no
//! behaviour beyond `Clone` / `PartialEq` / `Eq` (the runtime
//! invariants — symbol resolution, KVA translation, per-CPU offset
//! lookup — live in the op handler modules). These tests pin the
//! construction surface so a future serde / wire / resolver refactor
//! that breaks the shape fails at the source instead of at the
//! consumer.

use ktstr::prelude::*;

#[test]
fn kernel_target_symbol_clone_eq() {
    let a = KernelTarget::symbol("jiffies_64");
    let b = KernelTarget::symbol("jiffies_64");
    assert_eq!(a, b);
    assert_eq!(a.clone(), b);

    let other = KernelTarget::symbol("other");
    assert_ne!(a, other);

    // Distinct variants with the same backing u64 are not equal.
    let direct = KernelTarget::direct(0);
    assert_ne!(a, direct);
}

#[test]
fn kernel_target_per_cpu_field_constructor_shape() {
    let t = KernelTarget::per_cpu_field("runqueues", "clock", 3);
    let same = KernelTarget::per_cpu_field("runqueues", "clock", 3);
    assert_eq!(t, same);

    // Same symbol+field on a different CPU is a different target —
    // catches a regression where the constructor drops the cpu
    // arg or collapses variants by symbol alone.
    let different_cpu = KernelTarget::per_cpu_field("runqueues", "clock", 7);
    assert_ne!(t, different_cpu);
}

#[test]
fn kernel_value_variants_distinct() {
    // U32(0), U64(0), Bytes(vec![]) all encode "zero" but the
    // typed-variant discriminant must keep them distinguishable —
    // catches a future flattening regression that would let
    // U32(0) compare equal to U64(0).
    assert_ne!(KernelValue::u32(0), KernelValue::u64(0));
    assert_ne!(KernelValue::u64(0), KernelValue::bytes(Vec::<u8>::new()));
    assert_ne!(KernelValue::u32(0), KernelValue::bytes(Vec::<u8>::new()));
}

#[test]
fn kernel_value_bytes_clone_preserves_payload() {
    let v = KernelValue::bytes(vec![1u8, 2, 3]);
    let cloned = v.clone();
    assert_eq!(v, cloned);

    // Different payload of same length must compare unequal so a
    // future Bytes(Arc<...>) refactor that lost per-byte equality
    // surfaces immediately.
    let different = KernelValue::bytes(vec![1u8, 2, 4]);
    assert_ne!(v, different);
}

#[test]
fn op_write_kernel_hot_singleton_constructor_wraps_to_vec() {
    let target = KernelTarget::symbol("test_symbol");
    let value = KernelValue::u64(0xDEAD_BEEF);
    let op = Op::write_kernel_hot(target.clone(), value.clone());
    match op {
        Op::WriteKernelHot { writes } => {
            assert_eq!(writes.len(), 1);
            assert_eq!(writes[0].0, target);
            assert_eq!(writes[0].1, value);
        }
        _ => panic!("expected Op::WriteKernelHot"),
    }
}

#[test]
fn op_write_kernel_cold_batch_preserves_order() {
    let batch = vec![
        (KernelTarget::direct(0x1000), KernelValue::u32(1)),
        (KernelTarget::direct(0x2000), KernelValue::u32(2)),
        (KernelTarget::direct(0x3000), KernelValue::u32(3)),
    ];
    let op = Op::write_kernel_cold_batch(batch.clone());
    match op {
        Op::WriteKernelCold { writes } => {
            assert_eq!(writes.len(), 3);
            for (i, (target, value)) in batch.iter().enumerate() {
                assert_eq!(&writes[i].0, target, "batch order broken at index {i}");
                assert_eq!(&writes[i].1, value, "batch order broken at index {i}");
            }
        }
        _ => panic!("expected Op::WriteKernelCold"),
    }
}

#[test]
fn op_read_kernel_hot_tag_carried() {
    let op = Op::read_kernel_hot("my-tag", KernelTarget::symbol("x"));
    match op {
        Op::ReadKernelHot { tag, target } => {
            assert_eq!(tag, "my-tag");
            assert_eq!(target, KernelTarget::symbol("x"));
        }
        _ => panic!("expected Op::ReadKernelHot"),
    }
}

#[test]
fn op_read_kernel_cold_tag_carried() {
    let op = Op::read_kernel_cold("cold-tag", KernelTarget::kva(0xffff_c900_0000_0000));
    match op {
        Op::ReadKernelCold { tag, target } => {
            assert_eq!(tag, "cold-tag");
            assert_eq!(target, KernelTarget::kva(0xffff_c900_0000_0000));
        }
        _ => panic!("expected Op::ReadKernelCold"),
    }
}
