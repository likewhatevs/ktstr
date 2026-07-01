use ktstr::Payload as PayloadDerive;

// A struct named exactly `Payload` would strip-suffix to the empty
// string and produce an unnameable const. The derive macro rejects
// this at expansion time (the behavior this fixture pins). The derive
// macro is aliased purely to keep the derive path visually distinct
// from the deliberately-`Payload`-named struct below.
#[derive(PayloadDerive)]
#[payload(binary = "x")]
#[allow(dead_code)]
struct Payload;

fn main() {}
