use ktstr::Payload;

#[derive(Payload)]
#[payload(binary = "x", output = Json())]
#[allow(dead_code)]
struct CallFormPayload;

fn main() {}
