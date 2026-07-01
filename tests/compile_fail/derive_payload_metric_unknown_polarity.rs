// `#[metric(polarity = Nonsense)]` — an unrecognized polarity ident
// must fail to compile. The macro accepts four of the five `Polarity`
// variants (`HigherBetter`, `LowerBetter`, `Unknown`, and
// `TargetValue(<float>)`); `Informational` is not accepted via the
// attribute, and a bare unrecognized ident is rejected with
// `unknown polarity `<ident>`` so typos surface at compile time.
use ktstr::Payload;

#[derive(Payload)]
#[payload(binary = "metric_bad_polarity_bin")]
#[metric(name = "iops", polarity = Nonsense)]
#[allow(dead_code)]
struct MetricBadPolarityPayload;

fn main() {}
