use xd_tts::training::*;
use tracing::info;

fn main() -> anyhow::Result<()> {
    xd_tts::setup_logging();
    let dictionary = CmuDictionary::open("./data/cmudict-0.7b.txt")?;
    info!("Dictionary size (words): {}", dictionary.len());

    let dataset = lj_speech::Dataset::load("./data/metadata.csv")?;

    let mut analytics = AnalyticsGenerator::new(dictionary);

    for entry in dataset.entries.iter().map(|x| x.text.as_ref()) {
        analytics.push_sentence(entry);
    }
    let report = analytics.generate_report();

    info!("Number of OOV words: {}", report.oov.len());
    info!("Number of diphones: {}", report.diphones.len());
    info!("Number of phones: {}", report.phones.len());

    let report = serde_json::to_string_pretty(&report)?;
    std::fs::write("analysis.json", report)?;

    Ok(())
}
