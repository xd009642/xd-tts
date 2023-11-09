use xd_tts::training::prelude::*;

fn main() -> anyhow::Result<()> {
    let dictionary = CmuDictionary::open("./data/cmudict-0.7b.txt")?;

    println!(
        "How to pronounce: {:#?}",
        dictionary.get_pronunciations("apple")
    );
    Ok(())
}
