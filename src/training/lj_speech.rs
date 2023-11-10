use std::fs::File;
use std::io;
use std::path::Path;

pub struct Entry {
    pub id: String,
    pub text: String,
}

pub struct Dataset {
    pub entries: Vec<Entry>,
}

impl Dataset {
    pub fn load(p: impl AsRef<Path>) -> anyhow::Result<Self> {
        let f = File::open(p)?;
        let reader = io::BufReader::new(f);
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(false)
            .delimiter(b'|')
            .flexible(true)
            .from_reader(reader);

        let mut entries = vec![];

        for result in rdr.records() {
            let record = result?;
            match (record.get(0), record.get(1)) {
                (Some(id), Some(text)) => {
                    entries.push(Entry {
                        id: id.to_string(),
                        text: text.to_string(),
                    });
                }
                _ => println!("Incomplete record: {:?}", record),
            }
        }
        Ok(Self { entries })
    }
}
