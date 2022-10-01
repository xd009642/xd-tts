use deunicode::deunicode;
use once_cell::sync::OnceCell;
use regex::Regex;

pub fn normalise_text(x: &str) -> String {
    // This regex is just for duplicate pronunciations in CMU dict
    static VERSION_REGEX: OnceCell<Regex> = OnceCell::new();
    let version_regex = VERSION_REGEX.get_or_init(|| Regex::new(r#"\(\d+\)$"#).unwrap());

    let mut s = deunicode(&version_regex.replace_all(x, ""));
    s.retain(valid_char);
    s.make_ascii_uppercase();

    s
}

fn valid_char(x: char) -> bool {
    !r#"!"£$%^&*()-_=+[{]};:'@#~,<.>/?|\`¬"#.contains(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_removal() {
        assert_eq!(normalise_text("BATH(2)"), "BATH");
        assert_eq!(normalise_text("HELLO!(45)"), "HELLO");
        assert_eq!(normalise_text("(3)d"), "3D");
    }
}
