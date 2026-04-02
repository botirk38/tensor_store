use assert_cmd::Command;
use predicates::str::contains;

#[test]
fn convert_help_succeeds() {
    Command::cargo_bin("convert")
        .unwrap()
        .arg("--help")
        .assert()
        .success()
        .stdout(contains("Convert SafeTensors to ServerlessLLM format"));
}

#[test]
fn demo_help_succeeds() {
    Command::cargo_bin("demo")
        .unwrap()
        .arg("--help")
        .assert()
        .success()
        .stdout(contains("Showcase SafeTensors and ServerlessLLM loaders"));
}

#[test]
fn profile_help_succeeds() {
    Command::cargo_bin("profile")
        .unwrap()
        .arg("--help")
        .assert()
        .success()
        .stdout(contains("Profiling harness for tensor_store"));
}

#[test]
fn convert_rejects_zero_partitions() {
    Command::cargo_bin("convert")
        .unwrap()
        .args(["/tmp/does-not-matter", "/tmp/out", "--partitions", "0"])
        .assert()
        .failure()
        .stderr(contains("partition_count must be greater than 0"));
}
